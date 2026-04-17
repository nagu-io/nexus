"""Local-first privacy envelope helpers for NEXUS Hive."""

from __future__ import annotations

import base64
import hashlib
import json
from itertools import cycle

from nexus.hive.models import HiveCanaryChallenge, HiveNodeProfile, HiveTaskEnvelope, HiveTaskRequest


class HiveEnvelopeBuilder:
    """
    Build sealed task envelopes for Hive nodes.

    This is intentionally lightweight and local-first. It simulates sealed task
    envelopes without claiming production-grade cryptography.
    """

    def __init__(self, *, secret: str):
        self._secret = secret or "nexus-hive-local-secret"

    def build_task_envelope(self, task: HiveTaskRequest, node: HiveNodeProfile) -> HiveTaskEnvelope:
        """Create a sealed envelope for a real Hive task."""
        return self._build_envelope(task=task, node=node, is_canary=False, canary_id=None, canary_prompt=None)

    def build_canary_envelope(self, task: HiveTaskRequest, challenge: HiveCanaryChallenge) -> HiveTaskEnvelope:
        """Create a sealed envelope for a canary challenge."""
        node = HiveNodeProfile(node_id=challenge.node_id)
        return self._build_envelope(
            task=task,
            node=node,
            is_canary=True,
            canary_id=challenge.challenge_id,
            canary_prompt=challenge.prompt,
        )

    def _build_envelope(
        self,
        *,
        task: HiveTaskRequest,
        node: HiveNodeProfile,
        is_canary: bool,
        canary_id: str | None,
        canary_prompt: str | None,
    ) -> HiveTaskEnvelope:
        task_signature = self._task_signature(task)
        prompt_digest = hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
        masked_context = self._masked_context(task=task, node=node, is_canary=is_canary)
        payload = {
            "task_id": task.task_id,
            "intent": task.intent,
            "prompt": canary_prompt if canary_prompt is not None else task.prompt,
            "required_capabilities": list(task.required_capabilities),
            "latency_budget_ms": task.latency_budget_ms,
            "privacy_mode": task.privacy_mode,
            "is_canary": is_canary,
            "canary_id": canary_id,
        }
        sealed_payload = self._seal(payload)
        envelope_id = hashlib.sha256(f"{task.task_id}:{node.node_id}:{canary_id or 'real'}".encode("utf-8")).hexdigest()[:12]
        return HiveTaskEnvelope(
            envelope_id=envelope_id,
            node_id=node.node_id,
            task_id=task.task_id,
            task_signature=task_signature,
            prompt_digest=prompt_digest,
            masked_context=masked_context,
            sealed_payload=sealed_payload,
            privacy_mode=task.privacy_mode,
            capability_hints=tuple(task.required_capabilities),
            is_canary=is_canary,
            canary_id=canary_id,
        )

    def _task_signature(self, task: HiveTaskRequest) -> str:
        signature_input = json.dumps(
            {
                "task_id": task.task_id,
                "intent": task.intent,
                "capabilities": list(task.required_capabilities),
                "replication_factor": task.replication_factor,
                "max_nodes": task.max_nodes,
                "latency_budget_ms": task.latency_budget_ms,
            },
            sort_keys=True,
        )
        return hashlib.sha256(f"{self._secret}:{signature_input}".encode("utf-8")).hexdigest()

    def _masked_context(self, *, task: HiveTaskRequest, node: HiveNodeProfile, is_canary: bool) -> str:
        words = [segment for segment in task.prompt.strip().split() if segment]
        shape = (
            f"intent={task.intent}; words={len(words)}; "
            f"caps={','.join(task.required_capabilities) or 'none'}; "
            f"mode={task.privacy_mode}; node={node.node_id}; "
            f"canary={'yes' if is_canary else 'no'}"
        )
        if not words:
            return shape
        first = words[0][:12]
        last = words[-1][:12]
        return f"{shape}; prompt_shape={first}...{last}"

    def _seal(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        key = hashlib.sha256(self._secret.encode("utf-8")).digest()
        masked = bytes(byte ^ key_byte for byte, key_byte in zip(raw, cycle(key)))
        return base64.urlsafe_b64encode(masked).decode("ascii")
