"""Response synthesis helpers for NEXUS Hive."""

from __future__ import annotations

from nexus.hive.models import HiveAssemblyResult, HiveCandidateAssessment, HiveTaskRequest, clamp01


class HiveResponseAssembler:
    """Merge the strongest Hive candidates into one higher-signal answer."""

    def assemble(
        self,
        *,
        task: HiveTaskRequest,
        candidates: tuple[HiveCandidateAssessment, ...],
    ) -> HiveAssemblyResult:
        """Build a synthesized answer from the top non-blocked candidates."""
        if not candidates:
            return HiveAssemblyResult(
                output="NEXUS Hive could not assemble a response because no trusted candidates survived scoring.",
                source_nodes=(),
                confidence=0.0,
                note="no_viable_candidates",
            )

        source_nodes = tuple(candidate.node_id for candidate in candidates)
        snippets = self._collect_unique_snippets(candidates)
        if not snippets:
            snippets = [candidate.output.strip() for candidate in candidates if candidate.output.strip()]

        intro = (
            f"Hive synthesis for {task.intent}: "
            f"assembled from {len(source_nodes)} trusted candidate{'s' if len(source_nodes) != 1 else ''}."
        )
        body_lines = [intro]
        for snippet in snippets[:4]:
            body_lines.append(f"- {snippet}")

        confidence = clamp01(sum(candidate.network_score for candidate in candidates) / len(candidates))
        note = None
        if any(candidate.reflect_action == "warn" for candidate in candidates):
            note = "includes_medium_risk_inputs"
            body_lines.append("- Validation: some contributing nodes were medium risk, so verify integration details locally.")

        return HiveAssemblyResult(
            output="\n".join(body_lines),
            source_nodes=source_nodes,
            confidence=confidence,
            note=note,
        )

    def _collect_unique_snippets(self, candidates: tuple[HiveCandidateAssessment, ...]) -> list[str]:
        seen: set[str] = set()
        snippets: list[str] = []
        for candidate in candidates:
            for raw in candidate.output.replace("\r", "\n").split("\n"):
                line = raw.strip(" -\t")
                if not line:
                    continue
                normalized = " ".join(line.lower().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                snippets.append(line)
                break
        return snippets
