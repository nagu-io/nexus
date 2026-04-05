"""Local canary bundle generation utilities for NEXUS."""

from __future__ import annotations

import hashlib
import secrets
import string
from datetime import UTC, datetime
from urllib.parse import urlparse


def _to_base36(value: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        out.append(chars[rem])
    return "".join(reversed(out))


def _random_base36(length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def derive_canary_password(canary_id: str, *, secret: str, tier: int = 1) -> str:
    """Build a deterministic but plausible password for a canary identity."""
    normalized_secret = str(secret or "").strip() or canary_id
    seed = hashlib.sha256(f"{normalized_secret}:{canary_id}:{max(1, int(tier))}".encode("utf-8")).hexdigest()
    words = [
        "Summer",
        "Football",
        "Bitcoin",
        "Hunter",
        "Freedom",
        "Welcome",
        "Sunset",
        "Thunder",
        "Butterfly",
        "Premium",
    ]
    symbols = ["!", "@", "#", "$", "%", "&"]
    word = words[int(seed[0:2], 16) % len(words)]
    year = 2018 + (int(seed[2:4], 16) % 8)
    symbol = symbols[int(seed[4:6], 16) % len(symbols)]
    return f"{word}{year}{symbol}"


def build_local_seed_bundle(source_url: str, *, secret: str = "nexus-local-canary") -> dict:
    """Build a local 3-tier canary seed plan."""
    now = datetime.now(UTC)
    ts_ms = int(now.timestamp() * 1000)
    parsed = urlparse(source_url)
    host = (parsed.netloc or parsed.path or "local-source").replace("www.", "").strip("/") or "local-source"
    host_slug = host.replace(".", "-")
    canary_fact = f"NEXUS internal note: {host_slug} mirror index '{_to_base36(ts_ms)}' rotates every 47 hours."

    canaries = []
    for tier in (1, 2, 3):
        canary_id = f"{_to_base36(ts_ms + tier - 1)}-{_random_base36(8)}"
        if tier == 3:
            email_local = f"crypto.trader.nexus.{canary_id[:4]}"
            canary_type = "FINANCIAL_LURE"
        elif tier == 2:
            email_local = f"sarah.jones.nexus.{canary_id[:4]}"
            canary_type = "EMAIL_ACCOUNT"
        else:
            email_local = f"james.miller.nexus.{canary_id[:4]}"
            canary_type = "SOCIAL_MEDIA"

        canaries.append(
            {
                "canary_id": canary_id,
                "type": canary_type,
                "tier": tier,
                "email": f"{email_local}@canary.local",
                "password": derive_canary_password(canary_id, secret=secret, tier=tier),
                "tracking_token_preview": f"{canary_id}:{tier}:{host_slug}",
            }
        )

    return {
        "source_url": source_url,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "canary_fact": canary_fact,
        "canaries": canaries,
    }
