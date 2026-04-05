"""Internal alert risk scoring for NEXUS canary events."""

from __future__ import annotations


def compute_risk_score(
    abuse_score: int = 0,
    is_tor: bool = False,
    is_proxy: bool = False,
    is_vpn: bool = False,
    country: str = "",
    platform: str = "",
    canary_tier: int = 1,
    breach_count: int = 0,
    recent_attempts: int = 0,
    chain_attack: bool = False,
    virustotal_reputation: int = 0,
    shodan_exposed_ports: int = 0,
    shodan_has_vulns: bool = False,
    shodan_is_scanner: bool = False,
) -> dict:
    """Score a canary alert from low to critical risk."""
    score = 0

    if abuse_score >= 70:
        score += 30
    elif abuse_score >= 40:
        score += 20
    elif abuse_score >= 10:
        score += 10

    if is_tor:
        score += 25
    if is_proxy:
        score += 12
    if is_vpn:
        score += 8

    score += min(max(breach_count, 0) * 4, 16)
    score += min(max(recent_attempts, 0) * 3, 15)

    if chain_attack:
        score += 15

    if virustotal_reputation < 0:
        score += 15
    elif 0 < virustotal_reputation < 25:
        score += 8

    score += min(max(shodan_exposed_ports, 0), 10)
    if shodan_has_vulns:
        score += 12
    if shodan_is_scanner:
        score += 18

    score += {1: 5, 2: 10, 3: 15}.get(max(1, min(3, int(canary_tier or 1))), 5)

    country = (country or "").strip().upper()
    if country and country not in {"US", "IN", "GB", "CA", "AU", "DE"}:
        score += 5

    if platform and "api" in platform.lower():
        score += 4

    score = min(score, 100)
    if score >= 80:
        level = "critical"
    elif score >= 55:
        level = "high"
    elif score >= 30:
        level = "medium"
    else:
        level = "low"

    return {"score": score, "level": level}
