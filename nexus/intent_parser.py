"""
Intent parser for the NEXUS agent compiler.

Turns a free-form goal into a structured intent profile that the blueprint
generator can compile into an executable workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from nexus.router.mind_router import COMPLEXITY_HIGH_SIGNALS, INTENT_KEYWORDS


@dataclass
class IntentProfile:
    """Structured representation of a user goal."""

    goal: str
    primary_intent: str
    required_agents: list[str]
    complexity: str
    constraints: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class IntentParser:
    """Heuristic goal parser aligned with the existing AEON router keywords."""

    def parse(self, goal: str, project_context: dict | None = None) -> IntentProfile:
        goal = goal.strip()
        lowered = goal.lower()
        scores = {
            intent: sum(1 for keyword in keywords if keyword in lowered)
            for intent, keywords in INTENT_KEYWORDS.items()
        }
        required_agents = [intent for intent, score in scores.items() if score > 0]
        if not required_agents:
            required_agents = ["coding"]

        ranked_agents = sorted(required_agents, key=lambda name: scores.get(name, 0), reverse=True)
        primary_intent = ranked_agents[0]

        # In workspace/project mode, action-oriented tasks MUST route to coding
        # because only the coding agent can produce file tool actions.
        if project_context and project_context.get("enabled"):
            action_verbs = ("fix", "build", "create", "implement", "debug", "refactor",
                            "inspect", "update", "add", "remove", "delete", "edit",
                            "write", "generate", "scaffold", "setup", "install",
                            "migrate", "deploy", "test")
            if any(verb in lowered for verb in action_verbs):
                primary_intent = "coding"
                if "coding" not in ranked_agents:
                    ranked_agents.insert(0, "coding")

        complexity = self._classify_complexity(goal)
        constraints = self._extract_constraints(lowered, project_context=project_context)
        deliverables = self._extract_deliverables(ranked_agents)
        metadata = {
            "scores": scores,
            "word_count": len(goal.split()),
            "file_action": self._classify_file_action(lowered),
            "memory_action": self._classify_memory_action(lowered),
        }
        if project_context:
            metadata["project_mode"] = bool(project_context.get("enabled", False))
            metadata["project_root"] = project_context.get("project_root")
            metadata["project_signature"] = project_context.get("project_signature")
            metadata["project_context"] = dict(project_context)
        return IntentProfile(
            goal=goal,
            primary_intent=primary_intent,
            required_agents=ranked_agents,
            complexity=complexity,
            constraints=constraints,
            deliverables=deliverables,
            metadata=metadata,
        )

    def _classify_complexity(self, goal: str) -> str:
        lowered = goal.lower()
        word_count = len(goal.split())
        score = 0
        if word_count > 50:
            score += 2
        elif word_count > 20:
            score += 1
        if any(signal in lowered for signal in COMPLEXITY_HIGH_SIGNALS):
            score += 2
        if "full stack" in lowered or ("backend" in lowered and "frontend" in lowered):
            score += 2
        if len([word for word in goal.split() if word.lower() in {"and", "then", "after"}]) >= 1:
            score += 1

        if score >= 3:
            return "high"
        if score >= 1:
            return "medium"
        return "low"

    def _extract_constraints(self, lowered: str, project_context: dict | None = None) -> list[str]:
        constraints = []
        if "local" in lowered or "offline" in lowered:
            constraints.append("prefer_local_execution")
        if "json" in lowered:
            constraints.append("structured_output")
        if "safe" in lowered or "safely" in lowered:
            constraints.append("safe_execution")
        if project_context and project_context.get("enabled"):
            constraints.append("respect_existing_project_context")
            if (project_context.get("project_context") or {}).get("frameworks"):
                constraints.append("preserve_detected_frameworks")
        return constraints

    def _extract_deliverables(self, agents: list[str]) -> list[str]:
        deliverable_map = {
            "coding": "implementation guidance or code artifact",
            "research": "grounded summary",
            "memory": "memory recall or persistence result",
            "file": "file-system action result",
            "canary": "canary protection result",
        }
        return [deliverable_map[agent] for agent in agents if agent in deliverable_map]

    def _classify_file_action(self, lowered: str) -> str | None:
        if any(token in lowered for token in ("write", "create", "save")):
            return "write"
        if any(token in lowered for token in ("read", "open", "show")):
            return "read"
        if any(token in lowered for token in ("list", "directory", "folder")):
            return "list"
        return None

    def _classify_memory_action(self, lowered: str) -> str | None:
        if any(token in lowered for token in ("remember", "save", "store", "note")):
            return "store"
        if any(token in lowered for token in ("recall", "history", "previous", "last")):
            return "recall"
        if any(token in lowered for token in ("forget", "delete", "remove")):
            return "forget"
        return None
