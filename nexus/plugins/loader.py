"""Discover pluggable agents, critics, and tools for NEXUS."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Callable


@dataclass
class PluginSpec:
    """One discovered plugin registration."""

    name: str
    plugin_type: str
    capabilities: list[str] = field(default_factory=list)
    factory: Callable[..., Any] | None = None
    source: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)


class PluginLoader:
    """Discover plugin registrations from entry points or ~/.nexus/plugins."""

    def __init__(self, plugin_root: Path | None = None):
        self.plugin_root = Path(plugin_root).expanduser() if plugin_root else Path.home() / ".nexus" / "plugins"

    def discover(self) -> list[PluginSpec]:
        specs: list[PluginSpec] = []
        specs.extend(self._discover_entry_points())
        specs.extend(self._discover_local_plugins())
        unique: dict[tuple[str, str], PluginSpec] = {}
        for spec in specs:
            unique[(spec.plugin_type, spec.name)] = spec
        return list(unique.values())

    def _discover_entry_points(self) -> list[PluginSpec]:
        specs: list[PluginSpec] = []
        try:
            entry_points = importlib.metadata.entry_points(group="nexus_plugin")
        except TypeError:
            entry_points = importlib.metadata.entry_points().get("nexus_plugin", [])
        except Exception:
            return specs

        for entry_point in entry_points:
            try:
                loaded = entry_point.load()
            except Exception:
                continue
            spec = self._spec_from_loaded(
                loaded,
                fallback_name=entry_point.name,
                source=f"entry_point:{entry_point.value}",
            )
            if spec:
                specs.append(spec)
        return specs

    def _discover_local_plugins(self) -> list[PluginSpec]:
        root = self.plugin_root
        if not root.exists():
            return []

        specs: list[PluginSpec] = []
        for plugin_dir in root.iterdir():
            if not plugin_dir.is_dir():
                continue
            manifest_path = plugin_dir / "plugin.yaml"
            if not manifest_path.exists():
                continue
            manifest = self._parse_manifest(manifest_path.read_text(encoding="utf-8"))
            module_name = manifest.get("module") or manifest.get("package") or plugin_dir.name
            object_name = manifest.get("entry") or manifest.get("object") or manifest.get("class") or "Plugin"
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                module = importlib.import_module(module_name)
                loaded = getattr(module, object_name, module)
            except Exception:
                continue
            spec = self._spec_from_loaded(
                loaded,
                fallback_name=str(manifest.get("name") or plugin_dir.name),
                source=str(plugin_dir),
                manifest=manifest,
            )
            if spec:
                specs.append(spec)
        return specs

    def _spec_from_loaded(
        self,
        loaded: Any,
        *,
        fallback_name: str,
        source: str,
        manifest: dict[str, Any] | None = None,
    ) -> PluginSpec | None:
        manifest = dict(manifest or getattr(loaded, "manifest", {}) or {})
        plugin_type = str(
            manifest.get("type")
            or getattr(loaded, "plugin_type", "")
            or getattr(loaded, "type", "")
        ).strip().lower()
        if plugin_type not in {"agent", "critic", "tool"}:
            return None

        capabilities = manifest.get("capabilities") or getattr(loaded, "capabilities", []) or []
        if isinstance(capabilities, str):
            capabilities = [part.strip() for part in capabilities.split(",") if part.strip()]

        name = str(
            manifest.get("name")
            or getattr(loaded, "plugin_name", "")
            or getattr(loaded, "name", "")
            or fallback_name
        ).strip()
        if not name:
            return None

        factory = loaded if callable(loaded) else getattr(loaded, "factory", None)
        if factory is None and plugin_type == "tool" and hasattr(loaded, "execute"):
            factory = loaded.__class__
        if factory is None:
            return None

        return PluginSpec(
            name=name,
            plugin_type=plugin_type,
            capabilities=list(capabilities),
            factory=factory,
            source=source,
            manifest=manifest,
        )

    def _parse_manifest(self, raw: str) -> dict[str, Any]:
        data: dict[str, Any] = {}
        current_list_key: str | None = None
        for raw_line in raw.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- ") and current_list_key:
                data.setdefault(current_list_key, []).append(stripped[2:].strip())
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                current_list_key = key
                data.setdefault(key, [])
                continue
            current_list_key = None
            if value.startswith("[") and value.endswith("]"):
                value = [part.strip().strip("'\"") for part in value[1:-1].split(",") if part.strip()]
            else:
                value = value.strip("'\"")
            data[key] = value
        return data
