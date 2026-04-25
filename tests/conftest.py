from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pkg = types.ModuleType("support_ops_env")
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("support_ops_env", pkg)

server_pkg = types.ModuleType("support_ops_env.server")
server_pkg.__path__ = [str(ROOT / "server")]
sys.modules.setdefault("support_ops_env.server", server_pkg)

MODULE_ALIASES = {
    "support_ops_env.models": "models",
    "support_ops_env.tasks": "tasks",
    "support_ops_env.graders": "graders",
    "support_ops_env.client": "client",
    "support_ops_env.inference": "inference",
    "support_ops_env.server.app": "server.app",
    "support_ops_env.server.driftshield_environment": "server.driftshield_environment",
    # Legacy alias kept so external code that still imports the old path keeps working.
    "support_ops_env.server.support_ops_environment": "server.driftshield_environment",
}

for alias, target in MODULE_ALIASES.items():
    sys.modules.setdefault(alias, importlib.import_module(target))
