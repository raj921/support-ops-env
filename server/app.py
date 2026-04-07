from __future__ import annotations

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app

try:
    from ..models import SupportOpsAction, SupportOpsObservation
    from .support_ops_environment import SupportOpsEnvironment
except ImportError:
    from models import SupportOpsAction, SupportOpsObservation
    from server.support_ops_environment import SupportOpsEnvironment

app: FastAPI = create_app(
    SupportOpsEnvironment, SupportOpsAction, SupportOpsObservation,
    env_name="support_ops_env", max_concurrent_envs=8,
)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"name": "support_ops_env", "status": "ok", "docs": "/docs", "reset": "/reset"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        main()
    else:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=8000)
        a = p.parse_args()
        main(host=a.host, port=a.port)
