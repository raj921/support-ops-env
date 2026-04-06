from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SupportOpsAction, SupportOpsObservation, SupportOpsState
except ImportError:
    from models import SupportOpsAction, SupportOpsObservation, SupportOpsState


class SupportOpsEnv(EnvClient[SupportOpsAction, SupportOpsObservation, SupportOpsState]):

    def _step_payload(self, action: SupportOpsAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[SupportOpsObservation]:
        obs = SupportOpsObservation(**payload.get("observation", {}))
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportOpsState:
        return SupportOpsState(**payload)
