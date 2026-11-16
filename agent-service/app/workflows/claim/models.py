from enum import StrEnum
from typing import Any, NotRequired

from app.orchestrator.orchestrator_runtime import ParentState


class PolicyStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    ALL = "all"


class ClaimStatus(StrEnum):
    SUBMITTED = "submitted"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"
    ALL = "all"


class ClaimState(ParentState):
    """理赔子图私有状态，由子图 Checkpointer 跨轮保存。"""

    policies: NotRequired[list[dict[str, Any]]]
    claims: NotRequired[list[dict[str, Any]]]
