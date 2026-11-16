from enum import StrEnum
from operator import add
from typing import NotRequired, Optional, Any, Annotated

from pydantic import BaseModel, Field

from app.orchestrator.orchestrator_runtime import ParentState

# 保险类别
class InsuranceCategory(StrEnum):
    MEDICAL = "medical"
    CRITICAL_ILLNESS = "critical_illness"
    ACCIDENT = "accident"
    LIFE = "life"

# 被保人信息
class InsuredProfile(BaseModel):
    age: Optional[int] = Field(default=None, description="被保人年龄")
    budget: Optional[float] = Field(default=None, description="年缴保费预算")
    occupation: Optional[str] = Field(default=None, description="职业")

# 保险方案项
class PlanItem(BaseModel):
    product_id: int = Field(description="保险项id")
    recommend_reason: str = Field(description="推荐理由")

# 保险推荐方案
class RecommendationPlan(BaseModel):
    name: str = Field(description="方案简短名称")
    summary: str = Field(description="方案整体说明")
    budget_note: str = Field(description="年缴预算说明")
    plan_items: list[PlanItem] = Field(description="推荐方案中的方案项，也就是具体保险产品")


class RecommendationPlanState(ParentState):
    """推荐子图私有状态，由子图 Checkpointer 跨轮保存。"""

    insured_profile: NotRequired[dict[str, Any]]
    candidate_products: Annotated[list[dict[str, Any]], add]
    plan: NotRequired[dict[str, Any] | None]
