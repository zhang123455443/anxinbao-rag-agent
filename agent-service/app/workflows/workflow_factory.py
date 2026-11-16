from .base import Workflow
from .chitchat import ChitchatWorkflow
from app.workflows.claim import ClaimWorkflow
from app.workflows.recommendation.recommendation_plan import RecommendationPlanWorkflow


def create_workflows() -> list[Workflow]:
    return [
        ChitchatWorkflow(),
        RecommendationPlanWorkflow(),
        ClaimWorkflow(),
    ]
