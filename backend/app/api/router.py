from fastapi import APIRouter

from app.api.v1 import agent_configs, conversations, eval_runs, evaluations, health, rubrics, scenarios

api_router = APIRouter()

# Health (no prefix)
api_router.include_router(health.router)

# V1 endpoints
api_router.include_router(agent_configs.router)
api_router.include_router(scenarios.router)
api_router.include_router(rubrics.router)
api_router.include_router(eval_runs.router)
api_router.include_router(conversations.router)
api_router.include_router(evaluations.router)
