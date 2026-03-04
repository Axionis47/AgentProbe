from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.core.exceptions import AgentProbeError, agentprobe_error_handler, http_exception_handler
from app.core.logging import setup_logging
from app.core.middleware import RequestIdMiddleware, TimingMiddleware

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(debug=settings.debug)
    logger.info("starting_agentprobe", app_name=settings.app_name, debug=settings.debug)
    yield
    logger.info("shutting_down_agentprobe")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgentProbe API",
        description="Multi-Turn Agent Evaluation Platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware (order matters: last added = first executed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIdMiddleware)

    # Exception handlers
    app.add_exception_handler(AgentProbeError, agentprobe_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]

    # Routes
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()
