from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class AgentProbeError(Exception):
    """Base exception for AgentProbe."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AgentProbeError):
    def __init__(self, resource: str, resource_id: str) -> None:
        super().__init__(
            message=f"{resource} with id '{resource_id}' not found",
            status_code=404,
        )


class ConflictError(AgentProbeError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=409)


class ValidationError(AgentProbeError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=422)


class EvaluationError(AgentProbeError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=500)


class SimulationError(AgentProbeError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=500)


async def agentprobe_error_handler(request: Request, exc: AgentProbeError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": type(exc).__name__},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": "HTTPException"},
    )
