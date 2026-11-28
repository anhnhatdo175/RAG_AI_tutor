"""FastAPI application for the Moodle LMS integration."""
from __future__ import annotations

import hmac
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import Response

from lmsrag.application import NoLearningContentError
from lmsrag.service.db import ServiceDatabase, create_database
from lmsrag.service.engine import (
    ServiceConflictError,
    ServiceEngine,
    ServiceNotFoundError,
)
from lmsrag.service.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ProfileResponse,
    RecommendRequest,
    RecommendResponse,
    ReviewRequest,
    ReviewResponse,
    SubmitRequest,
    SubmitResponse,
    SyncRequest,
    SyncResponse,
    SyncResult,
    TutorRequest,
    TutorResponse,
)

bearer_scheme = HTTPBearer(auto_error=False)


def create_app(
    *,
    service_engine: ServiceEngine | None = None,
    service_token: str | None = None,
    database_url: str | None = None,
    learner_id_salt: str | None = None,
) -> FastAPI:
    token = service_token or os.getenv("SERVICE_TOKEN") or ""
    salt = learner_id_salt or os.getenv("LEARNER_ID_SALT")
    url = database_url or os.getenv("DATABASE_URL")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if not token or len(token) < 32:
            raise RuntimeError("SERVICE_TOKEN must contain at least 32 characters")
        if not service_engine and not url:
            raise RuntimeError("DATABASE_URL is required")
        if not service_engine and (not salt or len(salt) < 32):
            raise RuntimeError("LEARNER_ID_SALT must contain at least 32 characters")
        owned_database: ServiceDatabase | None = None
        active_engine = service_engine
        try:
            if active_engine is None:
                owned_database = create_database(str(url))
                owned_database.create_schema()
                active_engine = ServiceEngine.from_artifacts(owned_database, str(salt))
            else:
                active_engine.database.create_schema()
            app.state.service_engine = active_engine
            yield
        finally:
            if service_engine is None and active_engine is not None:
                active_engine.close()
            elif owned_database is not None:
                owned_database.close()

    app = FastAPI(title="RAG AI Tutor Moodle Service", version="1.0.0", lifespan=lifespan)
    app.state.service_token = token

    @app.middleware("http")
    async def ensure_decision_log(request: Request, call_next) -> Response:
        started = time.perf_counter()

        def record_fallback(http_status: int) -> None:
            if getattr(request.state, "decision_recorded", False):
                return
            active_engine = getattr(request.app.state, "service_engine", None)
            if active_engine is None:
                return
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            elapsed_ms = (time.perf_counter() - started) * 1000
            active_engine.database.add_decision(
                decision_id=str(uuid.uuid4()),
                learner_hash=None,
                endpoint=endpoint,
                query=None,
                retrieved_chunks=[],
                scores={},
                reasons={"http_status": http_status},
                model=None,
                stage_latency_ms={"request": elapsed_ms},
                latency_ms=elapsed_ms,
            )

        try:
            response = await call_next(request)
        except Exception:
            record_fallback(500)
            raise
        record_fallback(response.status_code)
        return response

    def require_bearer(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        expected = request.app.state.service_token
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(
                credentials.credentials.encode("utf-8"),
                expected.encode("utf-8"),
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid Bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def engine_for(request: Request) -> ServiceEngine:
        active_engine = getattr(request.app.state, "service_engine", None)
        if active_engine is None:
            raise HTTPException(status_code=503, detail="Service is starting")
        return active_engine

    def map_error(exc: Exception) -> HTTPException:
        if isinstance(exc, ServiceNotFoundError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, NoLearningContentError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ServiceConflictError):
            return HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=422, detail=str(exc))
        return HTTPException(status_code=500, detail="Service operation failed")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        active_engine = engine_for(request)
        started = time.perf_counter()
        try:
            active_engine.database.health()
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        active_engine.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=None,
            endpoint="health",
            query=None,
            retrieved_chunks=[],
            scores={},
            reasons={"http_status": 200},
            model=None,
            stage_latency_ms={"database_health": elapsed_ms},
            latency_ms=elapsed_ms,
        )
        request.state.decision_recorded = True
        return HealthResponse(status="ok")

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        dependencies=[Depends(require_bearer)],
    )
    def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
        try:
            recommendations = engine_for(request).recommend(
                body.user_id,
                body.skill_tag,
                body.k,
            )
            request.state.decision_recorded = True
        except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
            raise map_error(exc) from exc
        return RecommendResponse(recommendations=recommendations)

    @app.post(
        "/generate",
        response_model=GenerateResponse,
        dependencies=[Depends(require_bearer)],
    )
    def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        try:
            generated = engine_for(request).generate(
                teacher_id=body.teacher_id,
                course_id=body.course_id,
                skill_tag=body.skill_tag,
                difficulty=body.difficulty,
                count=body.count,
                mode=body.mode,
                subject=body.subject,
                grade=body.grade,
            )
            request.state.decision_recorded = True
        except (
            ServiceNotFoundError,
            ServiceConflictError,
            NoLearningContentError,
            ValueError,
        ) as exc:
            raise map_error(exc) from exc
        return GenerateResponse(tasks=generated)

    @app.post(
        "/tasks/{task_id}/review",
        response_model=ReviewResponse,
        dependencies=[Depends(require_bearer)],
    )
    def review(task_id: str, body: ReviewRequest, request: Request) -> ReviewResponse:
        edit = body.edit.model_dump(exclude_none=True) if body.edit else None
        try:
            task = engine_for(request).review(task_id, action=body.action, edit=edit)
            request.state.decision_recorded = True
        except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
            raise map_error(exc) from exc
        return ReviewResponse(task=task)

    @app.post(
        "/submit",
        response_model=SubmitResponse,
        dependencies=[Depends(require_bearer)],
    )
    def submit(body: SubmitRequest, request: Request) -> SubmitResponse:
        try:
            result = engine_for(request).submit(
                user_id=body.user_id,
                task_id=body.task_id,
                answer=body.answer,
                idempotency_key=body.idempotency_key,
                rt_ms=body.rt_ms,
            )
            request.state.decision_recorded = True
        except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
            raise map_error(exc) from exc
        return SubmitResponse(**result)

    @app.get(
        "/profile/{user_id}",
        response_model=ProfileResponse,
        dependencies=[Depends(require_bearer)],
    )
    def profile(user_id: str, request: Request) -> ProfileResponse:
        try:
            result = engine_for(request).profile(user_id)
            request.state.decision_recorded = True
        except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
            raise map_error(exc) from exc
        return ProfileResponse(**result)

    @app.post(
        "/tutor",
        response_model=TutorResponse,
        dependencies=[Depends(require_bearer)],
    )
    def tutor(body: TutorRequest, request: Request) -> TutorResponse:
        try:
            result = engine_for(request).tutor(
                user_id=body.user_id,
                task_id=body.task_id,
                session_id=body.session_id,
                message=body.message,
            )
            request.state.decision_recorded = True
        except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
            raise map_error(exc) from exc
        return TutorResponse(**result)

    @app.post(
        "/sync",
        response_model=SyncResponse,
        dependencies=[Depends(require_bearer)],
    )
    def sync(body: SyncRequest, request: Request) -> SyncResponse:
        try:
            results = engine_for(request).sync([
                event.model_dump() for event in body.events
            ])
            request.state.decision_recorded = True
        except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
            raise map_error(exc) from exc
        accepted = sum(result["status"] != "rejected" for result in results)
        return SyncResponse(
            accepted=accepted,
            rejected=len(results) - accepted,
            results=[SyncResult(**result) for result in results],
        )

    return app


app = create_app()
