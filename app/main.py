from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.face import router as face_router
from app.api.routes.door import router as door_router
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.rate_limiter import rate_limiter
from app.core.logging_config import configure_logging

configure_logging()


# --------------------------------------------------
# Network Security Configuration
# --------------------------------------------------

allowed_hosts = [
    host.strip()
    for host in settings.ALLOWED_HOSTS.split(",")
    if host.strip()
]

cors_origins = [
    origin.strip()
    for origin in settings.CORS_ORIGINS.split(",")
    if origin.strip()
]


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="User Authentication API",
    version="1.0.0",
)


# --------------------------------------------------
# Rate Limiting
# --------------------------------------------------
#
# The limiter is intentionally registered as application
# middleware so configured endpoint limits are enforced
# for every request.

app.middleware("http")(rate_limiter.middleware)


# --------------------------------------------------
# Trusted Host Protection
# --------------------------------------------------

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts,
)


# --------------------------------------------------
# CORS Security
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Door-Authorization",
    ],
)


# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(auth_router)
app.include_router(face_router)
app.include_router(door_router)


# --------------------------------------------------
# Health Checks
# --------------------------------------------------

@app.get("/health")
def health_check():
    """
    Lightweight liveness check.

    Confirms that the FastAPI application is running.
    Does not depend on the database.
    """
    return {
        "status": "ok"
    }


@app.get("/health/ready")
def readiness_check():
    """
    Readiness check.

    Confirms that the application can communicate
    with PostgreSQL.
    """
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))

        return {
            "status": "ready"
        }

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        )

    finally:
        db.close()


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "User Authentication API is running"
    }