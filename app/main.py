from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.face import router as face_router
from app.api.routes.door import router as door_router


app = FastAPI(
    title="User Authentication API",
    version="1.0.0",
)


# --------------------------------------------------
# Trusted Host Protection
# --------------------------------------------------
#
# "testserver" is required by FastAPI/Starlette's
# TestClient during automated testing.
#
# In production, replace this list with the actual
# API hostname(s).
#

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "testserver",
    ],
)


# --------------------------------------------------
# CORS Security
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
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
# Health Check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "User Authentication API is running"
    }