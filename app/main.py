import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import models  # noqa: F401  (registers tables)
from app.config import get_settings
from app.db import Base, engine
from app.routers import auth, chat, documents, evals, workspaces

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    if get_settings().jwt_secret == "change-me-in-production":
        logging.warning("JWT_SECRET is the default value. Set a long random one in .env before deploying.")
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)

for r in (auth.router, workspaces.router, documents.router, chat.router, evals.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm": f"{s.llm_provider}:{s.llm_model}",
            "fallback": f"{s.llm_fallback_provider}:{s.llm_fallback_model}" if s.llm_fallback_provider else None}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
