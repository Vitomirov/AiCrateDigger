from fastapi import FastAPI

from app.config import get_settings
from app.routers.search import router as search_router

app = FastAPI(title="AiCrateDigg API", version="0.1.0")
app.include_router(search_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    _ = get_settings()
    return {
        "status": "ok",
        "service": "backend",
    }
