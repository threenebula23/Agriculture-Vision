import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from agro_gis.routers import maps, jobs, models
from agro_gis.services.queue_worker import worker

# Управление жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запускаем фоновый воркер при старте
    worker_task = asyncio.create_task(worker())
    yield
    # При выключении сервера отменяем воркер
    worker_task.cancel()

app = FastAPI(
    title="AgroGIS Backend API",
    description="Бекенд автоматизированной системы детекции полей с реальной очередью задач.",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение модулей (роутер users удален)
app.include_router(maps.router)
app.include_router(jobs.router)
app.include_router(models.router)

@app.get("/")
async def root():
    return {"status": "healthy", "system": "AgroGIS API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)