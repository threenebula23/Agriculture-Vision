from fastapi import FastAPI
from logger import setup_applevel_logger
from backend.routers.segmentation import router as segmentation_router
from backend.routers.yolo_router import router as yolo_router
from backend.routers.segformer_router import router as segformer_router
from backend.routers.classification_router import router as classification_router

setup_applevel_logger()

app = FastAPI(
    title="Agriculture Vision API",
    description=(
        "Сервис искусственного интеллекта для автоматической сегментации "
        "границ полей, детекции точечных объектов и классификации культур. "
        "Поддерживаются архитектуры YOLO и SegFormer."
    ),
    version="1.1.0",
)


@app.get("/", tags=["Root"])
async def home():
    return {
        "message": "Agriculture Vision API — сервис сегментации аграрных полей",
        "docs": "/docs",
        "version": "1.1.0",
        "available_endpoints": {
            "segmentation": "/api/v1/segmentation/segment",
            "yolo": "/api/v1/yolo/segment",
            "segformer": "/api/v1/segformer/segment",
            "classification": "/api/v1/classification/classify",
        },
    }


app.include_router(segmentation_router)
app.include_router(yolo_router)
app.include_router(segformer_router)
app.include_router(classification_router)
