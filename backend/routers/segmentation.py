"""
Общий роутер сегментации — агрегирует информацию о доступных моделях
и предоставляет единую точку входа для сегментации (с авто-выбором архитектуры).
Подключается к основному приложению FastAPI.
"""

from __future__ import annotations
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Query
from starlette.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Literal, Optional

from models.settings import Settings
from models.runtime import SegmentationRuntime
from models.schemas import (
    SegmentRequest,
    SegmentResponse,
    YoloSegmentResponse,
    HealthResponse,
)
from models.utils.read_file import read_image

settings = Settings()
# Загружаем обе архитектуры при старте
_runtimes: dict[str, SegmentationRuntime] = {}

for arch in ("yolo", "segformer"):
    arch_settings = Settings(model_architecture=arch)
    rt = SegmentationRuntime(arch_settings)
    try:
        rt.load()
        _runtimes[arch] = rt
    except Exception as exc:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.warning("%s model failed to load: %s", arch, exc)

router = APIRouter(prefix="/api/v1/segmentation", tags=["General Segmentation"])


def _get_runtime(architecture: str) -> SegmentationRuntime:
    """Возвращает runtime для указанной архитектуры."""
    if architecture not in _runtimes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{architecture}' is not loaded or not available.",
        )
    return _runtimes[architecture]


@router.on_event("shutdown")
async def shutdown_all():
    """Выгружает все модели при остановке сервера."""
    for rt in _runtimes.values():
        rt.unload()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    architecture: Literal["yolo", "segformer", "all"] = Query(
        "all", description="Архитектура для проверки"
    ),
):
    """
    Проверка состояния моделей сегментации.

    Параметры:
    - **architecture**: Какая архитектура проверяется ("yolo", "segformer", "all")
    """
    if architecture == "all":
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(next(iter(_runtimes.values())).device) if _runtimes else "unknown",
            "available_models": list(_runtimes.keys()),
            "models": {
                name: {
                    "loaded": rt.is_loaded,
                    "checkpoint": rt.meta.get("checkpoint_path"),
                }
                for name, rt in _runtimes.items()
            },
        }
    runtime = _get_runtime(architecture)
    return HealthResponse(
        status="healthy" if runtime.is_loaded else "unhealthy",
        model_loaded=runtime.is_loaded,
        device=runtime.device,
        fp16=runtime.meta.get("fp16_active"),
        checkpoint=runtime.meta.get("checkpoint_path"),
    )


@router.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    architecture: Literal["yolo", "segformer"] = Query(
        "yolo", description="Архитектура сегментации"
    ),
    threshold: Optional[float] = None,
    tta: Optional[bool] = None,
    include_mask_png: bool = False,
    include_geojson: Optional[bool] = None,
):
    """
    Единый endpoint сегментации — автоматически выбирает модель
    на основе параметра architecture.

    Параметры:
    - **file**: RGB-изображение (JPEG, PNG, TIFF)
    - **architecture**: Выбор архитектуры: "yolo" или "segformer"
    - **threshold**: Порог уверенности (0.0–1.0)
    - **tta**: Использовать TTA (только YOLO)
    - **include_mask_png**: Включить PNG-маску в ответ
    - **include_geojson**: Включить GeoJSON-представление
    """
    runtime = _get_runtime(architecture)

    try:
        file_bytes = await file.read()
        img = read_image(BytesIO(file_bytes))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    request = SegmentRequest(
        threshold=threshold,
        tta=tta if architecture == "yolo" else None,
        include_mask_png=include_mask_png,
        include_geojson=include_geojson,
    )

    try:
        result = runtime.run_segmentation(img, request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed ({architecture}): {str(e)}",
        )


@router.get("/models")
async def list_available_models():
    """Возвращает информацию о всех загруженных моделях."""
    return {
        "available_models": list(_runtimes.keys()),
        "models": {
            name: {
                "loaded": rt.is_loaded,
                "device": rt.device,
                "checkpoint": rt.meta.get("checkpoint_path"),
                "val_metrics": rt.meta.get("val_metrics"),
            }
            for name, rt in _runtimes.items()
        },
    }


@router.post("/reload")
async def reload_model(
    architecture: Literal["yolo", "segformer", "all"] = Query(
        "all", description="Какая модель перезагружается"
    ),
):
    """Перезагрузка одной или всех моделей."""
    target_archs = ["yolo", "segformer"] if architecture == "all" else [architecture]
    results = {}
    for arch in target_archs:
        try:
            rt = _get_runtime(arch)
            rt.unload()
            rt.load()
            results[arch] = {
                "status": "ok",
                "checkpoint": rt.meta.get("checkpoint_path"),
            }
        except Exception as e:
            results[arch] = {"status": "error", "detail": str(e)}
    return {"results": results}
