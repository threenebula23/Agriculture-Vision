"""
Роутер для модели SegFormer.
Предоставляет endpoint для семантической сегментации полей.
Подключается к основному приложению FastAPI.
"""

from __future__ import annotations
from typing import Optional
from io import BytesIO
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from starlette.responses import JSONResponse, Response

from models.settings import Settings
from models.runtime import SegmentationRuntime
from models.schemas import (
    SegmentRequest,
    SegmentResponse,
    HealthResponse,
)
from models.utils.read_file import read_image
from models.utils.visualize import draw_mask_overlay, draw_polygons, encode_png

settings = Settings(model_architecture="segformer")
runtime = SegmentationRuntime(settings)
try:
    runtime.load()
except Exception as exc:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.warning("SegFormer model failed to load at startup: %s", exc)

router = APIRouter(prefix="/api/v1/segformer", tags=["SegFormer Segmentation"])


@router.on_event("shutdown")
async def shutdown_segformer():
    """Выгружает модель при остановке сервера."""
    runtime.unload()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния SegFormer-модели."""
    return HealthResponse(
        status="healthy" if runtime.is_loaded else "unhealthy",
        model_loaded=runtime.is_loaded,
        device=runtime.device,
        fp16=runtime.meta.get("fp16_active"),
        checkpoint=runtime.meta.get("checkpoint_path"),
    )


@router.post("/segment", response_model=SegmentResponse)
async def segment_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = None,
    include_mask_png: bool = False,
    include_geojson: Optional[bool] = None,
):
    """
    Семантическая сегментация изображения с помощью SegFormer.

    Параметры:
    - **file**: RGB-изображение (JPEG, PNG, TIFF)
    - **threshold**: Порог уверенности для бинаризации маски (0.0–1.0)
    - **include_mask_png**: Включить PNG-маску в ответ (base64)
    - **include_geojson**: Включить GeoJSON-представление результатов

    Возвращает:
    - Полигон сегментированной области, метрики и опционально маску/GeoJSON
    """
    if not runtime.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SegFormer model is not loaded. Check server logs.",
        )

    try:
        file_bytes = await file.read()
        img_bgr = read_image(BytesIO(file_bytes))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    request = SegmentRequest(
        threshold=threshold,
        include_mask_png=include_mask_png,
        include_overlay=include_mask_png,
        include_geojson=include_geojson,
    )

    try:
        result = runtime.run_segmentation(rgb, request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}",
        )


@router.post("/render")
async def render_segmentation(
    file: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    """
    Визуализация сегментации SegFormer — возвращает PNG-изображение
    с наложенной маской и контуром полигона.

    Откройте результат в браузере, чтобы наглядно увидеть
    сегментированную область.

    Параметры:
    - **file**: RGB-изображение (JPEG, PNG, TIFF)
    - **threshold**: Порог уверенности (0.0–1.0)
    """
    if not runtime.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SegFormer model is not loaded.",
        )

    try:
        file_bytes = await file.read()
        img_bgr = read_image(BytesIO(file_bytes))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    request = SegmentRequest(
        threshold=threshold,
        include_overlay=False,
    )

    try:
        result = runtime.run_segmentation(rgb, request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}",
        )

    # Рендерим overlay — рисуем полигон на оригинале
    overlay_bgr = draw_polygons(
        img_bgr,
        [result.navigable.polygon_px] if result.navigable.valid else [],
        color=(0, 255, 0),
        thickness=4,
    )
    # Добавляем полупрозрачную заливку
    if result.navigable.valid and len(result.navigable.polygon_px) >= 3:
        pts = np.array(result.navigable.polygon_px, dtype=np.int32).reshape((-1, 1, 2))
        overlay_copy = overlay_bgr.copy()
        cv2.fillPoly(overlay_copy, [pts], color=(0, 255, 0))
        overlay_bgr = cv2.addWeighted(overlay_bgr, 0.7, overlay_copy, 0.3, 0)

    png_bytes = encode_png(overlay_bgr)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="segformer_segmentation.png"',
            "X-Area-Px": f"{result.navigable.area_px:.0f}",
            "X-Inference-Ms": f"{result.metrics.inference_ms:.1f}",
        },
    )


@router.post("/reload")
async def reload_model():
    """Перезагрузка SegFormer-модели (например, после обновления весов)."""
    try:
        runtime.unload()
        runtime.load()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "checkpoint": runtime.meta.get("checkpoint_path")},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}",
        )


@router.get("/classes")
async def get_classes():
    """Возвращает список классов сегментации SegFormer."""
    return {
        "classes": runtime.engine.labels,
        "checkpoint": runtime.meta.get("checkpoint_path"),
    }
