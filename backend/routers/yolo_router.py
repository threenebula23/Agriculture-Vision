"""
Роутер для модели YOLO (Ultralytics).
Предоставляет endpoint для сегментации объектов на изображении.
Подключается к основному приложению FastAPI.
"""

from __future__ import annotations
from io import BytesIO
import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException, status, BackgroundTasks
from starlette.responses import JSONResponse, Response

from models.settings import Settings
from models.runtime import SegmentationRuntime
from models.schemas import (
    SegmentRequest,
    YoloSegmentResponse,
    HealthResponse,
)
from models.utils.read_file import read_image
from models.utils.visualize import draw_detections, encode_png

settings = Settings(model_architecture="yolo")
runtime = SegmentationRuntime(settings)
try:
    runtime.load()
except Exception as exc:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.warning("YOLO model failed to load at startup: %s", exc)

router = APIRouter(prefix="/api/v1/yolo", tags=["YOLO Segmentation"])


@router.on_event("shutdown")
async def shutdown_yolo():
    """Выгружает модель при остановке сервера."""
    runtime.unload()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния YOLO-модели."""
    return HealthResponse(
        status="healthy" if runtime.is_loaded else "unhealthy",
        model_loaded=runtime.is_loaded,
        device=runtime.device,
        fp16=runtime.meta.get("fp16_active"),
        checkpoint=runtime.meta.get("checkpoint_path"),
    )


@router.post("/segment", response_model=YoloSegmentResponse)
async def segment_image(
    file: UploadFile = File(...),
    threshold: float | None = None,
    tta: bool | None = None,
    use_sliding: bool = False,
    headland_margin_px: int | None = None,
    include_mask_png: bool = False,
    include_geojson: bool | None = None,
    background_tasks: BackgroundTasks | None = None,
):
    """
    Сегментация объектов на изображении с помощью YOLO.

    Параметры:
    - **file**: RGB-изображение (JPEG, PNG, TIFF)
    - **threshold**: Порог уверенности детекции (0.0–1.0)
    - **tta**: Использовать Test-Time Augmentation для повышения качества
    - **use_sliding**: Использовать скользящее окно для больших изображений
    - **include_mask_png**: Включить PNG-маску в ответ (base64)
    - **include_geojson**: Включить GeoJSON-представление результатов

    Возвращает:
    - Список детекций с полигонами, bounding box и метриками
    """
    if not runtime.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model is not loaded. Check server logs.",
        )

    try:
        file_bytes = await file.read()
        img_bgr = read_image(BytesIO(file_bytes))
        # Конвертируем BGR → RGB для движка
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    request = SegmentRequest(
        threshold=threshold,
        tta=tta,
        use_sliding=use_sliding,
        headland_margin_px=headland_margin_px,
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


@router.post("/reload")
async def reload_model():
    """Перезагрузка YOLO-модели (например, после обновления весов)."""
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
    """Возвращает список классов, которые умеет распознавать YOLO."""
    return {
        "classes": runtime.engine.labels,
        "checkpoint": runtime.meta.get("checkpoint_path"),
    }


@router.post("/render")
async def render_segmentation(
    file: UploadFile = File(...),
    threshold: float | None = None,
    tta: bool | None = None,
):
    """
    Визуализация сегментации YOLO — возвращает PNG-изображение
    с наложенными масками, bounding box и подписями.

    Откройте результат в браузере или сохраните как файл,
    чтобы наглядно увидеть, что распознала модель.

    Параметры:
    - **file**: RGB-изображение (JPEG, PNG, TIFF)
    - **threshold**: Порог уверенности (0.0–1.0)
    - **tta**: Test-Time Augmentation
    """
    if not runtime.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO model is not loaded. Check server logs.",
        )

    try:
        file_bytes = await file.read()
        img = read_image(BytesIO(file_bytes))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    request = SegmentRequest(
        threshold=threshold,
        tta=tta,
        include_overlay=False,  # не нужно base64 — вернём сырой PNG
    )

    try:
        result = runtime.run_segmentation(rgb, request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}",
        )

    # Рендерим overlay
    detections_dicts = [
        {
            "polygon_px": d.polygon_px,
            "bbox_xyxy": d.bbox_xyxy,
            "label": d.label,
            "confidence": d.confidence,
        }
        for d in result.detections
    ]
    overlay_bgr = draw_detections(img, detections_dicts)
    png_bytes = encode_png(overlay_bgr)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="yolo_segmentation.png"',
            "X-Detections-Count": str(len(result.detections)),
            "X-Inference-Ms": f"{result.metrics.inference_ms:.1f}",
        },
    )
