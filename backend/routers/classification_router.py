"""
Роутер для модуля классификации культур.
Определяет тип сельскохозяйственной культуры внутри полигона.
Подключается к основному приложению FastAPI.
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, Optional, List 
from models.settings import Settings

router = APIRouter(prefix="/api/v1/classification", tags=["Crop Classification"])

settings = Settings()

# Заглушка классификатора — в реальной системе здесь будет
# загрузка дообученной модели (например, EfficientNet / ResNet)
CLASSIFIER_READY = False


class CropClassificationRequest(BaseModel):
    """Запрос на классификацию культуры внутри полигона."""
    image_base64: str = Field(
        ...,
        description="Изображение полигона (область crop) в base64 (JPEG/PNG)",
    )
    threshold: Optional[float] = Field(
        None,
        description="Порог уверенности для пометки 'требует ручной проверки'",
        ge=0.0,
        le=1.0,
    )


class CropProbability(BaseModel):
    """Вероятность класса культуры."""
    crop_class: str = Field(..., description="Название культуры")
    probability: float = Field(..., description="Уверенность модели (0.0–1.0)")


class CropClassificationResponse(BaseModel):
    """Результат классификации культуры."""
    ok: bool = True
    predicted_class: Optional[str] = Field(
        None,
        description="Класс с максимальной вероятностью",
    )
    confidence: Optional[float] = Field(
        None,
        description="Уверенность в предсказанном классе",
    )
    probabilities: List[CropProbability] = Field(
        default_factory=list,
        description="Распределение вероятностей по всем классам",
    )
    requires_review: bool = Field(
        False,
        description="True, если confidence ниже порога — требуется ручная проверка",
    )
    threshold_used: float = Field(
        ...,
        description="Порог уверенности, использованный в этом запросе",
    )


def _dummy_classifier(image_base64: str, threshold: float) -> CropClassificationResponse:
    """
    Заглушка классификатора культур.
    В реальной системе здесь будет вызов дообученной модели.
    """
    import random
    crop_classes = settings.crop_classes
    probs = [random.random() for _ in crop_classes]
    total = sum(probs)
    probs = [p / total for p in probs]

    max_idx = max(range(len(probs)), key=lambda i: probs[i])
    predicted = crop_classes[max_idx]
    confidence = probs[max_idx]

    return CropClassificationResponse(
        ok=True,
        predicted_class=predicted,
        confidence=confidence,
        probabilities=[
            CropProbability(crop_class=cls, probability=round(p, 4))
            for cls, p in zip(crop_classes, probs)
        ],
        requires_review=confidence < threshold,
        threshold_used=threshold,
    )


@router.get("/health")
async def health_check():
    """Проверка состояния модуля классификации культур."""
    return {
        "status": "healthy" if CLASSIFIER_READY else "degraded",
        "classifier_loaded": CLASSIFIER_READY,
        "num_crop_classes": len(settings.crop_classes),
        "crop_classes": settings.crop_classes,
        "confidence_threshold": settings.classification_confidence_threshold,
    }


@router.post("/classify", response_model=CropClassificationResponse)
async def classify_crop(request: CropClassificationRequest):
    """
    Классификация культуры внутри переданного изображения полигона.

    Возвращает распределение вероятностей по всем классам культур.
    Если уверенность модели ниже порога — полигон помечается как
    'требует ручной проверки'.
    """
    threshold = request.threshold or settings.classification_confidence_threshold

    if not CLASSIFIER_READY:
        # Заглушка
        return _dummy_classifier(request.image_base64, threshold)

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Crop classifier model is not yet deployed.",
    )


@router.get("/classes")
async def get_crop_classes():
    """Возвращает список поддерживаемых классов культур и текущий порог."""
    return {
        "crop_classes": settings.crop_classes,
        "confidence_threshold": settings.classification_confidence_threshold,
        "classifier_model": settings.crop_classification_model,
    }