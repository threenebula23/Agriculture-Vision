from fastapi import APIRouter, HTTPException, status
from agro_gis.schemas import ModelVersion, ModelMetrics
from datetime import date
from typing import List

router = APIRouter(prefix="/api/v1/models", tags=["Model Management"])

MOCK_MODELS = [
    ModelVersion(
        version="v2.1.0",
        is_active=True,
        updated_at=date(2026, 6, 15),
        metrics=ModelMetrics(accuracy=0.91, f1_score=0.89, miou=0.84)
    ),
    ModelVersion(
        version="v2.0.1",
        is_active=False,
        updated_at=date(2026, 5, 10),
        metrics=ModelMetrics(accuracy=0.88, f1_score=0.86, miou=0.81)
    )
]


@router.get("/", response_model=List[ModelVersion])
async def get_model_versions():
    return MOCK_MODELS


@router.post("/rollback/{version}", status_code=status.HTTP_200_OK)
async def rollback_model(version: str):
    target_found = False
    for model in MOCK_MODELS:
        if model.version == version:
            model.is_active = True
            target_found = True
        else:
            model.is_active = False

    if not target_found:
        raise HTTPException(status_code=404, detail=f"Версия модели {version} не найдена в системе")

    return {"message": f"Успешный откат системы на версию модели {version}"}