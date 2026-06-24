from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime, date

# --- Карты и Объекты ---
class ProcessingConfig(BaseModel):
    filename: str = Field(..., description="Имя файла для обработки")
    detect_boundaries: bool = Field(..., alias="field_boundaries")
    detect_points: bool = Field(..., alias="point_objects")
    classify_crops: bool = Field(..., alias="crop_classification")
    confidence_threshold: float = Field(0.65, ge=0.0, le=1.0)

class GeoMetadata(BaseModel):
    crs: str = "EPSG:32636"
    area_km2: float = 12.4
    georeference_ok: bool = True

class ObjectAttributes(BaseModel):
    id: str
    object_class: str
    crop: Optional[str] = None
    area_ha: Optional[float] = None
    perimeter_m: Optional[int] = None
    confidence: float
    status: str
    crs: str = "EPSG:32636"
    processed_date: date

class ProbabilityDistribution(BaseModel):
    object_id: str
    probabilities: Dict[str, float]

# --- Очередь Заданий ---
class JobItem(BaseModel):
    id: str
    filename: str
    task_type: str
    status: str  # "Ожидание", "В процессе", "Готово", "Ошибка"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

# --- Модели ---
class ModelMetrics(BaseModel):
    accuracy: float
    f1_score: float
    miou: Optional[float] = None

class ModelVersion(BaseModel):
    version: str
    is_active: bool
    updated_at: date
    metrics: ModelMetrics