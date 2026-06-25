# Agriculture Vision API — Документация

## Обзор

Сервис искусственного интеллекта для автоматической сегментации границ полей, детекции точечных объектов и классификации сельскохозяйственных культур. Поддерживаются две архитектуры нейросетей: **YOLO** (Ultralytics) и **SegFormer** (HuggingFace Transformers).

**Базовый URL:** `http://localhost:8000`  
**Интерактивная документация:** `/docs` (Swagger UI)  
**Версия API:** 1.1.0

---

## 1. Главная страница

### `GET /`

Возвращает приветственное сообщение и ссылки на доступные endpoint'ы.

**Пример ответа:**
```json
{
  "message": "Agriculture Vision API — сервис сегментации аграрных полей",
  "docs": "/docs",
  "version": "1.1.0",
  "available_endpoints": {
    "segmentation": "/api/v1/segmentation/segment",
    "yolo": "/api/v1/yolo/segment",
    "segformer": "/api/v1/segformer/segment",
    "classification": "/api/v1/classification/classify"
  }
}
```

---

## 2. Общий роутер сегментации (`/api/v1/segmentation`)

Объединяет обе архитектуры через параметр `architecture`.

### `GET /api/v1/segmentation/health`

Проверка состояния моделей.

**Параметры запроса:**
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `architecture` | `"yolo" \| "segformer" \| "all"` | `"all"` | Какая архитектура проверяется |

**Пример ответа (architecture=all):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "available_models": ["yolo", "segformer"],
  "models": {
    "yolo": { "loaded": true, "checkpoint": "config/yolo_best.pt" },
    "segformer": { "loaded": true, "checkpoint": "config/segformer_best.pt" }
  }
}
```

### `POST /api/v1/segmentation/segment`

Единый endpoint сегментации с выбором архитектуры через параметр.

**Параметры запроса (multipart/form-data):**
| Параметр | Тип | Обязательный | По умолчанию | Описание |
|---|---|---|---|---|
| `file` | `file` | Да | — | RGB-изображение (JPEG, PNG, TIFF) |
| `architecture` | `"yolo" \| "segformer"` | Нет | `"yolo"` | Архитектура сегментации |
| `threshold` | `float` | Нет | `0.4` | Порог уверенности (0.0–1.0) |
| `tta` | `bool` | Нет | `null` | Test-Time Augmentation (только YOLO) |
| `include_mask_png` | `bool` | Нет | `false` | Включить PNG-маску (base64) |
| `include_geojson` | `bool` | Нет | `null` | Включить GeoJSON |

**Формат ответа (YOLO):**
```json
{
  "ok": true,
  "detections": [
    {
      "label": "field",
      "confidence": 0.95,
      "polygon_px": [[100, 200], [150, 250], ...],
      "area_px": 15000.5,
      "bbox_xyxy": [100, 200, 400, 500],
      "valid": true
    }
  ],
  "geojson": null,
  "mask_png_base64": "iVBORw0KGgo...",
  "image_hw": [1024, 768],
  "checkpoint": "config/yolo_best.pt",
  "val_metrics": { "mAP_0.5": 0.732 },
  "metrics": {
    "threshold_used": 0.4,
    "area_frac": 0.45,
    "prob_mean": 0.87,
    "prob_std": 0.12,
    "mode": "standard",
    "inference_ms": 234.5,
    "fp16": true,
    "device": "cuda:0"
  }
}
```

**Формат ответа (SegFormer):**
```json
{
  "ok": true,
  "navigable": {
    "polygon_px": [[100, 200], ...],
    "area_px": 500000.0,
    "valid": true
  },
  "geojson": null,
  "mask_png_base64": "...",
  "image_hw": [1024, 768],
  "checkpoint": "config/segformer_best.pt",
  "val_metrics": null,
  "metrics": { ... }
}
```

### `GET /api/v1/segmentation/models`

Список загруженных моделей с метаданными.

### `POST /api/v1/segmentation/reload`

Перезагрузка модели(ей). Параметр `architecture`: `"yolo"`, `"segformer"` или `"all"`.

---

## 3. YOLO Router (`/api/v1/yolo`)

Специализированный роутер для модели YOLO.

| Метод | Path | Описание |
|---|---|---|
| `GET` | `/api/v1/yolo/health` | Проверка состояния YOLO-модели |
| `POST` | `/api/v1/yolo/segment` | Сегментация изображения YOLO |
| `POST` | `/api/v1/yolo/reload` | Перезагрузка YOLO-модели |
| `GET` | `/api/v1/yolo/classes` | Список классов YOLO |

### `POST /api/v1/yolo/segment`

Параметры: `file`, `threshold`, `tta`, `use_sliding`, `include_mask_png`, `include_geojson`.  
Формат ответа: `YoloSegmentResponse`.

### `GET /api/v1/yolo/classes`

```json
{
  "classes": {
    "0": "field",
    "1": "double_plant",
    "2": "drydown",
    "3": "endrow",
    "4": "nutrient_deficiency",
    "5": "planter_skip",
    "6": "storm_damage",
    "7": "water",
    "8": "waterway",
    "9": "weed_cluster"
  },
  "checkpoint": "config/yolo_best.pt"
}
```

---

## 4. SegFormer Router (`/api/v1/segformer`)

Специализированный роутер для модели SegFormer.

| Метод | Path | Описание |
|---|---|---|
| `GET` | `/api/v1/segformer/health` | Проверка состояния SegFormer-модели |
| `POST` | `/api/v1/segformer/segment` | Семантическая сегментация SegFormer |
| `POST` | `/api/v1/segformer/reload` | Перезагрузка SegFormer-модели |
| `GET` | `/api/v1/segformer/classes` | Список классов сегментации |

### `POST /api/v1/segformer/segment`

Параметры: `file`, `threshold`, `include_mask_png`, `include_geojson`.  
Формат ответа: `SegmentResponse`.

### `GET /api/v1/segformer/classes`

```json
{
  "classes": {
    "0": "background",
    "1": "field",
    "2": "double_plant",
    "3": "drydown",
    "4": "endrow",
    "5": "nutrient_deficiency",
    "6": "planter_skip",
    "7": "storm_damage",
    "8": "water",
    "9": "waterway",
    "10": "weed_cluster"
  },
  "checkpoint": "config/segformer_best.pt"
}
```

---

## 5. Классификация культур (`/api/v1/classification`)

Модуль определения типа сельскохозяйственной культуры внутри полигона.

| Метод | Path | Описание |
|---|---|---|
| `GET` | `/api/v1/classification/health` | Состояние модуля |
| `POST` | `/api/v1/classification/classify` | Классификация культуры |
| `GET` | `/api/v1/classification/classes` | Список классов культур |

### `POST /api/v1/classification/classify`

**Body (JSON):**
```json
{
  "image_base64": "iVBORw0KGgo...",
  "threshold": 0.6
}
```

**Ответ:**
```json
{
  "ok": true,
  "predicted_class": "wheat",
  "confidence": 0.85,
  "probabilities": [
    { "crop_class": "wheat", "probability": 0.85 },
    { "crop_class": "corn", "probability": 0.10 },
    { "crop_class": "soybean", "probability": 0.03 },
    ...
  ],
  "requires_review": false,
  "threshold_used": 0.6
}
```

> **Важно:** Модель классификации культур пока находится в режиме заглушки (`CLASSIFIER_READY = False`). Для продакшн-использования необходимо подключить дообученный классификатор.

---

## 6. Описания моделей данных (Pydantic schemas)

Все схемы находятся в `models/schemas.py`.

### SegmentRequest
| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `threshold` | `float \| None` | `null` | Порог уверенности (0.0–1.0) |
| `tta` | `bool \| None` | `null` | Использовать TTA (Test-Time Augmentation) |
| `use_sliding` | `bool` | `false` | Использовать скользящее окно |
| `headland_margin_px` | `int \| None` | `null` | Отступ от краёв в пикселях |
| `include_mask_png` | `bool` | `false` | Включить PNG-маску (base64) |
| `include_geojson` | `bool \| None` | `null` | Включить GeoJSON |

### SegmentMetrics
| Поле | Тип | Описание |
|---|---|---|
| `threshold_used` | `float` | Фактический порог уверенности |
| `area_frac` | `float` | Доля площади изображения, покрытая маской |
| `prob_mean` | `float` | Средняя уверенность детекций |
| `prob_std` | `float` | Стандартное отклонение уверенности |
| `mode` | `str` | Режим инференса ("standard", "tta", "sliding") |
| `inference_ms` | `float` | Время инференса в миллисекундах |
| `fp16` | `bool` | Использовался ли FP16 |
| `device` | `str` | Устройство (cpu/cuda) |

### HealthResponse
| Поле | Тип | Описание |
|---|---|---|
| `status` | `str` | "healthy" или "unhealthy" |
| `model_loaded` | `bool` | Загружена ли модель |
| `device` | `str \| None` | Устройство |
| `fp16` | `bool \| None` | FP16 активен |
| `checkpoint` | `str \| None` | Путь к чекпоинту |

### PolygonPayload
| Поле | Тип | Описание |
|---|---|---|
| `polygon_px` | `list[tuple[int,int]]` | Список точек полигона (пиксели) |
| `area_px` | `float` | Площадь полигона в пикселях |
| `valid` | `bool` | Корректен ли полигон |

### YoloDetectionPayload
| Поле | Тип | Описание |
|---|---|---|
| `label` | `str` | Имя класса |
| `confidence` | `float` | Уверенность детекции |
| `polygon_px` | `list[tuple[int,int]]` | Полигон маски |
| `area_px` | `float` | Площадь маски |
| `bbox_xyxy` | `list[int]` | Bounding box [x1,y1,x2,y2] |
| `valid` | `bool` | Валидность детекции |

---

## 7. Запуск сервера

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Или через python
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 8. Примеры использования (curl)

### YOLO сегментация
```bash
curl -X POST "http://localhost:8000/api/v1/yolo/segment" \
  -F "file=@field_image.jpg" \
  -F "threshold=0.5" \
  -F "include_mask_png=true"
```

### SegFormer сегментация
```bash
curl -X POST "http://localhost:8000/api/v1/segformer/segment" \
  -F "file=@field_image.jpg" \
  -F "threshold=0.5"
```

### Через общий роутер
```bash
curl -X POST "http://localhost:8000/api/v1/segmentation/segment?architecture=yolo" \
  -F "file=@field_image.jpg"
```

### Проверка здоровья
```bash
curl "http://localhost:8000/api/v1/segmentation/health?architecture=all"
```

### Классификация культур
```bash
curl -X POST "http://localhost:8000/api/v1/classification/classify" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "iVBORw0KGgo...", "threshold": 0.6}'
```

---

## 9. Конфигурация (models/settings.py)

Файл `models/settings.py` содержит все настраиваемые параметры:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `model_architecture` | `"yolo"` | Активная архитектура |
| `max_concurrent_inferences` | `1` | Максимум параллельных инференсов |
| `default_confidence_threshold` | `0.4` | Порог уверенности по умолчанию |
| `default_iou_threshold` | `0.5` | NMS IoU порог |
| `inference_image_size` | `640` | Размер изображения для инференса |
| `yolo_weights_path` | `"config/yolo_best.pt"` | Путь к весам YOLO |
| `yolo_fallback_weights` | `"yolo11m-seg.pt"` | Запасная модель YOLO |
| `segformer_weights_path` | `"config/segformer_best.pt"` | Путь к весам SegFormer |
| `segformer_pretrained_model` | `"nvidia/segformer-b5-finetuned-ade-640-640"` | Pretrained модель SegFormer |
| `segformer_num_labels` | `10` | Количество классов SegFormer |
| `classification_confidence_threshold` | `0.6` | Порог для пометки "требует проверки" |

---

## 10. Архитектура проекта

```
Agriculture-Vision/
├── app.py                          # Главное приложение FastAPI
├── backend/
│   └── routers/
│       ├── segmentation.py         # Общий роутер (YOLO + SegFormer)
│       ├── yolo_router.py          # Специализированный YOLO роутер
│       ├── segformer_router.py     # Специализированный SegFormer роутер
│       └── classification_router.py # Роутер классификации культур
├── models/
│   ├── schemas.py                  # Pydantic схемы данных
│   ├── settings.py                 # Конфигурация параметров
│   ├── runtime.py                  # Оркестратор выполнения моделей
│   ├── engines/
│   │   ├── base.py                 # Базовый класс движка сегментации
│   │   ├── yolo.py                 # YOLO движок
│   │   └── segformer.py            # SegFormer движок
│   └── utils/
│       └── read_file.py            # Утилита чтения изображений
├── config/
│   ├── yolo_best.pt                # Веса дообученной YOLO
│   ├── segformer_best.pt           # Веса дообученного SegFormer
│   └── obj.yaml                    # Список классов объектов
├── logger/
│   ├── __init__.py
│   └── logger.py                   # Настройка логирования
├── docker/
│   ├── Dockerfile                  # Dockerfile для сборки
│   └── requirements.txt            # Зависимости Python
├── notebooks/                      # Jupyter ноутбуки для обучения
└── docs/
    ├── API_DOCUMENTATION.md        # Данный файл
    └── BUILD_DOCKER.md             # Инструкция по сборке Docker
```

---

## 11. Обработка ошибок

API возвращает стандартные HTTP-статусы:

| Статус | Описание |
|---|---|
| `200 OK` | Успешный запрос |
| `400 Bad Request` | Некорректный файл или параметры |
| `500 Internal Server Error` | Ошибка инференса модели |
| `503 Service Unavailable` | Модель не загружена |

Формат ошибки:
```json
{
  "detail": "Описание ошибки"
}
