# Отчёт по метрикам модели YOLO-сегментации

Документ описывает результаты обучения модели **YOLO26m-seg** на датасете **Agriculture-Vision 2021** для детекции и сегментации аномалий на аэрофотоснимках полей.

## Задача и классы

Модель решает задачу instance segmentation по 9 классам:

| ID | Класс |
|----|-------|
| 0 | double_plant |
| 1 | drydown |
| 2 | endrow |
| 3 | nutrient_deficiency |
| 4 | planter_skip |
| 5 | storm_damage |
| 6 | water |
| 7 | waterway |
| 8 | weed_cluster |

Обучение выполнялось на RGB-тайлах с размером входа **512×512**, batch size **64**, **~55 эпох**.

---

## Сводные метрики (валидация)

Итоговые значения по графику обучения на последних эпохах:

### Bounding Box (B)

| Метрика | Значение |
|---------|----------|
| Precision | ~0.55 |
| Recall | ~0.40 |
| mAP@0.5 | ~0.40 |
| mAP@0.5:0.95 | ~0.22 |

### Mask (M)

| Метрика | Значение |
|---------|----------|
| Precision | ~0.48 |
| Recall | ~0.35 |
| mAP@0.5 | ~0.33 |
| mAP@0.5:0.95 | ~0.15 |

### Оптимальные пороги уверенности

| Кривая | Лучший F1 | Порог confidence |
|--------|-----------|------------------|
| Box F1-Confidence | **0.44** | 0.173 |
| Mask F1-Confidence | **0.39** | 0.188 |

При пороге confidence **0.996** достигается precision **1.00** по всем классам (маски), при confidence **0.0** — recall **0.57**.

---

## Динамика обучения

<p align="center">
  <img src="image/results.png" alt="Динамика loss и метрик за эпохи" width="900">
</p>

*Рис. 1 — Динамика loss и метрик за эпохи (`results.png`)*

### Loss-функции

**Обучение (train):** все компоненты loss стабильно снижаются на протяжении всего цикла:

- `box_loss`: 2.0 → 1.4
- `seg_loss`: 4.2 → 2.4
- `cls_loss`: 15.0 → 4.0
- `dfl_loss`: 0.04 → 0.025
- `sem_loss`: 2.3 → 0.3

**Валидация (val):** loss снижается до ~30–40-й эпохи, после чего наблюдается лёгкий рост — признак начала переобучения:

- `val/box_loss`: минимум ~1.75 около 40-й эпохи
- `val/seg_loss`: минимум ~3.5 около 40-й эпохи
- `val/cls_loss`: минимум ~8.8 около 20-й эпохи, затем рост до ~9.3

### Метрики по эпохам

- Box-метрики (precision, recall, mAP) выходят на плато после ~20-й эпохи.
- Mask-метрики растут медленнее box-метрик и также стабилизируются после ~20-й эпохи.
- Оптимальные веса модели, вероятно, находятся в диапазоне **30–40 эпох**.

---

## Метрики по классам

### Precision–Recall (маски, AP@0.5)

<p align="center">
  <img src="image/MaskPR_curve.png" alt="Mask PR Curve" width="700">
</p>

*Рис. 2 — Precision–Recall кривая для масок*

| Класс | AP@0.5 |
|-------|--------|
| water | **0.700** |
| drydown | 0.501 |
| planter_skip | 0.437 |
| waterway | 0.353 |
| nutrient_deficiency | 0.310 |
| double_plant | 0.264 |
| weed_cluster | 0.211 |
| endrow | 0.172 |
| storm_damage | **0.017** |
| **Среднее (mAP@0.5)** | **0.330** |

### F1-Confidence

<table>
  <tr>
    <td align="center" width="50%">
      <img src="image/BoxF1_curve.png" alt="Box F1 Curve" width="420"><br>
      <em>Рис. 3 — F1 vs Confidence (bounding box)</em>
    </td>
    <td align="center" width="50%">
      <img src="image/MaskF1_curve.png" alt="Mask F1 Curve" width="420"><br>
      <em>Рис. 4 — F1 vs Confidence (mask)</em>
    </td>
  </tr>
</table>

**Лучшие классы (mask F1):**

| Класс | Пиковый F1 (прибл.) |
|-------|---------------------|
| water | ~0.70 |
| drydown | ~0.60 |
| waterway | ~0.57 |
| planter_skip | ~0.48 |

**Слабые классы:**

| Класс | Пиковый F1 (прибл.) |
|-------|---------------------|
| storm_damage | ~0.03 |
| weed_cluster | ~0.35 |
| endrow | ~0.45 |

### Precision и Recall vs Confidence

<table>
  <tr>
    <td align="center" width="50%">
      <img src="image/MaskP_curve.png" alt="Mask Precision Curve" width="420"><br>
      <em>Рис. 5 — Precision vs Confidence (mask)</em>
    </td>
    <td align="center" width="50%">
      <img src="image/MaskR_curve.png" alt="Mask Recall Curve" width="420"><br>
      <em>Рис. 6 — Recall vs Confidence (mask)</em>
    </td>
  </tr>
</table>

- Классы **water** и **planter_skip** сохраняют высокую precision уже при низком пороге confidence (~0.2–0.4).
- Класс **storm_damage** остаётся нестабильным на всём диапазоне confidence.
- Recall монотонно падает с ростом порога; класс **water** демонстрирует наилучшую полноту детекции.

---

## Визуальная оценка

### Валидация: ground truth vs предсказания

#### Batch 0

<table>
  <tr>
    <th align="center">Разметка (ground truth)</th>
    <th align="center">Предсказания модели</th>
  </tr>
  <tr>
    <td align="center">
      <img src="image/val_batch0_labels.jpg" alt="Валидация — разметка, batch 0" width="420">
    </td>
    <td align="center">
      <img src="image/val_batch0_pred.jpg" alt="Валидация — предсказания, batch 0" width="420">
    </td>
  </tr>
</table>

*Рис. 7 — Сравнение разметки и предсказаний, валидационный batch 0*

На примерах видно:

- **water** — детектируется уверенно (confidence 0.8–0.9), маски соответствуют затопленным участкам.
- **waterway** — корректно выделяются линейные водотоки.
- **nutrient_deficiency** — наиболее частый класс в выборке; confidence варьируется от 0.3 до 0.8, часть слабых предсказаний (0.3–0.4) может быть ложноположительными.
- **planter_skip** и **drydown** — присутствуют в разметке, на предсказаниях встречаются реже.

#### Batch 1

<table>
  <tr>
    <th align="center">Разметка (ground truth)</th>
    <th align="center">Предсказания модели</th>
  </tr>
  <tr>
    <td align="center">
      <img src="image/val_batch1_labels.jpg" alt="Валидация — разметка, batch 1" width="420">
    </td>
    <td align="center">
      <img src="image/val_batch1_pred.jpg" alt="Валидация — предсказания, batch 1" width="420">
    </td>
  </tr>
</table>

*Рис. 8 — Сравнение разметки и предсказаний, валидационный batch 1*

#### Batch 2

<p align="center">
  <img src="image/val_batch2_labels.jpg" alt="Валидация — разметка, batch 2" width="700">
</p>

*Рис. 9 — Разметка, валидационный batch 2*

### Обучающая выборка (аугментации)

Примеры батчей с применёнными аугментациями (mosaic, flip, HSV и др.):

<table>
  <tr>
    <td align="center">
      <img src="image/train_batch47550.jpg" alt="Train batch 47550" width="280"><br>
      <em>Рис. 10</em>
    </td>
    <td align="center">
      <img src="image/train_batch47551.jpg" alt="Train batch 47551" width="280"><br>
      <em>Рис. 11</em>
    </td>
    <td align="center">
      <img src="image/train_batch47552.jpg" alt="Train batch 47552" width="280"><br>
      <em>Рис. 12</em>
    </td>
  </tr>
</table>

---

## Галерея всех иллюстраций

| № | Файл | Описание |
|---|------|----------|
| 1 | `results.png` | Динамика loss и метрик |
| 2 | `MaskPR_curve.png` | Precision–Recall (mask) |
| 3 | `BoxF1_curve.png` | F1–Confidence (box) |
| 4 | `MaskF1_curve.png` | F1–Confidence (mask) |
| 5 | `MaskP_curve.png` | Precision–Confidence (mask) |
| 6 | `MaskR_curve.png` | Recall–Confidence (mask) |
| 7 | `val_batch0_labels.jpg` | Валидация batch 0 — разметка |
| 8 | `val_batch0_pred.jpg` | Валидация batch 0 — предсказания |
| 9 | `val_batch1_labels.jpg` | Валидация batch 1 — разметка |
| 10 | `val_batch1_pred.jpg` | Валидация batch 1 — предсказания |
| 11 | `val_batch2_labels.jpg` | Валидация batch 2 — разметка |
| 12 | `train_batch47550.jpg` | Обучение — аугментации |
| 13 | `train_batch47551.jpg` | Обучение — аугментации |
| 14 | `train_batch47552.jpg` | Обучение — аугментации |

<p align="center">
  <img src="image/results.png" width="280">
  <img src="image/MaskPR_curve.png" width="280">
  <img src="image/BoxF1_curve.png" width="280">
  <img src="image/MaskF1_curve.png" width="280">
  <img src="image/MaskP_curve.png" width="280">
  <img src="image/MaskR_curve.png" width="280">
</p>

<p align="center">
  <img src="image/val_batch0_labels.jpg" width="280">
  <img src="image/val_batch0_pred.jpg" width="280">
  <img src="image/val_batch1_labels.jpg" width="280">
  <img src="image/val_batch1_pred.jpg" width="280">
  <img src="image/val_batch2_labels.jpg" width="280">
</p>

<p align="center">
  <img src="image/train_batch47550.jpg" width="280">
  <img src="image/train_batch47551.jpg" width="280">
  <img src="image/train_batch47552.jpg" width="280">
</p>