# Field Detecter — ViT границы полей + детекция точек

Сегментация границ сельхозучастков (**SegFormer / ViT**, 4 канала RGB+NIR) и детекция деревьев/столбов (**YOLO26**) на датасете [Agriculture-Vision](https://github.com/SHI-Labs/Agriculture-Vision).

Метрики по ТЗ §5.1: IoU ≥ 0.90, Recall ≥ 0.95, Precision ≥ 0.90 (поля); F1 ≥ 0.80 (точки, на pseudo-GT).

## Результаты

- Первый прогон
```
Epoch: 2 val: {'iou_mean': 0.9398277102289697, 'precision_mean': 0.9424682089160502, 'recall_mean': 0.9955257925503495, 'n_samples': 1000}
```

- Второй прогон
```
Epoch: 12 val: {'iou_mean': 0.9461152559973611, 'precision_mean': 0.9513153409510501, 'recall_mean': 0.9928616511294276, 'n_samples': 1000}
```

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Скачать Agriculture-Vision (~21 GB)
python scripts/download_agvision.py --extract

# 2. Обучение SegFormer (NIR+RGB, аугментации, устойчивость к шуму)
python -m field_detecter.train_seg --config config/agvision.yaml
# или на ночь:
bash scripts/train_seg_overnight.sh

# 3. Псевдоразметка + YOLO26
python -m field_detecter.pseudo_points --config config/agvision.yaml
python -m field_detecter.train_det --config config/agvision.yaml
```

Блокнот end-to-end: [`notebooks/train_field_vit.ipynb`](notebooks/train_field_vit.ipynb)

Тест инференса и полигоны: [`notebooks/test_segmentation_polygons.ipynb`](notebooks/test_segmentation_polygons.ipynb)

Конфиг: [`config/agvision.yaml`](config/agvision.yaml)

## Структура

```
field_detecter/
  agvision_dataset.py   # 4ch + boundary + valid mask
  train_seg.py          # SegFormer-B4
  pseudo_points.py      # pseudo tree/pole
  train_det.py          # YOLO26
  metrics.py            # IoU, P, R, F1 + отчёт ТЗ
  polygon.py            # маска → GeoJSON
scripts/download_agvision.py
config/agvision.yaml
```


