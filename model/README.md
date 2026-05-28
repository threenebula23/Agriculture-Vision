# Model serving — SegFormer FP16 + HTTP API

Веса лежат в **`model/weights/`** (`best_iou.pth`, `last_epoch.pth`). Обучение пишет сюда же (`config/agvision.yaml` → `output_dir: model/weights`).

## Структура

```
model/
  weights/          # чекпоинты (.pth в .gitignore)
  config.yaml
  runtime.py        # загрузка модели
  inference.py      # FP16 forward
  pipeline.py         # RGB+NIR → полигон
  app.py            # FastAPI
  cli.py
```

## Запуск

```bash
pip install -r requirements.txt -r model/requirements.txt

python -m model --rgb photo.jpg --out result.json
uvicorn model.app:app --host 0.0.0.0 --port 8080
```
