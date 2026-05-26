#!/usr/bin/env bash
# Ночное обучение SegFormer (4ch NIR+RGB, аугментации, шум)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

echo "=== SegFormer overnight train ==="
echo "Config: config/agvision.yaml"
echo "Logs: outputs/segformer/train.log"
mkdir -p outputs/segformer

python -m field_detecter.train_seg --config config/agvision.yaml 2>&1 | tee outputs/segformer/train.log

echo "Done. Best: outputs/segformer/best_iou.pth"
