#!/usr/bin/env bash
set -o pipefail
cd /home/otherdrums/figtree-news || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LOGFILE="/home/otherdrums/figtree-news/logs/server-$(date +%Y%m%d-%H%M).log"
echo "[$(date)] Starting server..." >> "$LOGFILE"
exec /home/otherdrums/figtree/.venv_f39/bin/figtree-news serve \
  --db demo/news.lance \
  --sources demo/sources.json \
  --host 0.0.0.0 \
  --port 8000 >> "$LOGFILE" 2>&1
