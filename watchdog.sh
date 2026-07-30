#!/usr/bin/env bash
# Watchdog: restarts the figtree-news server if it crashes.
# Logs restarts to logs/watchdog.log
cd /home/otherdrums/figtree-news
LOG="logs/watchdog.log"
echo "[$(date)] Watchdog started" >> "$LOG"

while true; do
  nohup /home/otherdrums/figtree/.venv_f39/bin/figtree-news serve \
    --db demo/news.lance \
    --sources demo/sources.json \
    --host 0.0.0.0 \
    --port 8000 >> nohup_server.log 2>&1 &
  PID=$!
  echo "[$(date)] Server started (PID $PID)" >> "$LOG"
  wait $PID
  EXIT_CODE=$?
  echo "[$(date)] Server exited (PID $PID, code $EXIT_CODE) — restarting in 10s" >> "$LOG"
  sleep 10
done
