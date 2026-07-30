#!/usr/bin/env bash
# Quick health check for figtree-news server
# Usage: bash monitor.sh

SERVER_PID=$(pgrep -f "figtree-news serve" | head -1)

echo "═══════════════════════════════════════════"
echo "  figtree-news — $(date)"
echo "═══════════════════════════════════════════"

if [ -z "$SERVER_PID" ]; then
  echo "  ❌ Server: DEAD"
  exit 1
fi

ELAPSED=$(ps -p $SERVER_PID -o etime --no-headers 2>/dev/null | xargs)
MEM=$(ps -p $SERVER_PID -o rss --no-headers 2>/dev/null | xargs)
echo "  ✅ Server PID $SERVER_PID (up $ELAPSED, ${MEM}KB RSS)"

if ss -tlnp | grep -q 8000; then
  echo "  ✅ Port 8000: listening"
else
  echo "  ❌ Port 8000: NOT listening"
fi

echo "─── GPU ───"
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null

echo "─── API ───"
STATS=$(curl -s --max-time 5 http://localhost:8000/api/stats 2>/dev/null)
if [ -n "$STATS" ]; then
  echo "$STATS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Articles: {d.get(\"articles\", \"?\")}')
print(f'  Narratives: {d.get(\"narratives\", \"?\")}')
print(f'  Sources: {d.get(\"sources\", \"?\")}')
print(f'  Brief: {\"yes\" if d.get(\"has_brief\") else \"no\"}')
  " 2>/dev/null
else
  echo "  ❌ API unreachable (timeout)"
fi

echo "─── Crawl Status ───"
CRAWL=$(curl -s --max-time 5 http://localhost:8000/api/crawl/status 2>/dev/null)
if [ -n "$CRAWL" ]; then
  echo "$CRAWL" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Running: {d.get(\"running\", \"?\")}')
print(f'  Step: {d.get(\"current_step\", \"?\")}')
print(f'  Message: {d.get(\"message\", \"?\")[:80]}')
  " 2>/dev/null
else
  echo "  ❌ Crawl status unreachable"
fi

echo "─── Recent Log ───"
tail -5 /home/otherdrums/figtree-news/nohup_server.log 2>/dev/null | sed 's/^/  /'
echo ""
