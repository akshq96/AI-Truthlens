#!/usr/bin/env bash
# Start the DeepScan stack on macOS/Linux:
#   ML inference (FastAPI)  http://127.0.0.1:7070
#   API (Node/Express)      http://localhost:5050
#   Web app (React)         http://localhost:3000
# MongoDB must already be running on localhost:27017.
# Logs are written to ./logs. Stop everything with scripts/stop-all.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ML="$ROOT/deepscan-backend/ml_server"
BE="$ROOT/deepscan-backend"
FE="$ROOT/deepscan-frontend"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

port_in_use() { lsof -tiTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

if ! port_in_use 27017; then
  echo "WARNING: nothing is listening on 27017 (MongoDB). Analysis works, but history will not be saved."
fi

if port_in_use 7070; then
  echo "ML server already running on :7070"
else
  echo "Starting ML server (models take about a minute to load) ..."
  (cd "$ML" && nohup .venv/bin/python -m uvicorn image_server:app --host 127.0.0.1 --port 7070 > "$LOGS/ml_server.log" 2>&1 &)
fi

if port_in_use 5050; then
  echo "API already running on :5050"
else
  echo "Starting Node API ..."
  (cd "$BE" && nohup node server.js > "$LOGS/api.log" 2>&1 &)
fi

if port_in_use 3000; then
  echo "Web app already running on :3000"
else
  echo "Starting React web app ..."
  (cd "$FE" && BROWSER=none PORT=3000 nohup npm start > "$LOGS/frontend.log" 2>&1 &)
fi

echo "Waiting for the ML server to report healthy ..."
for _ in $(seq 1 90); do
  if curl -s --max-time 2 http://127.0.0.1:7070/health >/dev/null 2>&1; then
    echo "ML server ready."
    break
  fi
  sleep 2
done

echo ""
echo "DeepScan: A Synthetic Data-Augmented Deepfake Detection"
echo "  Web app  http://localhost:3000"
echo "  API      http://localhost:5050"
echo "  ML       http://127.0.0.1:7070/health"
echo "  Logs     $LOGS"
