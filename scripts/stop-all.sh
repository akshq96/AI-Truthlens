#!/usr/bin/env bash
# Stop the DeepScan web app (3000), API (5050) and ML server (7070). MongoDB is left running.
for port in 3000 5050 7070; do
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill $pids && echo "Stopped process on port $port"
  else
    echo "Nothing running on port $port"
  fi
done
