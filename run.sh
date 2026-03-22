#!/bin/bash

echo "=== Starting V9 AI service ==="
cd train
python3 serve_v9.py --model ../rl_v9_best_rule.pt --port 5001 &

echo "=== Starting game server ==="
cd ..
node server/index.js
