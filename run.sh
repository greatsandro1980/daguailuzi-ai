#!/bin/bash

echo "=== Starting AI service ==="
cd train
python3 serve.py --model ../stage2_final.pt --port 5001 &

echo "=== Starting game server ==="
cd ..
node server/index.js
