#!/bin/bash

echo "=== Starting V18 Evolution Strategy AI service ==="
cd train

# 优先使用V18进化策略优化模型
if [ -f "../rl_v18_best.pt" ]; then
    MODEL="../rl_v18_best.pt"
    echo "Using V18 Evolution Strategy model (vs规则95.0%, vs随机86.3%)"
elif [ -f "../rl_v14b_best.pt" ]; then
    MODEL="../rl_v14b_best.pt"
    echo "Using V14b Strategy Optimized model (vs规则94.2%, vs随机82.7%)"
elif [ -f "../public/self_play_model.pt" ]; then
    MODEL="../public/self_play_model.pt"
    echo "Using public model"
else
    echo "No model found, AI will use rule engine"
    MODEL=""
fi

if [ -n "$MODEL" ]; then
    python3 serve_v14b.py --model "$MODEL" --port 5001 &
    sleep 2
    echo "AI service started on port 5001"
fi

echo "=== Starting game server ==="
cd ..
node server/index.js
