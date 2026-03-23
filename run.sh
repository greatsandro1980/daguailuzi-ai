#!/bin/bash

echo "=== Starting V13 PPO AI service ==="
cd train

# 优先使用V13 PPO模型
if [ -f "../rl_v13_ppo_best.pt" ]; then
    MODEL="../rl_v13_ppo_best.pt"
    echo "Using V13 PPO model (vs规则90.3%, vs随机79.6%)"
elif [ -f "../public/self_play_model.pt" ]; then
    MODEL="../public/self_play_model.pt"
    echo "Using public model"
elif [ -f "../rl_v9_best_rule.pt" ]; then
    MODEL="../rl_v9_best_rule.pt"
    echo "Using V9 model (fallback)"
else
    echo "No model found, AI will use rule engine"
    MODEL=""
fi

if [ -n "$MODEL" ]; then
    python3 serve_v13.py --model "$MODEL" --port 5001 &
    sleep 2
    echo "AI service started on port 5001"
fi

echo "=== Starting game server ==="
cd ..
node server/index.js
