#!/bin/bash
# V12训练启动脚本 - 自动安装依赖并启动训练

echo "=== V12训练启动脚本 ==="

# 安装依赖
echo "[1/2] 安装PyTorch..."
pip3 install torch numpy --quiet 2>/dev/null

# 启动训练
echo "[2/2] 启动V12训练..."
cd /workspace/projects
nohup python train/rl_train_v12.py > /app/work/logs/bypass/v12_train.log 2>&1 &

echo "训练已启动，PID: $!"
echo "查看日志: tail -f /app/work/logs/bypass/v12_train.log"
