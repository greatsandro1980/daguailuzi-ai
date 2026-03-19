#!/bin/bash
# 大怪路子训练启动脚本

cd /workspace/projects

echo "=========================================="
echo "  大怪路子 AI 训练系统"
echo "=========================================="
echo ""
echo "目标: 20万局训练，胜率>60%"
echo ""

# 启动后台训练
echo "🚀 启动后台训练..."
nohup python3 train/train_background.py --episodes 200000 --save_dir . > /app/work/logs/bypass/training_bg.log 2>&1 &
TRAIN_PID=$!
echo "   训练进程 PID: $TRAIN_PID"
echo ""

# 启动监控服务器
echo "📊 启动监控服务..."
cd train
nohup python3 -m http.server 8888 > /app/work/logs/bypass/monitor.log 2>&1 &
MONITOR_PID=$!
echo "   监控服务 PID: $MONITOR_PID}"
echo ""

echo "=========================================="
echo "  访问监控页面:"
echo "  http://localhost:8888/monitor.html"
echo "=========================================="
echo ""
echo "训练日志: /app/work/logs/bypass/training_bg.log"
echo "状态文件: /workspace/projects/training_status.json"
echo ""
echo "使用以下命令查看实时日志:"
echo "  tail -f /app/work/logs/bypass/training_bg.log"
echo ""
