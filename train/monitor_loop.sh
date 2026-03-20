#!/bin/bash
# 训练监控脚本 - 每10分钟检查一次

while true; do
    echo ""
    echo "=== [$(date '+%H:%M:%S')] 监控检查 ==="
    
    # 检查训练进程
    TRAIN_PID=$(ps aux | grep "train_v3.py" | grep -v grep | awk '{print $2}' | head -1)
    
    if [ -z "$TRAIN_PID" ]; then
        echo "❌ 训练进程已停止，重启中..."
        
        # 检查torch
        python3 -c "import torch" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "安装torch..."
            pip install torch --index-url https://download.pytorch.org/whl/cpu -q
        fi
        
        # 重启训练
        cd /workspace/projects/train
        nohup python3 train_v3.py > /app/work/logs/bypass/training_v3.log 2>&1 &
        sleep 5
        echo "✅ 训练已重启"
    else
        echo "✅ 训练运行中 (PID: $TRAIN_PID)"
    fi
    
    # 显示状态
    if [ -f /workspace/projects/training_status.json ]; then
        EP=$(cat /workspace/projects/training_status.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('episode', 0))")
        WR=$(cat /workspace/projects/training_status.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('win_rate', 0))")
        SR=$(cat /workspace/projects/training_status.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('score_rate', 0))")
        PROGRESS=$(python3 -c "print(round($EP / 200000 * 100, 1))")
        echo "进度: $EP / 200,000 局 ($PROGRESS%) | 胜率: ${WR}% | 得分率: ${SR}%"
        
        # 检查是否完成
        if [ "$EP" -ge 200000 ]; then
            echo ""
            echo "🎉🎉🎉 训练完成！共 $EP 局 🎉🎉🎉"
            break
        fi
    fi
    
    # 等待10分钟
    echo "等待10分钟后再次检查..."
    sleep 600
done
