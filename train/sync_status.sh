#!/bin/bash
# 状态同步脚本 - 每分钟同步一次状态文件

while true; do
    # 复制状态文件
    cp /workspace/projects/train/training_status.json /workspace/projects/training_status.json 2>/dev/null
    
    # 检查训练进程是否存活
    if ! pgrep -f "train_background.py" > /dev/null; then
        echo "[$(date)] 训练进程已停止，尝试重启..."
        cd /workspace/projects
        nohup python3 train/train_background.py --episodes 200000 --save_dir . >> /app/work/logs/bypass/training_bg.log 2>&1 &
    fi
    
    sleep 60
done
