#!/bin/bash
# 自我博弈训练监控脚本
# 每10分钟检查，停止时自动从检查点继续

LOG_FILE="/app/work/logs/bypass/monitor_self_play.log"
TRAIN_LOG="/app/work/logs/bypass/self_play.log"
CHECKPOINT="/workspace/projects/self_play_checkpoint.json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "自我博弈训练监控启动"
log "检查间隔: 10分钟"
log "========================================"

while true; do
    # 检查自我博弈进程
    PID=$(ps aux | grep "self_play_v1.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PID" ]; then
        log "⚠️ 自我博弈训练未运行，检查是否完成..."
        
        # 检查是否已经完成
        if [ -f "$CHECKPOINT" ]; then
            EPISODE=$(cat "$CHECKPOINT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('episode', 0))" 2>/dev/null)
            if [ "$EPISODE" -ge 500000 ]; then
                log "✅ 训练已完成！总局数: $EPISODE"
                break
            fi
        fi
        
        log "🔄 重新启动自我博弈训练..."
        cd /workspace/projects
        nohup python3 train/self_play_v1.py >> "$TRAIN_LOG" 2>&1 &
        NEW_PID=$!
        log "训练已重启 (PID: $NEW_PID)"
    else
        # 显示当前进度
        if [ -f "/workspace/projects/training_status.json" ]; then
            EP=$(cat /workspace/projects/training_status.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('episode', 0))" 2>/dev/null)
            WR=$(cat /workspace/projects/training_status.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('red_win_rate', 0))" 2>/dev/null)
            SP=$(cat /workspace/projects/training_status.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('speed', 0))" 2>/dev/null)
            log "✅ 运行中 (PID: $PID) | 进度: $EP/500000 | 红队胜率: ${WR}% | 速度: ${SP}局/秒"
        fi
    fi
    
    sleep 600  # 10分钟
done

log "========================================"
log "监控结束"
log "========================================"
