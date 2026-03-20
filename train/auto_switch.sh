#!/bin/bash
# 自动切换训练脚本
# 监控当前训练，完成后自动启动自我博弈

LOG_FILE="/app/work/logs/bypass/auto_switch.log"
STATUS_FILE="/workspace/projects/training_status.json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "自动切换训练监控启动"
log "========================================"

# 等待当前训练完成
log "监控当前训练进度..."

while true; do
    if [ -f "$STATUS_FILE" ]; then
        EPISODE=$(cat "$STATUS_FILE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('episode', 0))" 2>/dev/null)
        TARGET=$(cat "$STATUS_FILE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('target', 200000))" 2>/dev/null)
        
        if [ -z "$EPISODE" ]; then
            EPISODE=0
        fi
        if [ -z "$TARGET" ]; then
            TARGET=200000
        fi
        
        log "当前进度: $EPISODE / $TARGET"
        
        if [ "$EPISODE" -ge "$TARGET" ]; then
            log "当前训练已完成！"
            break
        fi
    else
        log "状态文件不存在，检查训练进程..."
    fi
    
    # 检查训练进程是否存活
    PID=$(ps aux | grep "train_v3.py" | grep -v grep | awk '{print $2}')
    if [ -z "$PID" ]; then
        log "训练进程已停止"
        # 检查是否真的完成了
        if [ -f "$STATUS_FILE" ]; then
            EPISODE=$(cat "$STATUS_FILE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('episode', 0))" 2>/dev/null)
            if [ "$EPISODE" -ge 190000 ]; then
                log "训练接近完成，视为已完成"
                break
            fi
        fi
    fi
    
    sleep 30
done

log "========================================"
log "启动自我博弈训练 (50万局)"
log "========================================"

# 启动自我博弈训练
cd /workspace/projects
nohup python3 train/self_play_v1.py > /app/work/logs/bypass/self_play.log 2>&1 &
SELF_PID=$!

log "自我博弈训练已启动 (PID: $SELF_PID)"

# 等待一下确认启动成功
sleep 5
if ps -p $SELF_PID > /dev/null; then
    log "自我博弈训练运行正常"
else
    log "错误: 自我博弈训练启动失败"
    exit 1
fi

log "========================================"
log "监控完成，自我博弈训练进行中..."
log "查看进度: /workspace/projects/training_status.json"
log "========================================"
