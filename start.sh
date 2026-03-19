#!/bin/bash
# 大怪路子AI版 一键启动脚本
# 用法：
#   ./start.sh          → 只启动游戏（AI用规则引擎）
#   ./start.sh --ai     → 启动游戏 + 推理服务（需要先训练）

ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="$ROOT/train"
SERVER_DIR="$ROOT/server"
FRONT_DIR="$ROOT/daguailuzi"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 大怪路子AI版 启动 ===${NC}"

# 启动推理服务（可选）
if [[ "$1" == "--ai" ]]; then
    MODEL="$TRAIN_DIR/checkpoints/model_latest.pt"
    if [ -f "$MODEL" ]; then
        echo -e "${GREEN}[1/3] 启动神经网络推理服务...${NC}"
        cd "$TRAIN_DIR"
        python3 serve.py --model "$MODEL" --port 5001 &
        AI_PID=$!
        sleep 2
        echo "      推理服务 PID: $AI_PID"
        export AI_INFERENCE_URL="http://localhost:5001/ai_action"
    else
        echo -e "${YELLOW}[警告] 未找到训练模型 ($MODEL)，AI将使用规则引擎${NC}"
        echo -e "${YELLOW}       先运行: cd train && python3 train.py${NC}"
    fi
else
    echo -e "${YELLOW}[提示] 使用规则AI，加 --ai 参数可启用神经网络AI${NC}"
fi

# 启动游戏服务器
echo -e "${GREEN}[2/3] 启动游戏服务器 (port 3002)...${NC}"
cd "$SERVER_DIR"
PORT=3002 node index.js &
SERVER_PID=$!
sleep 1
echo "      游戏服务器 PID: $SERVER_PID"

# 启动前端开发服务器
echo -e "${GREEN}[3/3] 启动前端 (port 5174)...${NC}"
cd "$FRONT_DIR"
npx vite --port 5174 &
FRONT_PID=$!

echo ""
echo -e "${GREEN}✅ 启动完成！${NC}"
echo "   游戏地址: http://localhost:5174"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待并处理退出
trap "kill $SERVER_PID $FRONT_PID $AI_PID 2>/dev/null; echo '已停止'" INT TERM
wait
