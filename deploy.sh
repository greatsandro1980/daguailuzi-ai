#!/bin/bash
# deploy.sh - 一键部署脚本
# 用途：本地局域网或配合 ngrok 公网部署

set -e

echo "====== 大怪路子 - 构建部署脚本 ======"
echo ""

# 1. 构建前端
echo ">>> 构建前端..."
cd "$(dirname "$0")/daguailuzi"
npm install --quiet
npm run build
cd ..

# 2. 安装服务器依赖
echo ">>> 安装服务器依赖..."
cd server
npm install --quiet
cd ..

echo ""
echo "====== 构建完成！======"
echo ""
echo "启动方式："
echo "  node server/index.js"
echo ""
echo "然后访问 http://localhost:3001"
echo ""
echo "公网暴露（ngrok）："
echo "  ngrok http 3001"
echo ""
