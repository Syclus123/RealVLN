#!/bin/bash
# 语音通话客户端Go2启动脚本
# 使用 uv 管理 Python 环境和依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

check_uv() {
    if ! command -v uv &> /dev/null; then
        warn "uv 未安装，正在安装..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        if ! command -v uv &> /dev/null; then
            error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"
        fi
        info "uv 安装成功"
    fi
}

cleanup() {
    info "正在停止语音服务..."
    kill "$SERVER_PID" 2>/dev/null
    kill "$CLIENT_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    wait "$CLIENT_PID" 2>/dev/null
    info "语音服务已停止"
}
trap cleanup EXIT

# 清理旧日志和导航记录
info "清理旧日志文件..."
> nav_logs/navigation.jsonl
rm -f audio_client.log
rm -f audio_server.log
rm -f nohup.out.fast_client
rm -f main_client.log
rm -f nohup.out.local_server
info "旧日志清理完成"

check_uv
info "同步项目依赖..."
uv sync

# export DASHSCOPE_API_KEY="sk-5a03c62b9a1549f8bec8fba654f3fd52"
export DASHSCOPE_API_KEY="sk-6e58b18ffbbf491c8d960a9d74dee1d6"
export TENCENT_TTS_SECRET_ID="AKIDCsKYYIn46tRlriKhpibGgrmTzHK8h16M"
export TENCENT_TTS_SECRET_KEY="4sSowpTt77tSq2zjMmzzmMjsnbCYfYDg"

info "启动语音服务..."

uv run python -m src.audio_server.server --port 18790 --log-level DEBUG \
    > nohup.out.local_server 2>&1 &
SERVER_PID=$!
info "audio_server PID=$SERVER_PID"

sleep 3

uv run python -m src.main_client_fast --ws-url ws://localhost:18790/ws \
    > nohup.out.fast_client 2>&1 &
CLIENT_PID=$!
info "main_client_fast PID=$CLIENT_PID"
info "语音服务运行中 — Ctrl+C 停止"

wait
