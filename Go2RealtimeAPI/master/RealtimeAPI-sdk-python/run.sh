#!/bin/bash
# 语音通话客户端启动脚本
# 使用 uv 管理 Python 环境和依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 检查 uv 是否安装
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

# 同步依赖
sync_deps() {
    info "同步项目依赖..."
    uv sync
}

# 主流程
check_uv
sync_deps
info "启动语音通话客户端..."
echo ""
uv run python example_sync.py
