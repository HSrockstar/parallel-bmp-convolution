#!/usr/bin/env bash
# 默认基准统一到 O2 首测；历史多次最优数据保留在优化日志中。
# 用法：[pthread|omp|both] [input.bmp] [new-result-directory]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_official.sh" "$@"
