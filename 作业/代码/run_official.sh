#!/usr/bin/env bash
# run_official.sh —— 2026-11-28 测评口径自检
#
# 老师公告：统一使用最朴素的 O2 编译、统一编译运行指令、统一输入输出文件、
#           程序首尾计时、只记录第一次测评时间。
#
# 本脚本：
#   1) 用系统默认 gcc + 纯 -O2 构建（Makefile 的 official 目标）
#   2) 固定输入 timg.bmp、固定输出 ../结果/测评口径/official_output.bmp
#   3) 4 线程（Pthread/OpenMP 均为默认 4），每个配置只跑第一次并记录 Total time
#   4) 另跑一组 taskset 0,2,4,6（4 个物理 P 核）作为可复现参照
#   5) 每次跑完立刻过教师验卷器（若 ../../环境验证/test 存在）
#
# 用法: bash run_official.sh
set -euo pipefail

PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
RESULT="$PROG_DIR/../结果/测评口径"
INPUT="$TESTDIR/timg.bmp"
OUT="$RESULT/official_output.bmp"
CHECKER="$PROG_DIR/../../环境验证/test"

mkdir -p "$RESULT"
make -s -C "$PROG_DIR" official

{
    echo "# 2026-11-28 测评口径自检环境"
    echo "date     : $(date -Is)"
    echo "gcc      : $(gcc --version | head -1)"
    echo "cpu      : $(lscpu | awk -F: '/Model name/{sub(/^ +/, "", $2); print $2; exit}')"
    echo "input    : $INPUT"
    echo "output   : $OUT"
    echo "loadavg  : $(cat /proc/loadavg)"
    if command -v fincore >/dev/null 2>&1; then
        echo "input cache:"; fincore "$INPUT" 2>/dev/null || true
    fi
} > "$RESULT/environment.txt"

: > "$RESULT/summary.txt"
: > "$RESULT/checker.txt"

run_one() {                       # $1=标签 $2=二进制 $3=CPU 列表（空=不绑核）
    local name="$1" bin="$2" cpus="$3"
    local log="$RESULT/${name}_first.txt"
    rm -f "$OUT" "$OUT.kernel.txt" "$OUT.matrix.txt"
    local cmd=("$bin" "$INPUT" "$OUT")
    if [[ -n "$cpus" ]]; then
        cmd=(taskset -c "$cpus" "${cmd[@]}")
    fi
    {
        echo "===== $name ====="
        echo "command: ${cmd[*]}"
        echo "threads: 4 (默认)"
        "${cmd[@]}"
    } 2>&1 | tee "$log"

    local t
    t=$(awk '/^Total time/{print $4}' "$log")
    printf '%-16s Total time = %s ms\n' "$name" "$t" | tee -a "$RESULT/summary.txt"

    if [[ -x "$CHECKER" ]]; then
        "$CHECKER" "$INPUT" "$OUT" > "$log.checker" 2>&1 || true
        grep -E "PASS|MSE|Similarity" "$log.checker" | tee -a "$RESULT/checker.txt" >/dev/null
    fi
}

run_one pthread_nopin "$PROG_DIR/conv_pthread_o2" ""
run_one omp_nopin     "$PROG_DIR/conv_omp_o2"     ""
run_one pthread_ppin  "$PROG_DIR/conv_pthread_o2" "0,2,4,6"
run_one omp_ppin      "$PROG_DIR/conv_omp_o2"     "0,2,4,6"

echo ""
echo "===== summary ====="
cat "$RESULT/summary.txt"
if [[ -s "$RESULT/checker.txt" ]]; then
    echo "===== teacher checker ====="
    cat "$RESULT/checker.txt"
fi
