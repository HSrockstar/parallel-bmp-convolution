#!/usr/bin/env bash
# run_bench.sh —— 1/2/4 核 benchmark，输出加速比 S(p)=T(1)/T(p) 与效率 E(p)=S(p)/p
#
# 用法: bash run_bench.sh [pthread|omp|both] [input.bmp]
# 默认 both + timg.bmp；每组跑 RUNS 次取最优（多核机器上有抖动，取 min 更稳）
#
# 注意: 本机 i9-13900K 每个物理核有 2 个超线程 CPU 编号（0,1=核0；2,3=核1…），
#       "2 核/4 核"按物理核计，故 taskset 用 0,2 / 0,2,4,6 而不是 0,1 / 0-3。
#
# 结果文件: ../结果/bench_<版本>.txt 及汇总表 bench_summary.md
set -u

PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
RESULT="$PROG_DIR/../结果"
INPUT="${2:-$TESTDIR/timg.bmp}"
VERSIONS="${1:-both}"
RUNS=3

mkdir -p "$RESULT"

# 基准整备：先构建，失败即中止；输出图与 bench_round 统一放仓库文件系统（本机与 /tmp 同盘，纯口径统一）
make -s -C "$PROG_DIR" all || { echo "ERROR: 构建失败，中止基准" >&2; exit 1; }

total_ms_of() {            # 从程序输出里抓 Total time 数值
    awk '/^Total time/ {print $(NF-1)}' "$1"
}

bench_one() {              # $1=程序 $2=线程数 $3=cpu列表 $4=日志前缀
    local bin="$1" threads="$2" cpus="$3" log="$4"
    local best=""
    for _ in $(seq 1 $RUNS); do
        local out
        if [[ "$bin" == *omp ]]; then
            out=$(OMP_NUM_THREADS="$threads" taskset -c "$cpus" "$bin" "$INPUT" "$RESULT/bench_blur.bmp")
        else
            out=$(taskset -c "$cpus" "$bin" "$INPUT" "$RESULT/bench_blur.bmp" "$threads")
        fi
        [[ -n "$out" ]] && printf '%s\n' "$out" >> "$log"
        local t
        t=$(printf '%s\n' "$out" | awk '/^Total time/ {print $(NF-1)}')
        if [[ -z "$best" ]] || (( $(awk "BEGIN{print ($t < $best) ? 1 : 0}") )); then best="$t"; fi
    done
    echo "$best"
}

report() {                 # $1=版本名 $2=T1 $3=T2 $4=T4
    local name="$1" t1="$2" t2="$3" t4="$4"
    local s2 s4 e2 e4
    s2=$(awk "BEGIN{printf \"%.4f\", $t1/$t2}")
    s4=$(awk "BEGIN{printf \"%.4f\", $t1/$t4}")
    e2=$(awk "BEGIN{printf \"%.4f\", $t1/($t2*2)}")
    e4=$(awk "BEGIN{printf \"%.4f\", $t1/($t4*4)}")
    {
        echo "### $name（输入: $(basename "$INPUT")，取 $RUNS 次最优 Total time）"
        echo ""
        echo "| 指标 | 1 核 | 2 核 | 4 核 |"
        echo "|------|------|------|------|"
        echo "| 运行时间 T(p) ms | $t1 | $t2 | $t4 |"
        echo "| 加速比 S(p)      | 1.0000 | $s2 | $s4 |"
        echo "| 效率 E(p)        | 1.0000 | $e2 | $e4 |"
        echo ""
    } >> "$RESULT/bench_summary.md"
}

: > "$RESULT/bench_summary.md"
echo "input = $INPUT"
echo "runs per config = $RUNS (best kept)"
echo ""

if [[ "$VERSIONS" == "pthread" || "$VERSIONS" == "both" ]]; then
    LOG="$RESULT/bench_pthread.txt"
    : > "$LOG"
    echo "==================== Pthread ===================="
    T1=$(bench_one "$PROG_DIR/conv_pthread" 1 "0"       "$LOG")
    T2=$(bench_one "$PROG_DIR/conv_pthread" 2 "0,2"     "$LOG")
    T4=$(bench_one "$PROG_DIR/conv_pthread" 4 "0,2,4,6" "$LOG")
    echo "Pthread: T1=$T1 ms  T2=$T2 ms  T4=$T4 ms"
    report "Pthread 版" "$T1" "$T2" "$T4"
fi

if [[ "$VERSIONS" == "omp" || "$VERSIONS" == "both" ]]; then
    LOG="$RESULT/bench_omp.txt"
    : > "$LOG"
    echo "==================== OpenMP ====================="
    T1=$(bench_one "$PROG_DIR/conv_omp" 1 "0"       "$LOG")
    T2=$(bench_one "$PROG_DIR/conv_omp" 2 "0,2"     "$LOG")
    T4=$(bench_one "$PROG_DIR/conv_omp" 4 "0,2,4,6" "$LOG")
    echo "OpenMP : T1=$T1 ms  T2=$T2 ms  T4=$T4 ms"
    report "OpenMP 版" "$T1" "$T2" "$T4"
fi

echo ""
echo "汇总已写入 $RESULT/bench_summary.md"
cat "$RESULT/bench_summary.md"
