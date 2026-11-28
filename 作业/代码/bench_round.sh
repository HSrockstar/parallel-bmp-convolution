#!/usr/bin/env bash
# bench_round.sh —— 优化迭代用统一基准脚本（每轮记录原始输出，保证轮次间可比）
#
# 用法: bash bench_round.sh <round标签> [pthread|omp|both] [RUNS]
#   例:  bash bench_round.sh R0 both
#   RUNS 缺省 5（协议值）；共享机负载突发期可加大（如 9），在 bench_env 留档
#
# 统一协议（所有轮次严格一致）:
#   输入   timg.bmp (4096x2304x3, 附件原图)
#   核数   1/2/4 物理P核: taskset -c 0 / 0,2 / 0,2,4,6
#          (i9-13900K: CPU0-15 = 8个P核的两两HT, CPU16-31 = E核, 按物理核计数)
#   次数   每组 RUNS 次取 "Total time" 最优（共享机器有背景负载，取 min 抗噪声）
#   指标   程序自打印 Total time（读图+卷积+写图，不含文本导出的分析口径；多次最优不能作为首次测评成绩）
#
# 输出:
#   ../优化日志/<round>/bench_<版本>.txt   每次运行完整输出（含分段计时与逐线程时间）
#   ../优化日志/<round>/bench_env.txt      测量时机器状态（负载/频率）
#   ../优化日志/summary.tsv                追加: round version T1 T2 T4 S2 S4 E2 E4
set -u
PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
LOGROOT="$PROG_DIR/../优化日志"
INPUT="$TESTDIR/timg.bmp"
ROUND="${1:?用法: bench_round.sh <round标签> [pthread|omp|both] [RUNS]}"
VERSIONS="${2:-both}"
RUNS="${3:-5}"

OUTDIR="$LOGROOT/$ROUND"
mkdir -p "$OUTDIR"
SUMMARY="$LOGROOT/summary.tsv"
[ -f "$SUMMARY" ] || printf 'round\tversion\tT1_ms\tT2_ms\tT4_ms\tS2\tS4\tE2\tE4\n' > "$SUMMARY"
: > "$OUTDIR/bench_stats.txt"   # 基准整备：每配置 best+median（summary.tsv 口径不变，仍记 best）

# 基准整备（R16 教训）：先构建，失败即中止，杜绝拿旧二进制跑出无效数据
make -s -C "$PROG_DIR" all || { echo "ERROR: 构建失败，中止基准" >&2; exit 1; }

{
    echo "build commands (current default):"
    make -Bn -C "$PROG_DIR" all
    echo "date        : $(date '+%F %T')"
    echo "input       : $INPUT"
    echo "runs/config : $RUNS (best kept)"
    echo "loadavg     : $(cat /proc/loadavg)"
    echo "governor    : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
} > "$OUTDIR/bench_env.txt"

best_of() {   # $1=bin $2=threads $3=cpus $4=logfile → stdout: best Total time
    local bin="$1" th="$2" cpus="$3" log="$4" best="" out t
    local -a times=()
    for _ in $(seq "$RUNS"); do
        if [[ "$bin" == *omp ]]; then
            out=$(OMP_NUM_THREADS="$th" taskset -c "$cpus" "$bin" "$INPUT" "$OUTDIR/_bench_out.bmp" 2>&1)
        else
            out=$(taskset -c "$cpus" "$bin" "$INPUT" "$OUTDIR/_bench_out.bmp" "$th" 2>&1)
        fi
        printf '%s\n' "$out" >> "$log"
        t=$(printf '%s\n' "$out" | awk '/^Total time/ {print $(NF-1)}')
        [ -n "$t" ] || { echo "ERROR: 本次运行未输出 Total time（见 $log）" >&2; exit 1; }
        times+=("$t")
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    # 基准整备：best+median 留档（抗共享机抖动：最优看上限，中位看典型）
    printf '%s\tthreads=%s\tcpus=%s\tbest=%s\tmedian=%s\tall=%s\n' \
        "$(basename "$bin")" "$th" "$cpus" "$best" \
        "$(printf '%s\n' "${times[@]}" | sort -g | awk -v n="$RUNS" 'NR==int((n+1)/2)')" \
        "$(printf '%s,' "${times[@]}")" >> "$OUTDIR/bench_stats.txt"
    echo "$best"
}

bench_version() {   # $1=pthread|omp
    local V="$1" bin="$PROG_DIR/conv_$1" LOG="$OUTDIR/bench_$1.txt"
    : > "$LOG"
    local T1 T2 T4 s2 s4 e2 e4
    T1=$(best_of "$bin" 1 "0"       "$LOG")
    T2=$(best_of "$bin" 2 "0,2"     "$LOG")
    T4=$(best_of "$bin" 4 "0,2,4,6" "$LOG")
    s2=$(awk "BEGIN{printf \"%.4f\", $T1/$T2}")
    s4=$(awk "BEGIN{printf \"%.4f\", $T1/$T4}")
    e2=$(awk "BEGIN{printf \"%.4f\", $T1/($T2*2)}")
    e4=$(awk "BEGIN{printf \"%.4f\", $T1/($T4*4)}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ROUND" "$V" "$T1" "$T2" "$T4" "$s2" "$s4" "$e2" "$e4" >> "$SUMMARY"
    echo "$V: T1=$T1 ms  T2=$T2 ms  T4=$T4 ms   S(2)=$s2 S(4)=$s4  E(2)=$e2 E(4)=$e4"
}

case "$VERSIONS" in
    pthread|omp|both) ;;
    *) echo "版本参数须为 pthread|omp|both" >&2; exit 1 ;;
esac

echo "==== round $ROUND ($VERSIONS) ===="
[ "$VERSIONS" = pthread ] && bench_version pthread
[ "$VERSIONS" = omp ] && bench_version omp
[ "$VERSIONS" = both ] && { bench_version pthread; bench_version omp; }
