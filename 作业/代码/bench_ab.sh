#!/usr/bin/env bash
# bench_ab.sh —— 基线/候选交替对照基准（共享机器上抗负载/频率漂移）
#
# 用法: bash bench_ab.sh <round标签> <基线二进制目录> [pthread|omp|both] [RUNS]
#   基线二进制目录: 存有 conv_pthread / conv_omp 的目录（当前最优代码的编译副本，
#                   由  cp conv_pthread conv_omp ../优化日志/_ab_base/  维护）
#   例:  bash bench_ab.sh R17 ../优化日志/_ab_base both
#
# 协议与 bench_round.sh 完全一致（timg.bmp；taskset 0 / 0,2 / 0,2,4,6 物理P核；
# 指标 = 程序自打印 Total time），差别仅在执行方式：
#   每配置内基线与候选交替执行 base,cand,base,cand,…（各 RUNS 次），
#   两者承受相同的机器状态序列，消除"先跑完基线再跑候选"的单侧漂移偏差；
#   同时记录最优（min，看上限）与中位数（看典型）。
#
# 输出: ../优化日志/<round>/ab_<版本>.txt   汇总表（终端同步打印）
#       ../优化日志/<round>/ab_<版本>_raw.txt  每次运行完整输出
set -u
PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
LOGROOT="$PROG_DIR/../优化日志"
INPUT="$TESTDIR/timg.bmp"
ROUND="${1:?用法: bench_ab.sh <round> <base_bin_dir> [pthread|omp|both] [runs]}"
BASEDIR="${2:?缺基线二进制目录（存 conv_pthread/conv_omp 的目录）}"
VERSIONS="${3:-both}"
RUNS="${4:-5}"

case "$VERSIONS" in pthread|omp|both) ;; *) echo "版本参数须为 pthread|omp|both" >&2; exit 1;; esac
for V in pthread omp; do
    [ "$VERSIONS" != "$V" ] && [ "$VERSIONS" != both ] && continue
    [ -x "$BASEDIR/conv_$V" ] || { echo "ERROR: 基线缺失 $BASEDIR/conv_$V" >&2; exit 1; }
done

OUTDIR="$LOGROOT/$ROUND"
mkdir -p "$OUTDIR"

# 构建门禁：候选 = 当前源码构建产物；失败即中止
make -s -C "$PROG_DIR" all || { echo "ERROR: 构建失败，中止基准" >&2; exit 1; }

run_once() {  # $1=bin $2=threads $3=cpus $4=rawlog $5=tag → stdout: Total time
    local out t
    if [[ "$1" == *omp ]]; then
        out=$(OMP_NUM_THREADS="$2" taskset -c "$3" "$1" "$INPUT" "$OUTDIR/_ab_out.bmp" 2>&1)
    else
        out=$(taskset -c "$3" "$1" "$INPUT" "$OUTDIR/_ab_out.bmp" "$2" 2>&1)
    fi
    printf '===== %s =====\n%s\n' "$5" "$out" >> "$4"
    t=$(printf '%s\n' "$out" | awk '/^Total time/ {print $(NF-1)}')
    [ -n "$t" ] || { echo "ERROR: 本次运行未输出 Total time（$5）" >&2; exit 1; }
    echo "$t"
}

ab_version() {  # $1=pthread|omp
    local V="$1"
    local BASE="$BASEDIR/conv_$V" CAND="$PROG_DIR/conv_$V"
    local LOG="$OUTDIR/ab_$V.txt" RAW="$OUTDIR/ab_${V}_raw.txt"
    {
        echo "# A/B 交替对照  round=$ROUND  version=$V  runs/config=$RUNS"
        echo "# date $(date '+%F %T')   loadavg $(cat /proc/loadavg)"
        echo "# base=$BASE"
        echo "# cand=$CAND"
        echo "# analysis only; current candidate build commands:"
        make -Bn -C "$PROG_DIR" all
    } > "$LOG"
    : > "$RAW"
    local cfg th cpus t
    for cfg in "1:0" "2:0,2" "4:0,2,4,6"; do
        th="${cfg%%:*}"; cpus="${cfg##*:}"
        local -a tb=() tc=()
        for _ in $(seq "$RUNS"); do
            tb+=("$(run_once "$BASE" "$th" "$cpus" "$RAW" "base T$th run${_}")")
            tc+=("$(run_once "$CAND" "$th" "$cpus" "$RAW" "cand T$th run${_}")")
        done
        local bb mb bc mc db dc
        bb=$(printf '%s\n' "${tb[@]}" | sort -g | head -1)
        bc=$(printf '%s\n' "${tc[@]}" | sort -g | head -1)
        mb=$(printf '%s\n' "${tb[@]}" | sort -g | awk -v n="$RUNS" 'NR==int((n+1)/2)')
        mc=$(printf '%s\n' "${tc[@]}" | sort -g | awk -v n="$RUNS" 'NR==int((n+1)/2)')
        db=$(awk "BEGIN{printf \"%+.2f\", ($bc-$bb)/$bb*100}")
        dc=$(awk "BEGIN{printf \"%+.2f\", ($mc-$mb)/$mb*100}")
        echo "T$th  base: best=$bb med=$mb | cand: best=$bc med=$mc | Δbest=$db%  Δmed=$dc%" | tee -a "$LOG"
    done
}

echo "==== A/B round $ROUND ($VERSIONS), runs/config=$RUNS ===="
echo "base=$BASEDIR   cand=$PROG_DIR (当前构建)"
[ "$VERSIONS" = pthread ] && ab_version pthread
[ "$VERSIONS" = omp ]     && ab_version omp
[ "$VERSIONS" = both ]    && { ab_version pthread; ab_version omp; }
