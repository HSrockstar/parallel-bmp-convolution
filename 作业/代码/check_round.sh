#!/usr/bin/env bash
# check_round.sh —— 正确性门禁：优化后必须先过本脚本才允许进入基准测试
#
# 用法: bash check_round.sh <round标签> [--make-reference]
#
# 做三件事（timg.bmp, 4 线程, 4 物理P核）:
#   1. 跑 Pthread/OpenMP 两版程序，生成模糊图
#   2. 教师验卷器（环境验证/test, OpenCV MSE）打分 → verify_<版本>.txt
#      尺寸检查 + 相似度 > 99% 才算通过（作业验收红线）
#   3. 与参考图 reference.bmp 逐字节比对，统计差异字节数
#      （参考图由 baseline double 版生成，与验卷器 MSE=0，用来跟踪"逐像素一致"是否保持）
#
# --make-reference: 用本次 Pthread 版输出建立/覆盖参考图（仅在 R0 基线时用）
set -u
PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
LOGROOT="$PROG_DIR/../优化日志"
CHECKER="$PROG_DIR/../../环境验证/test"
ROUND="${1:?用法: check_round.sh <round标签> [--make-reference]}"
MKREF="${2:-}"
OUTDIR="$LOGROOT/$ROUND"
IN="$TESTDIR/timg.bmp"
REF="$LOGROOT/reference.bmp"

mkdir -p "$OUTDIR"
FAIL=0

taskset -c 0,2,4,6 "$PROG_DIR/conv_pthread" "$IN" "$OUTDIR/timg_blur_pthread.bmp" 4 > "$OUTDIR/run_pthread.txt" 2>&1 \
    || { echo "[FAIL] conv_pthread 运行出错（见 $OUTDIR/run_pthread.txt）"; FAIL=1; }
OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$PROG_DIR/conv_omp" "$IN" "$OUTDIR/timg_blur_omp.bmp" > "$OUTDIR/run_omp.txt" 2>&1 \
    || { echo "[FAIL] conv_omp 运行出错（见 $OUTDIR/run_omp.txt）"; FAIL=1; }
[ "$FAIL" = 0 ] || exit 1

for V in pthread omp; do
    OUT="$OUTDIR/timg_blur_$V.bmp"
    VLOG="$OUTDIR/verify_$V.txt"
    "$CHECKER" "$IN" "$OUT" > "$VLOG" 2>&1
    if grep -q '\[PASS\] Similarity test passed' "$VLOG"; then
        sim=$(awk '/Similarity rate/ {print $NF}' "$VLOG")
        mse=$(awk '/MSE result/ {print $NF}' "$VLOG")
        verdict="PASS (sim=$sim, MSE=$mse)"
    else
        verdict="FAIL (验卷器未通过, 见 $VLOG)"; FAIL=1
    fi
    if [ -f "$REF" ]; then
        if cmp -s "$REF" "$OUT"; then
            exact="与参考图逐字节一致 (0 bytes differ)"
        else
            ndiff=$(cmp -l "$REF" "$OUT" | wc -l)
            exact="与参考图存在差异: $ndiff bytes differ"
        fi
    else
        exact="(参考图尚未建立)"
    fi
    echo "$V: 验卷 $verdict ; $exact" | tee -a "$VLOG"
done

if [ "$MKREF" = "--make-reference" ]; then
    cp "$OUTDIR/timg_blur_pthread.bmp" "$REF"
    echo "参考图已建立: $REF"
fi

[ "$FAIL" = 0 ] || { echo "check_round: FAILED"; exit 1; }
echo "check_round: OK"
