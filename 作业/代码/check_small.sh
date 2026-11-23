#!/usr/bin/env bash
# 两张 512×512 官方小图 × Pthread/OpenMP 的教师验卷与逐轮原始日志。
# 用法: bash check_small.sh <round>（round 必须是尚不存在的新目录）
set -u
PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDIR="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
CHECKER="$PROG_DIR/../../环境验证/test"
ROUND="${1:?用法: check_small.sh <round>}"
OUTDIR="$PROG_DIR/../优化日志/$ROUND"
if [ -e "$OUTDIR" ]; then
    printf 'ERROR: 留档目录已存在，换用新标签以保留旧证据: %s\n' "$OUTDIR" >&2
    exit 1
fi
make -s -C "$PROG_DIR" all || { printf 'ERROR: 构建失败\n' >&2; exit 1; }
mkdir -p "$OUTDIR"
FAIL=0
for IMG in Pure_white Pure_black; do
    for V in pthread omp; do
        IN="$TESTDIR/$IMG.bmp"
        OUT="$OUTDIR/${IMG}_blur_$V.bmp"
        RUN="$OUTDIR/run_${IMG}_$V.txt"
        VERIFY="$OUTDIR/verify_${IMG}_$V.txt"
        if [ "$V" = omp ]; then
            OMP_NUM_THREADS=4 taskset -c 0,2,4,6 "$PROG_DIR/conv_omp" "$IN" "$OUT" > "$RUN" 2>&1
        else
            taskset -c 0,2,4,6 "$PROG_DIR/conv_pthread" "$IN" "$OUT" 4 > "$RUN" 2>&1
        fi
        if [ "$?" -ne 0 ]; then
            printf '[FAIL] %s/%s 运行失败：%s\n' "$IMG" "$V" "$RUN"
            FAIL=1
            continue
        fi
        "$CHECKER" "$IN" "$OUT" > "$VERIFY" 2>&1
        if grep -q '\[PASS\] Similarity test passed' "$VERIFY" && grep -q '\[PASS\] Shape test passed' "$VERIFY"; then
            awk '/MSE result|Similarity rate/ {print}' "$VERIFY"
            printf '[PASS] %s/%s，日志=%s\n' "$IMG" "$V" "$VERIFY"
        else
            printf '[FAIL] %s/%s：%s\n' "$IMG" "$V" "$VERIFY"
            FAIL=1
        fi
    done
done
[ "$FAIL" -eq 0 ] || { printf 'check_small: FAILED\n'; exit 1; }
printf 'check_small: OK (4/4 teacher checks)\n'
