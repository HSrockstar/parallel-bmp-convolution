#!/usr/bin/env bash
# O2 首次测评基线：默认 4 线程、不额外绑核、默认完整文本输出。
# 不绑核是基线选择，老师没有禁止 taskset；可在外层 taskset 后另开结果目录。
# 用法：bash run_official.sh [pthread|omp|both] [input.bmp] [不存在的结果目录]
set -euo pipefail

PROG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

verify_checker_log() {
    local checker_log="$1"
    local pass_count
    pass_count=$(awk '/^\[PASS\]/{n++} END{print n+0}' "$checker_log")
    if grep -q '\[FAIL\]' "$checker_log" || [[ "$pass_count" -ne 2 ]]; then
        cat "$checker_log" >&2
        echo '教师验卷未通过，停止测评。' >&2
        return 1
    fi
}

main() {
    local versions=(pthread omp)
    case "${1:-}" in
        pthread|omp) versions=("$1"); shift ;;
        both) shift ;;
    esac
    if [[ "$#" -gt 2 ]]; then
        echo 'Usage: bash run_official.sh [pthread|omp|both] [input.bmp] [new-result-directory]' >&2
        return 1
    fi
    local testdir="$PROG_DIR/../../附件/采用Pthread对BMP图片进行卷积作业附件/测试程序/测试程序"
    local input="${1:-$testdir/timg.bmp}"
    local result="${2:-$PROG_DIR/../结果/O2首测_$(date +%Y%m%d_%H%M%S)_$$}"
    [[ -f "$input" ]] || { echo "找不到输入：$input" >&2; return 1; }
    [[ ! -e "$result" ]] || { echo "结果目录已存在：$result" >&2; return 1; }
    mkdir -p "$result"
    result="$(cd "$result" && pwd)"
    input="$(realpath "$input")"
    local out="$result/official_output.bmp"
    local checker="$result/teacher_checker"

    # 清除调用环境中可能改变线程数、绑核或动态调度的设置。
    unset OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY
    export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE
    {
        date -Is
        gcc --version | head -1
        uname -sr
        lscpu | awk -F: '/Model name/{print; exit}'
        echo "input=$input"
        echo "output=$out"
        echo "loadavg=$(cat /proc/loadavg)"
        echo "allowed_cpus=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
        echo "CONV_CPUS=${CONV_CPUS:-<unset>}"
        echo 'OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE; no additional affinity in this script'
        if command -v fincore >/dev/null; then fincore "$input"; fi
    } > "$result/environment.txt"

    # 强制重新编译，避免不同编译选项生成的旧二进制被复用。
    make -B -C "$PROG_DIR" CC=gcc CFLAGS=-O2 official > "$result/build.txt" 2>&1
    local opencv_flags=()
    read -r -a opencv_flags <<< "$(pkg-config --cflags --libs opencv4)"
    g++ "$testdir/test.cpp" -o "$checker" "${opencv_flags[@]}" >> "$result/build.txt" 2>&1

    run_one() {
        local version="$1" threads="$2"
        local label="${version}_${threads}"
        local log="$result/${label}_first.txt"
        local cmd=("$PROG_DIR/conv_$version" "$input" "$out")
        # 四线程组不传可选参数，验证默认运行与默认完整输出。
        if [[ "$threads" -ne 4 ]]; then cmd+=("$threads"); fi
        rm -f -- "$out" "$out.kernel.txt" "$out.matrix.txt"
        printf '\n===== %s 首次运行 =====\n' "$label"
        printf 'command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}" 2>&1 | tee "$log"
        local actual
        actual=$(awk '/^Threads/{print $3}' "$log")
        [[ "$actual" == "$threads" ]] || { echo "线程数不符：$actual / $threads" >&2; return 1; }
        "$checker" "$input" "$out" > "$result/${label}.checker.txt" 2>&1
        verify_checker_log "$result/${label}.checker.txt"
        cat "$result/${label}.checker.txt"
        if [[ "$threads" -eq 4 ]]; then
            cp --reflink=auto "$out" "$result/timg_blur_${version}.bmp"
            cp --reflink=auto "$out.kernel.txt" "$result/timg_blur_${version}.bmp.kernel.txt"
            cp --reflink=auto "$out.matrix.txt" "$result/timg_blur_${version}.bmp.matrix.txt"
        fi
    }

    # 无预跑；先记录默认四线程，再补一、二线程用于加速比。
    local version
    for version in "${versions[@]}"; do run_one "$version" 4; done
    for version in "${versions[@]}"; do run_one "$version" 1; done
    for version in "${versions[@]}"; do run_one "$version" 2; done

    python3 - "$result" "${versions[@]}" <<'PY'
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
records = []
for version in sys.argv[2:]:
    data = {}
    for threads in (1, 2, 4):
        text = (root / f'{version}_{threads}_first.txt').read_text()
        def value(pattern):
            found = re.search(pattern, text, re.M)
            if not found:
                raise SystemExit('Missing timing: ' + pattern)
            return float(found.group(1))
        data[threads] = {
            'version': version, 'threads': threads,
            'total_ms': value(r'^Total time\s*:\s*([\d.]+)'),
            'conv_ms': value(r'^Conv\s*:\s*\(parallel region[^\n]*?\)\s*([\d.]+)'),
            'full_ms': value(r'^Wall time \(full\)\s*:\s*([\d.]+)'),
        }
    for threads, row in data.items():
        row['speedup_total'] = data[1]['total_ms'] / row['total_ms']
        row['efficiency_total'] = row['speedup_total'] / threads
        row['speedup_conv'] = data[1]['conv_ms'] / row['conv_ms']
        row['efficiency_conv'] = row['speedup_conv'] / threads
        records.append(row)
lines = ['O2 first-run results; no best-of selection; milliseconds',
         'version threads total_ms conv_ms full_ms S_total E_total S_conv E_conv']
for row in records:
    lines.append('{version:8s} {threads:d} {total_ms:.3f} {conv_ms:.3f} {full_ms:.3f} '
                 '{speedup_total:.4f} {efficiency_total:.4f} {speedup_conv:.4f} {efficiency_conv:.4f}'.format(**row))
(root / 'summary.txt').write_text('\n'.join(lines) + '\n')
(root / 'summary.json').write_text(json.dumps(records, indent=2))
print('\n'.join(lines))
PY
    echo "RESULT_DIRECTORY=$result"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
