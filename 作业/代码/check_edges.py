#!/usr/bin/env python3
"""非均匀窄 BMP 的零填充卷积回归；不依赖第三方库或教师素材。

用法:
  python3 -I check_edges.py --pthread ./conv_pthread --omp ./conv_omp --work-dir DIR
检查宽度 1..12、高度 1/2/3/4/5/8，输出必须位于非空新建目录。
"""

import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

KERNEL = (
    (0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881),
    (0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402),
    (0.03507270, 0.06831229, 0.08531173, 0.06831229, 0.03507270),
    (0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402),
    (0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881),
)
WIDTHS = tuple(range(1, 13))
HEIGHTS = (1, 2, 3, 4, 5, 8)


def make_bmp(width, height):
    stride = (width * 3 + 3) & ~3
    pixels = bytearray(stride * height)
    for y in range(height):
        for x in range(width):
            for c in range(3):
                # 非对称、非均匀值，覆盖零填充边界和各 BGR 通道。
                pixels[y * stride + x * 3 + c] = (37 + 53 * x + 29 * y + 71 * c + 17 * x * y) % 256
        # 非零输入行填充：验证它不会混入卷积，且输出尾填充被清零。
        for n in range(width * 3, stride):
            pixels[y * stride + n] = 145 + n - width * 3
    header = bytearray(54)
    header[:2] = b"BM"
    struct.pack_into("<I", header, 2, 54 + len(pixels))
    struct.pack_into("<I", header, 10, 54)
    struct.pack_into("<I", header, 14, 40)
    struct.pack_into("<ii", header, 18, width, height)
    struct.pack_into("<HH", header, 26, 1, 24)
    struct.pack_into("<I", header, 34, len(pixels))
    return bytes(header + pixels), stride


def direct_blur(pixels, width, height, stride):
    expected = bytearray(len(pixels))
    for y in range(height):
        for x in range(width):
            for c in range(3):
                acc = 0.0
                for ky in range(5):
                    sy = y + ky - 2
                    if sy < 0 or sy >= height:
                        continue
                    for kx in range(5):
                        sx = x + kx - 2
                        if 0 <= sx < width:
                            acc += KERNEL[ky][kx] * pixels[sy * stride + sx * 3 + c]
                expected[y * stride + x * 3 + c] = min(255, max(0, int(acc + 0.5)))
    return expected


def compare_bmp(actual, expected_pixels, width, height):
    if len(actual) != 54 + len(expected_pixels) or actual[:2] != b"BM":
        return False, 0, 999, 0.0
    if struct.unpack_from("<ii", actual, 18) != (width, height):
        return False, 0, 999, 0.0
    diffs = [abs(a - b) for a, b in zip(actual[54:], expected_pixels) if a != b]
    ndiff = len(diffs)
    mse = sum(d * d for d in diffs) / (width * height * 3)
    similarity = 100.0 - math.sqrt(mse) / 2.55
    passed = similarity > 99.0 and max(diffs, default=0) <= 1
    return passed, ndiff, max(diffs, default=0), similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pthread", required=True, type=Path)
    parser.add_argument("--omp", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.work_dir.exists():
        parser.error("--work-dir 必须不存在；请使用新标签保留旧证据")
    bins = (("pthread", args.pthread.resolve()), ("omp", args.omp.resolve()))
    for _, binary in bins:
        if not binary.is_file():
            parser.error("找不到可执行文件: %s" % binary)
    args.work_dir.mkdir(parents=True)
    lines = ["version\twidth\theight\tpass\tdiff_bytes\tmax_abs\tsimilarity_pct\tstdout"]
    failures = 0
    cases = 0
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=1")
    for version, binary in bins:
        for width in WIDTHS:
            for height in HEIGHTS:
                tag = "%s_w%d_h%d" % (version, width, height)
                bmp, stride = make_bmp(width, height)
                expected = direct_blur(bmp[54:], width, height, stride)
                source = args.work_dir / (tag + ".bmp")
                target = args.work_dir / (tag + "_blur.bmp")
                stdout = args.work_dir / (tag + ".log")
                source.write_bytes(bmp)
                cmd = [str(binary), str(source), str(target)]
                if version == "pthread":
                    cmd.append("4")
                try:
                    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            timeout=30, check=False)
                    stdout.write_text(result.stdout, encoding="utf-8")
                    if result.returncode != 0:
                        passed, ndiff, max_abs, similarity = False, -1, 999, 0.0
                        print("FAIL %s: exit=%d（日志 %s）" % (tag, result.returncode, stdout), flush=True)
                    elif not target.is_file():
                        passed, ndiff, max_abs, similarity = False, -1, 999, 0.0
                    else:
                        passed, ndiff, max_abs, similarity = compare_bmp(
                            target.read_bytes(), expected, width, height)
                        if not passed:
                            print("FAIL %s: diff_bytes=%d max_abs=%d similarity=%.5f%%" %
                                  (tag, ndiff, max_abs, similarity), flush=True)
                except subprocess.TimeoutExpired as error:
                    output = error.stdout or ""
                    if isinstance(output, bytes):
                        output = output.decode(errors="replace")
                    stdout.write_text(output + "\nTIMEOUT\n", encoding="utf-8")
                    passed, ndiff, max_abs, similarity = False, -1, 999, 0.0
                    print("FAIL %s: timeout" % tag, flush=True)
                cases += 1
                failures += not passed
                lines.append("%s\t%d\t%d\t%s\t%d\t%d\t%.6f\t%s" %
                             (version, width, height, "PASS" if passed else "FAIL",
                              ndiff, max_abs, similarity, stdout.name))
    lines.append("TOTAL\tcases=%d\tfailures=%d" % (cases, failures))
    (args.work_dir / "results.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("%d cases, %d failures；results=%s" % (cases, failures, args.work_dir / "results.tsv"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
