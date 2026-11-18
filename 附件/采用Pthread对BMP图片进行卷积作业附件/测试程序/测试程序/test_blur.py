import sys
import cv2
import numpy as np
from math import sqrt


def compare_mse(pic1, pic2):
    diff = (pic1.astype("float") - pic2.astype("float")) ** 2
    sse = np.sum(diff)
    mse = sse / (pic1.shape[0] * pic1.shape[1] * pic1.shape[2])
    return mse


def sim_rate(mse):
    return 100 - sqrt(mse) / 2.55 


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_blur.py <Teacher Raw Image> <Your Blurred Image>")
        sys.exit(1)

    raw = cv2.imread(sys.argv[1])
    your = cv2.imread(sys.argv[2])

    if raw is None:
        print("Error: Cannot read teacher raw image.")
        sys.exit(1)

    if your is None:
        print("Error: Cannot read your blurred image.")
        sys.exit(1)

    kernel = np.array([
        [0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881],
        [0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402],
        [0.03507270, 0.06831229, 0.08531173, 0.06831229, 0.03507270],
        [0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402],
        [0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881]
    ])

    blur = cv2.filter2D(raw, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    if blur.shape != your.shape:
        print("[FAIL] Shape mismatch!")
        print(f"Standard image shape : {blur.shape}")
        print(f"Your image shape     : {your.shape}")
        sys.exit(1)
    else:
        print("[PASS] Shape test passed!")

    mse = compare_mse(blur, your)
    sim = sim_rate(mse)

    print(f"MSE result: {mse:.2f}")
    print(f"Similarity: {sim:.2f}%")

    if sim > 99:
        print("[PASS] Similarity test passed!")
    else:
        print("[FAIL] Similarity test failed (< 99%).")
