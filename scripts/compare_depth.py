#!/usr/bin/env python3
"""Compare two float32 .npy depth maps for ORT-vs-TRT numeric parity.

Loads both arrays, asserts their shapes are equal, and prints the max-abs and
mean-abs element-wise difference. Exits 0 iff the shapes match and the mean-abs
difference is within --tol (default 1e-3).
"""

import argparse
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two .npy depth maps for numeric parity.")
    parser.add_argument("a", help="first .npy file (e.g. ORT depth dump)")
    parser.add_argument("b", help="second .npy file (e.g. TRT depth dump)")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="maximum allowed mean-abs difference for PASS (default: 1e-3)",
    )
    args = parser.parse_args()

    arr_a = np.load(args.a)
    arr_b = np.load(args.b)

    if arr_a.shape != arr_b.shape:
        print(f"FAIL shape mismatch: {arr_a.shape} vs {arr_b.shape}")
        return 1

    diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0

    print(f"shapes: {arr_a.shape} == {arr_b.shape}")
    print(f"max-abs:  {max_abs:.6g}")
    print(f"mean-abs: {mean_abs:.6g} (tol={args.tol:g})")

    if mean_abs <= args.tol:
        print("PASS parity")
        return 0

    print("FAIL parity (mean-abs exceeds tol)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
