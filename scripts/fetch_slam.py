"""Download the SLAM dataset (minimal by default) into ./data.

    python scripts/fetch_slam.py             # 5 train + 1 test scan (with k-space)
    python scripts/fetch_slam.py --test-only # full test split w/ k-space (recommended)
    python scripts/fetch_slam.py --full      # full set (200+ train, ~2-3 days)

Our method is scan-specific (per-slice INR optimization), so it needs no
training set -- the full *test* split is the recommended acquisition for real
experiments. No credentials required. See src/presinr/data/slam.py for layout.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.data import slam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="download train + test (2-3 days)")
    ap.add_argument("--test-only", action="store_true", help="download full test split with k-space")
    args = ap.parse_args()

    if args.full:
        print("Downloading FULL SLAM dataset (this can take 2-3 days)...")
        slam.pull_metadata(minimal=False, override=True)
        slam.pull_volumes("train", load_ksp=True, override=True)
        slam.pull_volumes("test", load_ksp=True, override=True)
        slam.prepare_test()
    elif args.test_only:
        print("Downloading full SLAM TEST split (with k-space)...")
        slam.pull_metadata(minimal=False, override=True)
        slam.pull_volumes("test", load_ksp=True, override=True)
        slam.prepare_test()
    else:
        slam.download_minimal()


if __name__ == "__main__":
    main()
