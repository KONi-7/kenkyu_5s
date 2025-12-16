"""Compare evaluation metrics with token pruning enabled vs disabled.

This script runs `test.py` twice with identical sampled batches and prints the
stdout tail for each run so you can quickly inspect metric differences.

It intentionally shells out to `test.py` because that file currently owns most
of the wiring (model construction, loaders, Deepspeed init, etc.).

Usage example:
  python tools/compare_pruning.py \
    --python /path/to/python \
    --version ./ck/SIDA-7B \
    --dataset_dir ./test \
    --vision_pretrained ./ck/sam_vit_h_4b8939.pth \
    --precision fp16 \
    --test_batch_size 1 \
    --sample_ratio 0.001

Notes:
- We force a stable sample set via --sample_seed and --sample_indices_path.
- Edit default args below if you want different pruning ratios.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = proc.stdout
    # Return entire output to allow grepping, but also print a short tail.
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compare pruning on/off on same sampled batches")
    p.add_argument("--python", default="python", help="Python executable to run test.py")

    # Common test.py args
    p.add_argument("--version", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--vision_pretrained", required=True)
    p.add_argument("--precision", default="fp16")
    p.add_argument("--test_batch_size", type=int, default=1)
    p.add_argument("--sample_ratio", type=float, default=0.0001)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument(
        "--sample_indices_path",
        default="./runs/compare/sample_indices.json",
        help="Where to save/load sampled batch indices (JSON)",
    )

    # Pruning settings (used only for pruning-enabled run)
    p.add_argument("--prune_keep_ratio", type=float, default=0.3)
    p.add_argument("--prune_observe_layer", type=int, default=-24)

    args = p.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    indices_path = str((repo_dir / args.sample_indices_path).resolve())

    base = [
        args.python,
        "test.py",
        "--test_only",
        "--version",
        args.version,
        "--dataset_dir",
        args.dataset_dir,
        "--vision_pretrained",
        args.vision_pretrained,
        "--precision",
        args.precision,
        "--test_batch_size",
        str(args.test_batch_size),
        "--sample_ratio",
        str(args.sample_ratio),
        "--sample_seed",
        str(args.sample_seed),
        "--sample_indices_path",
        indices_path,
    ]

    cmd_pruning_on = base + [
        "--prune_keep_ratio",
        str(args.prune_keep_ratio),
        "--prune_observe_layer",
        str(args.prune_observe_layer),
    ]

    cmd_pruning_off = base + ["--disable_token_pruning"]

    print("\n=== Run A: pruning ENABLED ===")
    out_on = _run(cmd_pruning_on, cwd=repo_dir)
    print("\n--- tail (enabled) ---")
    print("\n".join(out_on.splitlines()[-60:]))

    print("\n=== Run B: pruning DISABLED ===")
    out_off = _run(cmd_pruning_off, cwd=repo_dir)
    print("\n--- tail (disabled) ---")
    print("\n".join(out_off.splitlines()[-60:]))

    # Minimal heuristic summary
    def _extract(lines: list[str], key: str) -> str | None:
        for ln in reversed(lines):
            if ln.strip().startswith(key):
                return ln.strip()
        return None

    lines_on = out_on.splitlines()
    lines_off = out_off.splitlines()

    keys = [
        "giou:",
        "Classification Accuracy:",
        "Pixel Accuracy:",
        "IoU (Tampered):",
        "AUC (ROC) for localization:",
        "AUC (PR / AP) for localization:",
    ]

    print("\n=== Quick metric lines ===")
    for k in keys:
        a = _extract(lines_on, k)
        b = _extract(lines_off, k)
        print(f"{k}")
        print(f"  enabled : {a}")
        print(f"  disabled: {b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
