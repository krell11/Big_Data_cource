from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


REQUIRED_KEYS = {"instruction", "input", "output", "source_id"}


def _open_text(fp: Path):
    if fp.suffix == ".gz":
        return gzip.open(fp, "rt", encoding="utf-8")
    return open(fp, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--curated-dir", required=True, help="Spark output dir (contains part-*.json.gz or .json)")
    p.add_argument("--max-files", type=int, default=8, help="How many part files to scan for schema checks")
    p.add_argument(
        "--metrics-out",
        default=None,
        help="If set, write full JSON line count + file stats to this path (scans all part files)",
    )
    args = p.parse_args()

    root = Path(args.curated_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    parts = sorted(root.glob("part-*.json.gz")) + sorted(root.glob("part-*.json"))
    if not parts:
        print(f"ERROR: no part-*.json(.gz) under {root}", file=sys.stderr)
        sys.exit(1)

    n_lines = 0
    for fp in parts[: args.max_files]:
        with _open_text(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not REQUIRED_KEYS.issubset(obj.keys()):
                    print(f"ERROR: bad keys in {fp}: {obj.keys()}", file=sys.stderr)
                    sys.exit(1)
                if not str(obj.get("output", "")).strip():
                    print(f"ERROR: empty output in {fp}", file=sys.stderr)
                    sys.exit(1)
                n_lines += 1

    if n_lines == 0:
        print("ERROR: no JSON lines parsed", file=sys.stderr)
        sys.exit(1)
    print(f"OK curated: sampled {n_lines} lines from {min(len(parts), args.max_files)} files")

    if args.metrics_out:
        total = 0
        for fp in parts:
            with _open_text(fp) as f:
                for line in f:
                    if line.strip():
                        total += 1
        metrics = {
            "curated_dir": str(root),
            "part_files": len(parts),
            "json_lines": total,
        }
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {args.metrics_out} (json_lines={total})")


if __name__ == "__main__":
    main()
