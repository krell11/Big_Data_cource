from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir",
        default=None,
        help="Directory with manifest.json and raw.parquet (default: PIPELINE_RAW_DIR or data/raw)",
    )
    args = p.parse_args()

    import os

    raw_dir = Path(args.raw_dir or os.environ.get("PIPELINE_RAW_DIR", "data/raw"))
    manifest_path = raw_dir / "manifest.json"
    parquet_path = raw_dir / "raw.parquet"

    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not parquet_path.is_file():
        print(f"ERROR: missing {parquet_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cols = set(manifest.get("columns") or [])
    need = {"article", "highlights"}
    if not need.issubset(cols):
        print(f"ERROR: manifest columns missing {need - cols}: {cols}", file=sys.stderr)
        sys.exit(1)

    rows = int(manifest.get("num_rows") or 0)
    if rows <= 0:
        print("ERROR: num_rows must be > 0", file=sys.stderr)
        sys.exit(1)

    print(f"OK raw: {rows} rows, columns include article/highlights")


if __name__ == "__main__":
    main()
