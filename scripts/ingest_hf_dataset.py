from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cnn_dailymail", help="HF datasets hub name")
    p.add_argument("--config", default="3.0.0", help="Dataset config (e.g. 3.0.0 for cnn_dailymail)")
    p.add_argument("--split", default="train", help="Split name")
    p.add_argument("--max-rows", type=int, default=5000, help="Cap rows for coursework-sized runs")
    p.add_argument(
        "--output-dir",
        default=os.environ.get("PIPELINE_RAW_DIR", "data/raw"),
        help="Directory for manifest + parquet shard",
    )
    args = p.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.max_rows is not None and args.max_rows > 0:
        n = min(args.max_rows, len(ds))
        ds = ds.select(range(n))

    out_parquet = out_dir / "raw.parquet"
    ds.to_parquet(str(out_parquet))

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "num_rows": len(ds),
        "columns": ds.column_names,
        "output_parquet": str(out_parquet),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest['num_rows']} rows to {manifest['output_parquet']}")


if __name__ == "__main__":
    main()
