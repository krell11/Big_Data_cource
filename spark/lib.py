from __future__ import annotations

import glob
import json
import os
import shutil

from pyspark.sql import DataFrame


def write_single_jsonl(df: DataFrame, output_path: str) -> int:
    tmp_dir = output_path + "._spark_tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    row_count = df.count()
    if row_count == 0:
        raise RuntimeError("write_single_jsonl: no rows to write")

    df.coalesce(1).write.mode("overwrite").json(tmp_dir)
    parts = sorted(glob.glob(os.path.join(tmp_dir, "part-*.json*")))
    if not parts:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"No part-*.json files under {tmp_dir}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for part in parts:
            with open(part, encoding="utf-8") as inp:
                shutil.copyfileobj(inp, out)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return row_count


def write_meta(meta_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
