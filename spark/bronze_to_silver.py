from __future__ import annotations

import argparse
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from lib import write_meta, write_single_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Bronze validated JSONL → silver JSONL (PySpark)")
    p.add_argument("--input", required=True, help="Path to arxiv_validated.jsonl")
    p.add_argument("--output", required=True, help="Path to articles.jsonl")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    spark = (
        SparkSession.builder.appName("arxiv_bronze_to_silver")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        df = spark.read.json(args.input)
        required = {"url", "title", "content", "published_at", "process_date"}
        missing = required - set(df.columns)
        if missing:
            print(f"ERROR: missing columns {missing}, have {df.columns}", file=sys.stderr)
            sys.exit(2)

        title = F.trim(F.col("title"))
        body = F.trim(F.col("content"))
        authors = F.coalesce(
            F.col("authors"),
            F.array().cast(ArrayType(StringType())),
        )

        silver = df.select(
            F.col("url"),
            title.alias("title"),
            body.alias("body"),
            authors.alias("authors"),
            F.col("published_at"),
            F.col("process_date"),
            F.length(title).alias("title_len"),
            F.length(body).alias("body_len"),
            F.size(authors).alias("n_authors"),
            F.lit("silver").alias("layer"),
        )

        rows = write_single_jsonl(silver, args.output)
        meta_path = os.path.join(os.path.dirname(args.output), "_transform_meta.json")
        write_meta(
            meta_path,
            {
                "rows": rows,
                "engine": "pyspark",
                "spark_app": "arxiv_bronze_to_silver",
                "input": args.input,
                "output": args.output,
            },
        )
        print(f"Wrote {rows} silver rows to {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
