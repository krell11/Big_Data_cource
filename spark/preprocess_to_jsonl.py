from __future__ import annotations

import argparse
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


INSTRUCTION = ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw.parquet or parquet directory")
    p.add_argument("--output", required=True, help="Output directory for Spark JSON (part-*.json)")
    p.add_argument("--partitions", type=int, default=4, help="Output partition count")
    p.add_argument("--min-article-chars", type=int, default=200)
    p.add_argument("--min-summary-chars", type=int, default=10)
    args = p.parse_args()

    spark = SparkSession.builder.appName("llm_dataset_preprocess").getOrCreate()
    try:
        df = spark.read.parquet(args.input)

        required = {"article", "highlights"}
        cols = set(df.columns)
        missing = required - cols
        if missing:
            print(f"ERROR: missing columns {missing}, have {cols}", file=sys.stderr)
            sys.exit(2)

        if "id" not in df.columns:
            df = df.withColumn("id", F.monotonically_increasing_id().cast("string"))

        a = F.trim(F.col("article"))
        h = F.trim(F.col("highlights"))
        df = (
            df.filter(F.length(a) >= args.min_article_chars)
            .filter(F.length(h) >= args.min_summary_chars)
            .withColumn("_h", F.sha2(F.col("article"), 256))
            .dropDuplicates(["_h"])
            .drop("_h")
            .withColumn("instruction", F.lit(INSTRUCTION))
            .withColumn("input", F.concat(F.lit("## Article\n\n"), F.col("article")))
            .withColumn("output", F.col("highlights"))
            .withColumn("source_id", F.concat(F.lit("cnn_dailymail:"), F.col("id").cast("string")))
        )

        out = df.select("instruction", "input", "output", "source_id")
        (
            out.repartition(args.partitions)
            .write.mode("overwrite")
            .option("compression", "gzip")
            .json(args.output)
        )

        cnt = out.count()
        print(f"Wrote {cnt} rows to {args.output} ({args.partitions} partitions)")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
