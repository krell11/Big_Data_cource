from __future__ import annotations

import argparse
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf

from lib import write_meta, write_single_jsonl

_TOPIC_POOL = ["nlp", "deep-learning", "systems", "theory", "evaluation"]


@udf(ArrayType(StringType()))
def mock_topics_udf(title: str | None) -> list[str]:
    if not title:
        return ["unknown"]
    h = sum(ord(c) for c in title[:80]) % len(_TOPIC_POOL)
    return [_TOPIC_POOL[h], _TOPIC_POOL[(h + 2) % len(_TOPIC_POOL)]]


def main() -> None:
    p = argparse.ArgumentParser(description="Silver JSONL → gold JSONL (PySpark)")
    p.add_argument("--input", required=True, help="Path to articles.jsonl (silver)")
    p.add_argument("--output", required=True, help="Path to articles_enriched.jsonl")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    spark = (
        SparkSession.builder.appName("arxiv_silver_to_gold")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        df = spark.read.json(args.input)
        required = {"title", "body", "body_len"}
        missing = required - set(df.columns)
        if missing:
            print(f"ERROR: missing columns {missing}, have {df.columns}", file=sys.stderr)
            sys.exit(2)

        title = F.coalesce(F.col("title"), F.lit(""))
        body_len = F.coalesce(F.col("body_len"), F.length(F.coalesce(F.col("body"), F.lit("")))).cast(
            "int"
        )
        mock_score = F.round(F.lit(0.55) + F.least(body_len, F.lit(8000)) / F.lit(20000.0), 4)

        gold = (
            df.withColumn("mock_topics", mock_topics_udf(title))
            .withColumn("mock_quality_score", mock_score)
            .withColumn(
                "mock_summary",
                F.concat(
                    F.lit("[MOCK] Статья «"),
                    F.substring(title, 1, 120),
                    F.lit("»; ~"),
                    body_len.cast("string"),
                    F.lit(" символов текста."),
                ),
            )
            .withColumn("gold_schema_version", F.lit(1))
            .withColumn("layer", F.lit("gold"))
        )

        rows = write_single_jsonl(gold, args.output)
        meta_path = os.path.join(os.path.dirname(args.output), "_enrich_meta.json")
        write_meta(
            meta_path,
            {
                "rows": rows,
                "engine": "pyspark",
                "spark_app": "arxiv_silver_to_gold",
                "input": args.input,
                "output": args.output,
            },
        )
        print(f"Wrote {rows} gold rows to {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
