"""
Airflow DAG scaffold for News -> Spark -> SFT pipeline.

This file contains only orchestration skeleton.
Fill business logic in referenced modules yourself.
"""

from datetime import datetime, timedelta
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from airflow.exceptions import AirflowException
from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator


def extract_news(**context):
    ds = context.get("ds") or datetime.utcnow().strftime("%Y-%m-%d")
    run_id = context.get("run_id", "manual")

    base_url = "http://export.arxiv.org/api/query"
    search_query = 'cat:cs.AI OR cat:cs.CL OR cat:cs.LG'
    page_size = 100
    pages = 2
    delay_seconds = 3

    output_dir = os.path.join("data", "bronze", "news", f"process_date={ds}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "arxiv_raw.jsonl")

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    total_saved = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for page in range(pages):
            params = {
                "search_query": search_query,
                "start": page * page_size,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "BigDataCourseProject/1.0 (educational use)"},
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                xml_payload = response.read()

            root = ET.fromstring(xml_payload)
            entries = root.findall("atom:entry", namespace)
            if not entries:
                break

            for entry in entries:
                title = entry.findtext("atom:title", default="", namespaces=namespace).strip()
                summary = entry.findtext("atom:summary", default="", namespaces=namespace).strip()
                published = entry.findtext(
                    "atom:published", default="", namespaces=namespace
                ).strip()
                updated = entry.findtext("atom:updated", default="", namespaces=namespace).strip()
                paper_id = entry.findtext("atom:id", default="", namespaces=namespace).strip()

                authors = [
                    a.findtext("atom:name", default="", namespaces=namespace).strip()
                    for a in entry.findall("atom:author", namespace)
                ]

                record = {
                    "source": "arxiv",
                    "url": paper_id,
                    "title": title,
                    "content": summary,
                    "authors": authors,
                    "published_at": published,
                    "updated_at": updated,
                    "process_date": ds,
                    "run_id": run_id,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_saved += 1

            time.sleep(delay_seconds)

    return {
        "output_path": output_path,
        "saved_records": total_saved,
        "process_date": ds,
    }


def _bronze_raw_path(context) -> str:
    ti = context.get("ti")
    if ti is not None:
        payload = ti.xcom_pull(task_ids="extract_news")
        if isinstance(payload, dict) and payload.get("output_path"):
            return payload["output_path"]
    ds = context.get("ds") or datetime.utcnow().strftime("%Y-%m-%d")
    return os.path.join(
        "data", "bronze", "news", f"process_date={ds}", "arxiv_raw.jsonl"
    )


def _is_valid_arxiv_record(obj: dict) -> bool:
    required = (
        "source",
        "url",
        "title",
        "content",
        "published_at",
        "process_date",
    )
    if not isinstance(obj, dict):
        return False
    if any(k not in obj for k in required):
        return False
    if not isinstance(obj.get("title"), str) or len(obj["title"].strip()) < 5:
        return False
    if not isinstance(obj.get("content"), str) or len(obj["content"].strip()) < 20:
        return False
    url = obj.get("url", "")
    if not isinstance(url, str) or "arxiv" not in url.lower():
        return False
    published = obj.get("published_at", "")
    if not isinstance(published, str) or not published.strip():
        return False
    try:
        datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return False
    authors = obj.get("authors", [])
    if authors is not None and not isinstance(authors, list):
        return False
    if authors:
        if not all(isinstance(a, str) for a in authors):
            return False
    return True


def validate_raw(**context):
    raw_path = _bronze_raw_path(context)
    if not os.path.isfile(raw_path):
        raise AirflowException(f"Bronze raw file not found: {raw_path}")

    out_dir = os.path.dirname(raw_path)
    validated_path = os.path.join(out_dir, "arxiv_validated.jsonl")
    report_path = os.path.join(out_dir, "validation_report.json")

    total = 0
    valid = 0
    invalid = 0

    with open(raw_path, encoding="utf-8") as inp, open(
        validated_path, "w", encoding="utf-8"
    ) as outp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if _is_valid_arxiv_record(obj):
                valid += 1
                outp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                invalid += 1

    if total == 0:
        raise AirflowException(f"No lines read from {raw_path}")

    invalid_ratio = invalid / total
    max_invalid_ratio = 0.5
    if valid == 0:
        raise AirflowException("No valid records after validation")
    if invalid_ratio > max_invalid_ratio:
        raise AirflowException(
            f"Too many invalid rows: {invalid}/{total} "
            f"({invalid_ratio:.1%} > {max_invalid_ratio:.0%})"
        )

    report = {
        "raw_path": raw_path,
        "validated_path": validated_path,
        "total_lines": total,
        "valid": valid,
        "invalid": invalid,
        "invalid_ratio": round(invalid_ratio, 4),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


default_args = {
    "owner": "1",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="news_llm_pipeline",
    description="Scaffold DAG: Airflow + Spark + SFT/LoRA",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["big-data", "airflow", "spark", "llm"],
) as dag:
    start = EmptyOperator(task_id="start")

    extract_news = PythonOperator(
        task_id="extract_news",
        python_callable=extract_news,
    )

    validate_raw = PythonOperator(
        task_id="validate_raw",
        python_callable=validate_raw,
    )

    # TODO: replace with Spark submit operator or custom operator.
    spark_transform_bronze_to_silver = EmptyOperator(
        task_id="spark_transform_bronze_to_silver"
    )

    # TODO: replace with Spark submit operator or custom operator.
    spark_enrich_silver_to_gold = EmptyOperator(
        task_id="spark_enrich_silver_to_gold"
    )

    # TODO: replace EmptyOperator with python/shell task calling your script.
    build_sft_dataset = EmptyOperator(task_id="build_sft_dataset")
    train_lora_model = EmptyOperator(task_id="train_lora_model")
    evaluate_model = EmptyOperator(task_id="evaluate_model")
    publish_artifacts = EmptyOperator(task_id="publish_artifacts")
    end = EmptyOperator(task_id="end")

    (
        start
        >> extract_news
        >> validate_raw
        >> spark_transform_bronze_to_silver
        >> spark_enrich_silver_to_gold
        >> build_sft_dataset
        >> train_lora_model
        >> evaluate_model
        >> publish_artifacts
        >> end
    )
