
from datetime import datetime, timedelta
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from airflow.exceptions import AirflowException
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator


def _project_dir() -> str:
    return (os.environ.get("PROJECT_DIR") or "").strip() or "."


def _data_root() -> str:
    project_dir = (os.environ.get("PROJECT_DIR") or "").strip()
    if project_dir:
        return os.path.join(project_dir, "data")
    return "data"


def _spark_submit_cmd(script_name: str, cli_args: list[str]) -> list[str]:
    project = _project_dir()
    spark_master = (os.environ.get("PIPELINE_SPARK_MASTER") or "local[*]").strip()
    script_path = os.path.join(project, "spark", script_name)
    lib_path = os.path.join(project, "spark", "lib.py")
    if not os.path.isfile(script_path):
        raise AirflowException(f"Spark script not found: {script_path}")
    if not os.path.isfile(lib_path):
        raise AirflowException(f"Spark helper not found: {lib_path}")
    return [
        "spark-submit",
        "--master",
        spark_master,
        "--py-files",
        lib_path,
        script_path,
        *cli_args,
    ]


def _run_spark_submit(script_name: str, cli_args: list[str], timeout: int = 1800) -> None:
    cmd = _spark_submit_cmd(script_name, cli_args)
    proc = subprocess.run(
        cmd,
        cwd=_project_dir(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-6000:]
        raise AirflowException(
            f"spark-submit {script_name} exit {proc.returncode}. Cmd: {' '.join(cmd)}\n{tail}"
        )


_EXTRACT_PRESETS = {
    "smoke": {"page_size": 5, "pages": 1, "delay_seconds": 0},
    "test": {"page_size": 20, "pages": 1, "delay_seconds": 1},
    "full": {"page_size": 100, "pages": 2, "delay_seconds": 3},
}


def _resolve_scale(context: dict) -> str:
    dag_run = context.get("dag_run")
    conf = (getattr(dag_run, "conf", None) or {}) if dag_run else {}
    if not isinstance(conf, dict):
        conf = {}
    env_scale = (os.environ.get("NEWS_PIPELINE_SCALE") or "").strip().lower()
    params = context.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    scale = (
        conf.get("scale")
        or env_scale
        or params.get("scale")
        or "full"
    )
    if isinstance(scale, str):
        scale = scale.strip().lower()
    if scale not in _EXTRACT_PRESETS:
        scale = "full"
    return scale


_SFT_MAX_STEPS_BY_SCALE = {"smoke": 50, "test": 120, "full": 300}


def _sft_max_steps(context: dict) -> int:
    raw = (os.environ.get("SFT_MAX_STEPS") or "").strip()
    if raw.isdigit():
        return max(5, int(raw))
    scale = _resolve_scale(context)
    return _SFT_MAX_STEPS_BY_SCALE.get(scale, 80)


def extract_news(**context):
    ds = context.get("ds") or datetime.utcnow().strftime("%Y-%m-%d")
    run_id = context.get("run_id", "manual")

    scale = _resolve_scale(context)
    preset = _EXTRACT_PRESETS[scale]
    page_size = preset["page_size"]
    pages = preset["pages"]
    delay_seconds = preset["delay_seconds"]

    base_url = "http://export.arxiv.org/api/query"
    search_query = 'cat:cs.AI OR cat:cs.CL OR cat:cs.LG'

    output_dir = os.path.join(_data_root(), "bronze", "news", f"process_date={ds}")
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

            if delay_seconds and page < pages - 1:
                time.sleep(delay_seconds)

    return {
        "output_path": output_path,
        "saved_records": total_saved,
        "process_date": ds,
        "scale": scale,
    }


def _bronze_raw_path(context) -> str:
    ti = context.get("ti")
    if ti is not None:
        payload = ti.xcom_pull(task_ids="extract_news")
        if isinstance(payload, dict) and payload.get("output_path"):
            return payload["output_path"]
    ds = context.get("ds") or datetime.utcnow().strftime("%Y-%m-%d")
    return os.path.join(
        _data_root(), "bronze", "news", f"process_date={ds}", "arxiv_raw.jsonl"
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


def _ds(context: dict) -> str:
    return context.get("ds") or datetime.utcnow().strftime("%Y-%m-%d")


def _validated_path_from_context(context: dict) -> str:
    ti = context.get("ti")
    if ti is not None:
        report = ti.xcom_pull(task_ids="validate_raw")
        if isinstance(report, dict) and report.get("validated_path"):
            return report["validated_path"]
    ds = _ds(context)
    return os.path.join(
        _data_root(),
        "bronze",
        "news",
        f"process_date={ds}",
        "arxiv_validated.jsonl",
    )


def _silver_path_for_ds(ds: str) -> str:
    out_dir = os.path.join(_data_root(), "silver", "news", f"process_date={ds}")
    return os.path.join(out_dir, "articles.jsonl")


def _gold_path_for_ds(ds: str) -> str:
    out_dir = os.path.join(_data_root(), "gold", "news", f"process_date={ds}")
    return os.path.join(out_dir, "articles_enriched.jsonl")


def transform_bronze_to_silver(**context):
    ds = _ds(context)
    inp_path = _validated_path_from_context(context)
    if not os.path.isfile(inp_path):
        raise AirflowException(f"Validated bronze not found: {inp_path}")

    out_path = _silver_path_for_ds(ds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _run_spark_submit(
        "bronze_to_silver.py",
        ["--input", inp_path, "--output", out_path],
    )

    meta_path = os.path.join(os.path.dirname(out_path), "_transform_meta.json")
    if not os.path.isfile(meta_path):
        raise AirflowException(f"PySpark did not write meta: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta.get("rows") or 0)
    if n == 0:
        raise AirflowException("Silver transform (PySpark): no rows written")

    return {
        "output_path": out_path,
        "rows": n,
        "process_date": ds,
        "engine": "pyspark",
        "spark_app": meta.get("spark_app"),
    }


def _silver_path_from_context(context: dict) -> str:
    ti = context.get("ti")
    if ti is not None:
        payload = ti.xcom_pull(task_ids="transform_bronze_to_silver")
        if isinstance(payload, dict) and payload.get("output_path"):
            return payload["output_path"]
    return _silver_path_for_ds(_ds(context))


def enrich_silver_to_gold(**context):
    ds = _ds(context)
    inp_path = _silver_path_from_context(context)
    if not os.path.isfile(inp_path):
        raise AirflowException(f"Silver file not found: {inp_path}")

    out_path = _gold_path_for_ds(ds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _run_spark_submit(
        "silver_to_gold.py",
        ["--input", inp_path, "--output", out_path],
    )

    meta_path = os.path.join(os.path.dirname(out_path), "_enrich_meta.json")
    if not os.path.isfile(meta_path):
        raise AirflowException(f"PySpark did not write meta: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta.get("rows") or 0)
    if n == 0:
        raise AirflowException("Gold enrich (PySpark): no rows written")

    return {
        "output_path": out_path,
        "rows": n,
        "process_date": ds,
        "engine": "pyspark",
        "spark_app": meta.get("spark_app"),
    }


def _gold_path_from_context(context: dict) -> str:
    ti = context.get("ti")
    if ti is not None:
        payload = ti.xcom_pull(task_ids="enrich_silver_to_gold")
        if isinstance(payload, dict) and payload.get("output_path"):
            return payload["output_path"]
    return _gold_path_for_ds(_ds(context))


def build_sft_dataset(**context):
    ds = _ds(context)
    gold_path = _gold_path_from_context(context)
    if not os.path.isfile(gold_path):
        raise AirflowException(f"Gold file not found: {gold_path}")

    out_dir = os.path.join(_data_root(), "sft", "news", f"process_date={ds}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sft_train.jsonl")

    n = 0
    with open(gold_path, encoding="utf-8") as inp, open(out_path, "w", encoding="utf-8") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            body = (obj.get("body") or "").strip()
            if len(body) < 160:
                continue
            head = body[:1400]
            tail = body[1400:2800]
            if len(tail) < 50:
                tail = body[800:1500]
            if len(tail) < 50:
                continue
            user_content = (
                "Given the arXiv paper title and the beginning of the abstract, "
                "continue the abstract in English.\n\n"
                f"Title:\n{title}\n\nAbstract (prefix):\n{head}\n"
            )
            assistant_content = tail.strip()
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n += 1

    if n == 0:
        raise AirflowException("SFT build: no rows (need longer abstracts in gold)")

    return {"output_path": out_path, "rows": n, "process_date": ds}


def _sft_path_from_context(context: dict) -> str:
    ti = context.get("ti")
    if ti is not None:
        payload = ti.xcom_pull(task_ids="build_sft_dataset")
        if isinstance(payload, dict) and payload.get("output_path"):
            return payload["output_path"]
    ds = _ds(context)
    return os.path.join(
        _data_root(), "sft", "news", f"process_date={ds}", "sft_train.jsonl"
    )


def train_qwen_sft(**context):
    ds = _ds(context)
    sft_path = _sft_path_from_context(context)
    if not os.path.isfile(sft_path):
        raise AirflowException(f"SFT jsonl not found: {sft_path}")

    project_dir = (os.environ.get("PROJECT_DIR") or "").strip() or "."
    script = os.path.join(project_dir, "scripts", "sft.py")
    if not os.path.isfile(script):
        raise AirflowException(
            f"Скрипт обучения не найден: {script}. Соберите образ из Dockerfile.airflow."
        )

    out_dir = os.path.join(_data_root(), "models", "qwen_sft", f"process_date={ds}")
    os.makedirs(out_dir, exist_ok=True)
    max_steps = _sft_max_steps(context)

    hf_home = os.path.join(project_dir, "data", ".hf_cache")
    os.makedirs(hf_home, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", hf_home)
    env.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    model_name = (os.environ.get("SFT_MODEL_NAME") or "Qwen/Qwen2.5-0.5B-Instruct").strip()
    batch = (os.environ.get("SFT_BATCH_SIZE") or "1").strip()
    lr = (os.environ.get("SFT_LEARNING_RATE") or "").strip()

    cmd = [
        sys.executable,
        script,
        "--jsonl",
        sft_path,
        "--output_dir",
        out_dir,
        "--model_name",
        model_name,
        "--max_steps",
        str(max_steps),
        "--max_length",
        "512",
        "--batch_size",
        batch,
    ]
    if lr:
        cmd.extend(["--learning_rate", lr])

    proc = subprocess.run(
        cmd,
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=4 * 3600,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        raise AirflowException(f"sft.py exit {proc.returncode}. Tail:\n{tail}")

    manifest_path = os.path.join(out_dir, "train_manifest.json")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise AirflowException(f"Обучение не записало metrics.json: {metrics_path}")

    return {
        "output_dir": out_dir,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "process_date": ds,
        "max_steps": max_steps,
        "model_name": model_name,
    }


def evaluate_qwen_sft(**context):
    import math

    ds = _ds(context)
    ti = context.get("ti")
    train_payload = {}
    if ti is not None:
        train_payload = ti.xcom_pull(task_ids="train_qwen_sft") or {}
    if not isinstance(train_payload, dict):
        train_payload = {}

    metrics_path = train_payload.get("metrics_path") or os.path.join(
        _data_root(), "models", "qwen_sft", f"process_date={ds}", "metrics.json"
    )
    if not os.path.isfile(metrics_path):
        raise AirflowException(f"metrics.json not found: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    loss = metrics.get("train_loss")
    perplexity = None
    if isinstance(loss, (int, float)):
        try:
            perplexity = round(math.exp(loss), 4)
        except OverflowError:
            perplexity = None

    out_dir = os.path.join(_data_root(), "eval", "qwen_sft", f"process_date={ds}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "eval_report.json")
    report = {
        "process_date": ds,
        "mock": False,
        "train_loss": loss,
        "perplexity_estimate": perplexity,
        "metrics_source": metrics_path,
        "device": metrics.get("device"),
        "model_name": metrics.get("model_name"),
        "notes": "Метрики с train; отдельный test split не использовался.",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {"report_path": report_path, "process_date": ds}


def publish_artifacts(**context):
    ds = _ds(context)
    ti = context.get("ti")

    def pull(task_id: str) -> dict:
        if ti is None:
            return {}
        val = ti.xcom_pull(task_ids=task_id)
        return val if isinstance(val, dict) else {}

    extract_x = pull("extract_news")
    validate_x = pull("validate_raw")
    silver_x = pull("transform_bronze_to_silver")
    gold_x = pull("enrich_silver_to_gold")
    sft_x = pull("build_sft_dataset")
    train_x = pull("train_qwen_sft")
    eval_x = pull("evaluate_qwen_sft")

    out_dir = os.path.join(_data_root(), "publish", f"process_date={ds}")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "PIPELINE_MANIFEST.json")

    bundle = {
        "process_date": ds,
        "run_id": context.get("run_id", "manual"),
        "pipeline": "news_llm_arxiv_qwen_sft",
        "artifacts": {
            "bronze_raw": extract_x.get("output_path"),
            "bronze_validated": validate_x.get("validated_path"),
            "validation_report": (
                os.path.join(
                    os.path.dirname(validate_x["validated_path"]),
                    "validation_report.json",
                )
                if validate_x.get("validated_path")
                else None
            ),
            "silver": silver_x.get("output_path"),
            "gold": gold_x.get("output_path"),
            "sft_train": sft_x.get("output_path"),
            "qwen_checkpoint_dir": train_x.get("output_dir"),
            "qwen_train_manifest": train_x.get("manifest_path"),
            "qwen_metrics": train_x.get("metrics_path"),
            "eval_report": eval_x.get("report_path"),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    return {"publish_manifest": manifest_path, "process_date": ds}


default_args = {
    "owner": "1",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="news_llm_pipeline",
    description="Bronze→Gold (PySpark) + SFT Qwen2.5-0.5B на arXiv (HF Trainer, CUDA при наличии GPU)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    params={"scale": "full"},
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

    transform_bronze_to_silver = PythonOperator(
        task_id="transform_bronze_to_silver",
        python_callable=transform_bronze_to_silver,
        execution_timeout=timedelta(minutes=30),
    )

    enrich_silver_to_gold = PythonOperator(
        task_id="enrich_silver_to_gold",
        python_callable=enrich_silver_to_gold,
        execution_timeout=timedelta(minutes=30),
    )

    build_sft_dataset = PythonOperator(
        task_id="build_sft_dataset",
        python_callable=build_sft_dataset,
    )

    train_qwen_sft = PythonOperator(
        task_id="train_qwen_sft",
        python_callable=train_qwen_sft,
        execution_timeout=timedelta(hours=4),
    )

    evaluate_qwen_sft = PythonOperator(
        task_id="evaluate_qwen_sft",
        python_callable=evaluate_qwen_sft,
    )

    publish_artifacts = PythonOperator(
        task_id="publish_artifacts",
        python_callable=publish_artifacts,
    )
    end = EmptyOperator(task_id="end")

    (
        start
        >> extract_news
        >> validate_raw
        >> transform_bronze_to_silver
        >> enrich_silver_to_gold
        >> build_sft_dataset
        >> train_qwen_sft
        >> evaluate_qwen_sft
        >> publish_artifacts
        >> end
    )
