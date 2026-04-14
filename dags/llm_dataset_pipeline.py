"""
DAG: ingest HF CNN/DailyMail -> validate raw -> Spark preprocess -> validate curated -> optional QLoRA.

Env:
  PIPELINE_ROOT   — repo root mounted in container (default /opt/airflow/project)
  PIPELINE_RAW_DIR, PIPELINE_CURATED_DIR — override paths
  PIPELINE_RUN_FINETUNE — set to 1 to run training task (needs GPU deps on worker)
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", "/opt/airflow/project")
RAW_DIR = os.environ.get("PIPELINE_RAW_DIR", f"{PIPELINE_ROOT}/data/raw")
CURATED_JSON_DIR = os.environ.get("PIPELINE_CURATED_DIR", f"{PIPELINE_ROOT}/data/curated/json")
MAX_ROWS = os.environ.get("PIPELINE_MAX_ROWS", "5000")
SPARK_MASTER = os.environ.get("PIPELINE_SPARK_MASTER", "local[*]")

default_args = {
    "owner": "bigdata-course",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


def _validate_raw_callable() -> None:
    import os
    import subprocess
    import sys

    root = os.environ.get("PIPELINE_ROOT", "/opt/airflow/project")
    raw = os.environ.get("PIPELINE_RAW_DIR", f"{root}/data/raw")
    cmd = [
        sys.executable,
        f"{root}/scripts/validate_raw.py",
        "--raw-dir",
        raw,
    ]
    subprocess.run(cmd, check=True)


def _validate_curated_callable() -> None:
    import os
    import subprocess
    import sys

    root = os.environ.get("PIPELINE_ROOT", "/opt/airflow/project")
    curated = os.environ.get("PIPELINE_CURATED_DIR", f"{root}/data/curated/json")
    metrics = os.environ.get("PIPELINE_METRICS_PATH", f"{root}/data/curated/metrics.json")
    cmd = [
        sys.executable,
        f"{root}/scripts/validate_curated.py",
        "--curated-dir",
        curated,
        "--metrics-out",
        metrics,
    ]
    subprocess.run(cmd, check=True)


with DAG(
    dag_id="llm_dataset_pipeline",
    default_args=default_args,
    description="HF ingest + Spark JSONL prep + optional finetune",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["spark", "llm", "ingest"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_hf_dataset",
        bash_command=(
            f'python "{PIPELINE_ROOT}/scripts/ingest_hf_dataset.py" '
            f'--output-dir "{RAW_DIR}" --max-rows {MAX_ROWS}'
        ),
        env={"PIPELINE_RAW_DIR": RAW_DIR},
    )

    validate_raw = PythonOperator(
        task_id="validate_raw_layout",
        python_callable=_validate_raw_callable,
    )

    spark_preprocess = BashOperator(
        task_id="spark_preprocess",
        bash_command=(
            f'spark-submit --master "{SPARK_MASTER}" '
            f'"{PIPELINE_ROOT}/spark/preprocess_to_jsonl.py" '
            f'--input "{RAW_DIR}/raw.parquet" --output "{CURATED_JSON_DIR}"'
        ),
    )

    validate_curated = PythonOperator(
        task_id="validate_curated_dataset",
        python_callable=_validate_curated_callable,
    )

    finetune = BashOperator(
        task_id="finetune_lora_optional",
        bash_command=r"""
set -e
if [ "${PIPELINE_RUN_FINETUNE:-0}" = "1" ]; then
  ROOT="${PIPELINE_ROOT:-/opt/airflow/project}"
  CUR="${PIPELINE_CURATED_DIR:-$ROOT/data/curated/json}"
  python "$ROOT/training/train_qlora.py" \
    --dataset-glob "$CUR/part-*.json.gz" \
    --output-dir "$ROOT/data/models/lora_run"
else
  echo "Skip finetune (set PIPELINE_RUN_FINETUNE=1 and install training/requirements-train.txt)"
fi
""",
    )

    ingest >> validate_raw >> spark_preprocess >> validate_curated >> finetune
