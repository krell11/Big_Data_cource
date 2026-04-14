# Big Data: Airflow + Spark → LLM fine-tuning dataset

End-to-end pipeline: **ingest** public CNN/Daily Mail via Hugging Face → **validate** raw layout → **PySpark** clean/dedup → **JSONL-style shards** (gzip) → optional **QLoRA** training script.

## Layout

| Path | Role |
|------|------|
| [dags/llm_dataset_pipeline.py](dags/llm_dataset_pipeline.py) | Airflow DAG |
| [scripts/ingest_hf_dataset.py](scripts/ingest_hf_dataset.py) | Landing Parquet + manifest |
| [scripts/validate_raw.py](scripts/validate_raw.py) / [scripts/validate_curated.py](scripts/validate_curated.py) | Quality gates + `metrics.json` |
| [spark/preprocess_to_jsonl.py](spark/preprocess_to_jsonl.py) | Distributed preprocess |
| [training/train_qlora.py](training/train_qlora.py) | LoRA/QLoRA (GPU; optional in DAG) |
| [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md) | JSONL schema for SFT |
| [docs/SOURCES.md](docs/SOURCES.md) | Data sources |
| [docs/REPORT.md](docs/REPORT.md) | Защита: схема, метрики, чеклист |

## Run with Docker (Airflow standalone)

```bash
docker compose build
docker compose up
```

Open `http://localhost:8080`. `airflow standalone` prints admin credentials in the container logs. Trigger DAG **`llm_dataset_pipeline`**.

Environment (override in `docker-compose.yml`):

- `PIPELINE_MAX_ROWS` — cap downloaded rows (default `5000`).
- `PIPELINE_RUN_FINETUNE=1` — runs the training task (requires PyTorch stack inside the image; default Airflow image does **not** include it — install [training/requirements-train.txt](training/requirements-train.txt) in a custom image or run training on a GPU host, see below).

Data persists in the `pipeline_data` volume under `/opt/airflow/project/data`.

## Run Spark training without Airflow

```bash
pip install -r docker/airflow/requirements-airflow.txt
python scripts/ingest_hf_dataset.py --output-dir data/raw --max-rows 2000
python scripts/validate_raw.py --raw-dir data/raw
spark-submit --master "local[*]" spark/preprocess_to_jsonl.py ^
  --input data/raw/raw.parquet --output data/curated/json
python scripts/validate_curated.py --curated-dir data/curated/json --metrics-out data/curated/metrics.json
```

## Fine-tune (GPU host)

```bash
pip install -r training/requirements-train.txt
set PIPELINE_BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python training/train_qlora.py --dataset-glob "data/curated/json/part-*.json.gz" --output-dir data/models/lora_run
```

Use a larger instruct model if VRAM allows; 4-bit path activates when CUDA is available.

## Docker note (JAVA_HOME)

The Airflow image is built for `amd64` with `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`. On ARM, adjust `JAVA_HOME` in [docker/airflow/Dockerfile](docker/airflow/Dockerfile) to match the OpenJDK path inside the container.
