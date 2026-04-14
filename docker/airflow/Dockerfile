FROM apache/airflow:2.8.1-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY docker/airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

ENV PIPELINE_ROOT=/opt/airflow/project \
    PIPELINE_RAW_DIR=/opt/airflow/project/data/raw \
    PIPELINE_CURATED_DIR=/opt/airflow/project/data/curated/json \
    PIPELINE_MAX_ROWS=5000 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
