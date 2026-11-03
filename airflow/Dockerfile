# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM apache/airflow:3.1.5-python3.13

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/credleaf /opt/credleaf-venv \
    && chown airflow:0 /opt/credleaf /opt/credleaf-venv \
    && chmod g+w /opt/credleaf /opt/credleaf-venv
COPY --from=uv /uv /usr/local/bin/uv

USER airflow
WORKDIR /opt/credleaf
COPY --chown=airflow:0 pyproject.toml uv.lock ./
COPY --chown=airflow:0 airflow/export_requirements.py /opt/export_requirements.py
# Airflow stays in the base image's interpreter; ML packages use their own venv.
RUN python /opt/export_requirements.py /opt/credleaf /tmp/pipeline-requirements.txt \
    && uv venv /opt/credleaf-venv --python /usr/local/bin/python \
    && uv pip install --no-cache --python /opt/credleaf-venv/bin/python \
         --torch-backend cpu -r /tmp/pipeline-requirements.txt

ARG SPACY_MODEL_VERSION=3.8.0
RUN uv pip install --no-cache --python /opt/credleaf-venv/bin/python --torch-backend cpu \
    -c /tmp/pipeline-requirements.txt \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-${SPACY_MODEL_VERSION}/en_core_web_trf-${SPACY_MODEL_VERSION}-py3-none-any.whl" \
    && rm /tmp/pipeline-requirements.txt

COPY --chown=airflow:0 src ./src
COPY --chown=airflow:0 scripts ./scripts
COPY --chown=airflow:0 dags ./dags
COPY --chown=airflow:0 airflow/start.sh ./airflow/start.sh
COPY --chown=airflow:0 airflow/plugins /opt/airflow/plugins
COPY --chown=airflow:0 airflow/catalog ./airflow/catalog
COPY --chown=airflow:0 notebooks/topic_tree_labels.json ./notebooks/topic_tree_labels.json

ENV CREDLEAF_ROOT=/opt/credleaf \
    CREDLEAF_PYTHON=/opt/credleaf-venv/bin/python \
    PYTHONPATH=/opt/credleaf/src \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/credleaf/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAG_RUN_CONF_OVERRIDES_PARAMS=True \
    XDG_CACHE_HOME=/opt/credleaf/data/cache \
    HF_HOME=/opt/credleaf/data/cache/huggingface \
    TORCH_HOME=/opt/credleaf/data/cache/torch \
    MPLBACKEND=Agg

CMD ["bash", "/opt/credleaf/airflow/start.sh"]
