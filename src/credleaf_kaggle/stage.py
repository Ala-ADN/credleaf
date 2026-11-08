"""Kaggle execution backend for the corpus DAG's single embed task."""
import os
from pathlib import Path

from credleaf_kaggle.client import KaggleCLI
from credleaf_kaggle.jobs import advance, job_lock, load_job, prepare
from credleaf_kaggle.results import write_embedding
from pipeline.runner import StagePending


def embed_snapshot(request):
    config = request["config"]
    directory = Path(request["checkpoint_dir"]) / "kaggle"
    job = prepare(request["root"], directory, owner=os.environ["CREDLEAF_KAGGLE_OWNER"],
        inputs=request["inputs"], source_root=request["code_root"],
        model=config["embed_model"], revision=config["kaggle_revision"],
        shard_documents=config["kaggle_shard_documents"],
        encode_batch_size=config["kaggle_encode_batch_size"], write_batch_size=config["embed_batch_size"],
        accelerator=config["kaggle_accelerator"], kernel_timeout=config["kaggle_kernel_timeout"])
    print(f"Kaggle job {job['job_id']}: {directory}", flush=True)
    if not advance(directory, KaggleCLI()):
        job = load_job(directory)
        done = sum(shard["status"] == "downloaded" for shard in job["shards"])
        raise StagePending(f"Downloaded {done}/{len(job['shards'])} Kaggle shards; work directory: {directory}")
    with job_lock(directory):
        return write_embedding(directory, request["output"], qdrant_url=config["qdrant_url"],
                               qdrant_prefix=config["qdrant_prefix"], batch_size=config["embed_batch_size"])
