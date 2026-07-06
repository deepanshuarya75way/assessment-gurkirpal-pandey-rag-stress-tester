# src/evaluation/mlflow_logger.py
# MLFLOW EXPERIMENT TRACKING
# Logs all experiment parameters and metrics
# to MLflow for easy comparison and visualization

import mlflow
import os
import json
from datetime import datetime


def log_experiment(config: dict, metrics: dict,
                   config_path: str = None,
                   per_question_results: list = None):
    """
    Logs one experiment run to MLflow.

    Parameters:
        config: the yaml config dict
        metrics: recall@k, MRR, context precision scores
        config_path: path to config yaml file
        per_question_results: per-question breakdown
    """

    # set experiment name
    mlflow.set_experiment("RAG-Pipeline-Stress-Tester")

    with mlflow.start_run():

        # ── LOG PARAMETERS ──────────────────────────
        # these are the settings we varied
        mlflow.log_param("chunker", config.get("chunker"))
        mlflow.log_param("embedder", config.get("embedder"))
        mlflow.log_param("retriever", config.get("retriever"))

        # chunker params
        chunker_params = config.get("chunker_params", {})
        mlflow.log_param(
            "chunk_size",
            chunker_params.get("chunk_size", "N/A"))
        mlflow.log_param(
            "chunk_overlap",
            chunker_params.get("chunk_overlap", "N/A"))

        # retriever params
        retriever_params = config.get("retriever_params", {})
        mlflow.log_param(
            "top_k",
            retriever_params.get("top_k", 5))
        mlflow.log_param(
            "alpha",
            retriever_params.get("alpha", "N/A"))

        # reranker
        mlflow.log_param(
            "reranker_depth",
            config.get("reranker_depth", "none"))

        # config file name
        if config_path:
            mlflow.log_param(
                "config_file",
                os.path.basename(config_path))

        # ── LOG METRICS ──────────────────────────────
        # these are the scores we measured
        mlflow.log_metric("recall_at_1",
                          metrics.get("recall@1", 0))
        mlflow.log_metric("recall_at_3",
                          metrics.get("recall@3", 0))
        mlflow.log_metric("recall_at_5",
                          metrics.get("recall@5", 0))
        mlflow.log_metric("mrr",
                          metrics.get("mrr", 0))
        mlflow.log_metric("context_precision",
                          metrics.get("context_precision", 0))

        # ── LOG ARTIFACTS ────────────────────────────
        # save per-question results as artifact
        if per_question_results:
            artifact_path = "temp_results.json"
            with open(artifact_path, 'w') as f:
                json.dump(per_question_results, f, indent=2)
            mlflow.log_artifact(artifact_path)
            os.remove(artifact_path)

        print(f"MLflow: experiment logged successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


def log_all_existing_results():
    """
    Reads all existing result JSON files and
    logs them all to MLflow retroactively.
    This catches up all 20 experiments we already ran!
    """
    results_dir = "results"
    logged = 0

    mlflow.set_experiment("RAG-Pipeline-Stress-Tester")

    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith('.json'):
            continue
        if 'findings' in filename or 'notes' in filename:
            continue

        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # reconstruct config dict from result
            config = {
                "chunker": data.get("chunker", "unknown"),
                "embedder": data.get("embedder", "unknown"),
                "retriever": data.get("retriever", "unknown"),
                "reranker_depth": data.get("reranker_depth"),
                "chunker_params": {},
                "retriever_params": {}
            }

            metrics = data.get("metrics", {})
            per_q = data.get("per_question_results", [])

            with mlflow.start_run(
                    run_name=filename.replace('.json', '')):

                mlflow.log_param("chunker", config["chunker"])
                mlflow.log_param("embedder", config["embedder"])
                mlflow.log_param("retriever", config["retriever"])
                mlflow.log_param(
                    "reranker_depth",
                    config["reranker_depth"] or "none")
                mlflow.log_param("source_file", filename)

                mlflow.log_metric(
                    "recall_at_1", metrics.get("recall@1", 0))
                mlflow.log_metric(
                    "recall_at_3", metrics.get("recall@3", 0))
                mlflow.log_metric(
                    "recall_at_5", metrics.get("recall@5", 0))
                mlflow.log_metric(
                    "mrr", metrics.get("mrr", 0))
                mlflow.log_metric(
                    "context_precision",
                    metrics.get("context_precision", 0))

            logged += 1
            print(f"Logged: {filename}")

        except Exception as e:
            print(f"Skipped {filename}: {e}")

    print(f"\nTotal logged to MLflow: {logged} experiments")
    return logged


if __name__ == "__main__":
    print("Logging all existing results to MLflow...")
    print("This catches up all 20 experiments!\n")

    count = log_all_existing_results()

    print(f"\nDone! {count} experiments logged.")
    print(f"\nTo view MLflow dashboard:")
    print(f"Run: mlflow ui")
    print(f"Then open: http://localhost:5000")