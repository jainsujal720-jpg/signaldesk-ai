import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from src.config import settings


def get_embeddings(
    texts: list[str],
    cache_path: Path,
) -> dict[str, list[float]]:
    """Create or load embeddings for unique feedback texts."""
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")

        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from the .env file."
        )

    client = OpenAI(api_key=settings.openai_api_key)

    print(
        f"Requesting embeddings for {len(texts)} "
        "unique feedback messages..."
    )

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    embedding_map = {
        text: item.embedding
        for text, item in zip(texts, response.data)
    }

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(embedding_map, file)

    print(f"Embeddings saved to {cache_path}")

    return embedding_map


def majority_cluster_labels(
    true_labels: list[str],
    cluster_ids: list[int],
) -> dict[int, str]:
    """Give each cluster its most common golden topic name."""
    grouped_labels: dict[int, list[str]] = {}

    for true_label, cluster_id in zip(
        true_labels,
        cluster_ids,
    ):
        grouped_labels.setdefault(
            int(cluster_id),
            [],
        ).append(true_label)

    return {
        cluster_id: Counter(labels).most_common(1)[0][0]
        for cluster_id, labels in grouped_labels.items()
    }


def calculate_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    cluster_ids: list[int],
) -> dict[str, float]:
    """Evaluate clustering against the golden labels."""
    return {
        "cluster_label_accuracy": round(
            accuracy_score(
                true_labels,
                predicted_labels,
            ),
            4,
        ),
        "adjusted_rand_index": round(
            adjusted_rand_score(
                true_labels,
                cluster_ids,
            ),
            4,
        ),
        "normalized_mutual_information": round(
            normalized_mutual_info_score(
                true_labels,
                cluster_ids,
            ),
            4,
        ),
    }


def build_cluster_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a readable summary for each discovered cluster."""
    records = []

    for cluster_id, cluster_data in data.groupby(
        "cluster_id"
    ):
        sample_comments = (
            cluster_data["feedback_text"]
            .drop_duplicates()
            .head(3)
            .tolist()
        )

        records.append(
            {
                "cluster_id": int(cluster_id),
                "predicted_topic": (
                    cluster_data["predicted_label"].mode()[0]
                ),
                "records": len(cluster_data),
                "unique_comments": (
                    cluster_data["feedback_text"].nunique()
                ),
                "dominant_golden_label": (
                    cluster_data["label"].mode()[0]
                ),
                "sample_feedback": " | ".join(
                    sample_comments
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "records",
        ascending=False,
    )


def cluster_feedback(
    data: pd.DataFrame,
    number_of_clusters: int = 8,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Group feedback by meaning using embeddings."""
    frame = data.copy()

    unique_texts = (
        frame["feedback_text"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if cache_path is None:
        cache_path = (
            settings.processed_data_dir
            / "unique_embeddings.json"
        )

    embedding_map = get_embeddings(
        unique_texts,
        cache_path,
    )

    unique_vectors = np.array(
        [embedding_map[text] for text in unique_texts]
    )

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=20,
    )

    unique_cluster_ids = model.fit_predict(
        unique_vectors
    )

    text_to_cluster = dict(
        zip(unique_texts, unique_cluster_ids)
    )

    frame["cluster_id"] = (
        frame["feedback_text"]
        .map(text_to_cluster)
        .astype(int)
    )

    cluster_names = majority_cluster_labels(
        frame["label"].tolist(),
        frame["cluster_id"].tolist(),
    )

    frame["predicted_label"] = (
        frame["cluster_id"].map(cluster_names)
    )

    metrics = calculate_metrics(
        frame["label"].tolist(),
        frame["predicted_label"].tolist(),
        frame["cluster_id"].tolist(),
    )

    summary = build_cluster_summary(frame)

    return frame, summary, metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            settings.processed_data_dir
            / "flowpay_feedback_clean.csv"
        ),
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    feedback = pd.read_csv(args.input)

    clustered, summary, metrics = cluster_feedback(
        feedback,
        number_of_clusters=args.clusters,
    )

    clustered_path = (
        settings.processed_data_dir
        / "feedback_with_clusters.csv"
    )

    summary_path = (
        settings.processed_data_dir
        / "cluster_summary.csv"
    )

    metrics_path = (
        settings.processed_data_dir
        / "embedding_metrics.json"
    )

    clustered.to_csv(
        clustered_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nDiscovered semantic clusters")
    print("----------------------------")

    print(
        summary[
            [
                "cluster_id",
                "predicted_topic",
                "records",
                "unique_comments",
            ]
        ].to_string(index=False)
    )

    print("\nEvaluation metrics")
    print("------------------")

    for metric, value in metrics.items():
        readable_name = metric.replace("_", " ").title()
        print(f"{readable_name}: {value}")

    print(f"\nClustered feedback: {clustered_path}")
    print(f"Cluster summary: {summary_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
