from src.analysis.semantic_clustering import (
    calculate_metrics,
    majority_cluster_labels,
)


def test_majority_cluster_labels():
    true_labels = [
        "Payment",
        "Payment",
        "Login",
        "Login",
    ]

    cluster_ids = [0, 0, 1, 1]

    mapping = majority_cluster_labels(
        true_labels,
        cluster_ids,
    )

    assert mapping == {
        0: "Payment",
        1: "Login",
    }


def test_perfect_clustering_metrics():
    true_labels = [
        "Payment",
        "Payment",
        "Login",
        "Login",
    ]

    predicted_labels = [
        "Payment",
        "Payment",
        "Login",
        "Login",
    ]

    cluster_ids = [0, 0, 1, 1]

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
        cluster_ids,
    )

    assert metrics["cluster_label_accuracy"] == 1.0
    assert metrics["adjusted_rand_index"] == 1.0
    assert metrics["normalized_mutual_information"] == 1.0
