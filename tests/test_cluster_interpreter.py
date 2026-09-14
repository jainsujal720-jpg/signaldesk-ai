import pandas as pd

from src.analysis.cluster_interpreter import (
    ClusterInsight,
    ClusterInsightCollection,
    prepare_cluster_evidence,
    validate_evidence,
)


def sample_feedback() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "customer_id": "CUST-1",
                "feedback_text": "Payment failed.",
                "sentiment_label": "Negative",
                "label": "Payment Failure",
            },
            {
                "cluster_id": 0,
                "customer_id": "CUST-2",
                "feedback_text": "I was charged twice.",
                "sentiment_label": "Negative",
                "label": "Payment Failure",
            },
        ]
    )


def test_prepares_evidence_without_golden_labels():
    clusters, _ = prepare_cluster_evidence(
        sample_feedback()
    )

    assert "label" not in clusters[0]
    assert clusters[0]["feedback_count"] == 2


def test_removes_invented_evidence():
    _, allowed = prepare_cluster_evidence(
        sample_feedback()
    )

    collection = ClusterInsightCollection(
        insights=[
            ClusterInsight(
                cluster_id=0,
                topic_name="Payment problems",
                summary="Customers report payment problems.",
                severity="Critical",
                evidence=[
                    "Payment failed.",
                    "This comment was invented.",
                ],
                recommended_action="Investigate payments.",
                confidence=0.9,
            )
        ]
    )

    validated = validate_evidence(
        collection,
        allowed,
    )

    assert validated.insights[0].evidence == [
        "Payment failed."
    ]
