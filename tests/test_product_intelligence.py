import pandas as pd

from src.analysis.product_intelligence import (
    assign_priority,
    build_topic_insights,
    calculate_growth,
)


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feedback_id": "FB-1",
                "created_at": "2026-08-25",
                "customer_id": "CUST-1",
                "label": "Payment Failure",
                "sentiment_label": "Negative",
                "severity_label": "Critical",
                "monthly_revenue": 50000,
            },
            {
                "feedback_id": "FB-2",
                "created_at": "2026-08-20",
                "customer_id": "CUST-2",
                "label": "Payment Failure",
                "sentiment_label": "Negative",
                "severity_label": "Critical",
                "monthly_revenue": 70000,
            },
            {
                "feedback_id": "FB-3",
                "created_at": "2026-07-25",
                "customer_id": "CUST-3",
                "label": "Payment Failure",
                "sentiment_label": "Negative",
                "severity_label": "Critical",
                "monthly_revenue": 30000,
            },
            {
                "feedback_id": "FB-4",
                "created_at": "2026-08-22",
                "customer_id": "CUST-4",
                "label": "Invoice Export",
                "sentiment_label": "Neutral",
                "severity_label": "Medium",
                "monthly_revenue": 20000,
            },
        ]
    )


def test_growth_calculation():
    assert calculate_growth(30, 20) == 50.0


def test_growth_when_previous_period_is_zero():
    assert calculate_growth(10, 0) == 100.0


def test_priority_thresholds():
    assert assign_priority(85) == "Critical"
    assert assign_priority(70) == "High"
    assert assign_priority(50) == "Medium"
    assert assign_priority(30) == "Low"


def test_builds_one_row_per_topic():
    insights = build_topic_insights(sample_data())
    assert len(insights) == 2


def test_opportunity_scores_remain_bounded():
    insights = build_topic_insights(sample_data())
    assert insights["opportunity_score"].between(0, 100).all()


def test_payment_failure_has_more_mentions():
    insights = build_topic_insights(sample_data())
    payment = insights[
        insights["topic"] == "Payment Failure"
    ].iloc[0]

    assert payment["mentions"] == 3
    assert payment["unique_customers"] == 3
