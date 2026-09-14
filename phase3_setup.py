from pathlib import Path

ROOT = Path(__file__).resolve().parent

INTELLIGENCE_CODE = '''
import argparse
from pathlib import Path

import pandas as pd

from src.config import settings


SEVERITY_SCORES = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100,
}

WEIGHTS = {
    "frequency": 0.25,
    "severity": 0.25,
    "trend": 0.20,
    "customer_value": 0.15,
    "recency": 0.15,
}


def normalize(series: pd.Series) -> pd.Series:
    """Convert values to a comparable 0–100 scale."""
    maximum = series.max()

    if maximum == 0:
        return pd.Series(0.0, index=series.index)

    return (series / maximum * 100).clip(0, 100)


def calculate_growth(
    recent_mentions: int,
    previous_mentions: int,
) -> float:
    """Calculate percentage change between two periods."""
    if previous_mentions == 0:
        return 100.0 if recent_mentions > 0 else 0.0

    return round(
        (
            (recent_mentions - previous_mentions)
            / previous_mentions
        )
        * 100,
        2,
    )


def assign_priority(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def build_topic_insights(
    data: pd.DataFrame,
    topic_column: str = "label",
) -> pd.DataFrame:
    """Calculate product metrics for every feedback topic."""
    frame = data.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"])

    frame["severity_score"] = (
        frame["severity_label"]
        .map(SEVERITY_SCORES)
        .fillna(25)
    )

    maximum_date = frame["created_at"].max()
    recent_start = maximum_date - pd.Timedelta(days=29)
    previous_start = recent_start - pd.Timedelta(days=30)

    recent = frame[frame["created_at"] >= recent_start]
    previous = frame[
        (frame["created_at"] >= previous_start)
        & (frame["created_at"] < recent_start)
    ]

    records = []

    for topic, topic_data in frame.groupby(topic_column):
        recent_count = int(
            (recent[topic_column] == topic).sum()
        )
        previous_count = int(
            (previous[topic_column] == topic).sum()
        )

        mentions = len(topic_data)
        growth = calculate_growth(
            recent_count,
            previous_count,
        )

        records.append(
            {
                "topic": topic,
                "mentions": mentions,
                "unique_customers": (
                    topic_data["customer_id"].nunique()
                ),
                "negative_percentage": round(
                    (
                        topic_data["sentiment_label"]
                        .eq("Negative")
                        .mean()
                    )
                    * 100,
                    2,
                ),
                "average_severity_score": round(
                    topic_data["severity_score"].mean(),
                    2,
                ),
                "affected_monthly_revenue": round(
                    topic_data["monthly_revenue"].sum(),
                    2,
                ),
                "average_customer_value": round(
                    topic_data["monthly_revenue"].mean(),
                    2,
                ),
                "recent_mentions": recent_count,
                "previous_mentions": previous_count,
                "growth_percentage": growth,
                "recency_ratio": round(
                    recent_count / mentions * 100,
                    2,
                ),
            }
        )

    insights = pd.DataFrame(records)

    insights["frequency_score"] = normalize(
        insights["mentions"]
    ).round(2)

    insights["severity_score"] = (
        insights["average_severity_score"]
        .clip(0, 100)
        .round(2)
    )

    insights["trend_score"] = (
        insights["growth_percentage"]
        .clip(lower=0, upper=200)
        .div(2)
        .round(2)
    )

    insights["customer_value_score"] = normalize(
        insights["average_customer_value"]
    ).round(2)

    insights["recency_score"] = normalize(
        insights["recency_ratio"]
    ).round(2)

    insights["opportunity_score"] = (
        insights["frequency_score"]
        * WEIGHTS["frequency"]
        + insights["severity_score"]
        * WEIGHTS["severity"]
        + insights["trend_score"]
        * WEIGHTS["trend"]
        + insights["customer_value_score"]
        * WEIGHTS["customer_value"]
        + insights["recency_score"]
        * WEIGHTS["recency"]
    ).round(2)

    insights["priority"] = (
        insights["opportunity_score"]
        .apply(assign_priority)
    )

    insights["emerging_issue"] = (
        (insights["recent_mentions"] >= 10)
        & (insights["growth_percentage"] >= 50)
    )

    return insights.sort_values(
        "opportunity_score",
        ascending=False,
    ).reset_index(drop=True)


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
        "--output",
        type=Path,
        default=(
            settings.processed_data_dir
            / "topic_insights.csv"
        ),
    )
    args = parser.parse_args()

    feedback = pd.read_csv(args.input)
    insights = build_topic_insights(feedback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    insights.to_csv(args.output, index=False)

    display_columns = [
        "topic",
        "mentions",
        "recent_mentions",
        "growth_percentage",
        "opportunity_score",
        "priority",
        "emerging_issue",
    ]

    print("\\nSignalDesk Product Intelligence")
    print("--------------------------------")
    print(insights[display_columns].to_string(index=False))
    print(f"\\nInsights saved to: {args.output}")


if __name__ == "__main__":
    main()
'''

TEST_CODE = '''
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
'''

(ROOT / "src/analysis/product_intelligence.py").write_text(
    INTELLIGENCE_CODE.strip() + "\n",
    encoding="utf-8",
)

(ROOT / "tests/test_product_intelligence.py").write_text(
    TEST_CODE.strip() + "\n",
    encoding="utf-8",
)

print("Phase 3 Product Intelligence Engine created.")