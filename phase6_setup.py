from pathlib import Path

ROOT = Path(__file__).resolve().parent

INTERPRETER_CODE = '''
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import settings


class ClusterInsight(BaseModel):
    cluster_id: int
    topic_name: str = Field(
        description="Short product topic name"
    )
    summary: str = Field(
        description="Evidence-based summary of the customer problem"
    )
    severity: Literal[
        "Low",
        "Medium",
        "High",
        "Critical",
    ]
    evidence: list[str] = Field(
        description="Exact customer comments from the supplied evidence"
    )
    recommended_action: str
    confidence: float = Field(
        ge=0,
        le=1,
    )


class ClusterInsightCollection(BaseModel):
    insights: list[ClusterInsight]


def prepare_cluster_evidence(
    data: pd.DataFrame,
    samples_per_cluster: int = 8,
) -> tuple[list[dict], dict[int, set[str]]]:
    """Prepare feedback evidence without exposing golden labels."""
    clusters = []
    allowed_evidence = {}

    for cluster_id, cluster_data in data.groupby(
        "cluster_id"
    ):
        comments = (
            cluster_data["feedback_text"]
            .drop_duplicates()
            .head(samples_per_cluster)
            .tolist()
        )

        allowed_evidence[int(cluster_id)] = set(comments)

        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "feedback_count": len(cluster_data),
                "unique_customers": (
                    cluster_data["customer_id"].nunique()
                ),
                "negative_percentage": round(
                    cluster_data["sentiment_label"]
                    .eq("Negative")
                    .mean()
                    * 100,
                    2,
                ),
                "sample_comments": comments,
            }
        )

    return clusters, allowed_evidence


def validate_evidence(
    collection: ClusterInsightCollection,
    allowed_evidence: dict[int, set[str]],
) -> ClusterInsightCollection:
    """Remove evidence that was not present in customer feedback."""
    for insight in collection.insights:
        allowed = allowed_evidence.get(
            insight.cluster_id,
            set(),
        )

        insight.evidence = [
            comment
            for comment in insight.evidence
            if comment in allowed
        ]

    return collection


def analyse_clusters(
    data: pd.DataFrame,
    cache_path: Path,
) -> ClusterInsightCollection:
    """Generate structured, evidence-backed cluster insights."""
    if cache_path.exists():
        print(f"Loading cached AI insights from {cache_path}")

        saved = json.loads(
            cache_path.read_text(encoding="utf-8")
        )

        return ClusterInsightCollection.model_validate(
            saved
        )

    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from the .env file."
        )

    clusters, allowed_evidence = prepare_cluster_evidence(
        data
    )

    prompt = f"""
You are a Voice of Customer product analyst.

Analyse the semantic feedback clusters supplied below.

For each cluster:
1. Create a concise topic name.
2. Summarize only what the evidence supports.
3. Assign severity: Low, Medium, High or Critical.
4. Return two or three exact customer comments as evidence.
5. Recommend a specific product-management action.
6. Provide confidence between 0 and 1.

Important rules:
- Do not invent customer facts.
- Do not mention golden labels.
- Evidence must be copied exactly from sample_comments.
- Financial loss, duplicate charging or blocked payment
  activity should be treated as highly serious.
- A recommendation is advisory; a human PM makes the
  final decision.

Clusters:
{json.dumps(clusters, indent=2)}
"""

    client = OpenAI(
        api_key=settings.openai_api_key
    )

    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return structured product intelligence "
                    "grounded only in supplied evidence."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format=ClusterInsightCollection,
    )

    parsed = completion.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError(
            "The model did not return a parsed response."
        )

    validated = validate_evidence(
        parsed,
        allowed_evidence,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        json.dumps(
            validated.model_dump(),
            indent=2,
        ),
        encoding="utf-8",
    )

    return validated


def create_insight_table(
    collection: ClusterInsightCollection,
) -> pd.DataFrame:
    records = []

    for insight in collection.insights:
        records.append(
            {
                "cluster_id": insight.cluster_id,
                "ai_topic_name": insight.topic_name,
                "ai_summary": insight.summary,
                "ai_severity": insight.severity,
                "ai_recommended_action": (
                    insight.recommended_action
                ),
                "ai_confidence": insight.confidence,
                "evidence_count": len(insight.evidence),
                "evidence": " | ".join(insight.evidence),
            }
        )

    return pd.DataFrame(records).sort_values(
        "cluster_id"
    )


def main() -> None:
    input_path = (
        settings.processed_data_dir
        / "feedback_with_clusters.csv"
    )

    cache_path = (
        settings.processed_data_dir
        / "cluster_ai_insights.json"
    )

    output_path = (
        settings.processed_data_dir
        / "cluster_ai_insights.csv"
    )

    feedback = pd.read_csv(input_path)

    collection = analyse_clusters(
        feedback,
        cache_path,
    )

    insight_table = create_insight_table(
        collection
    )

    insight_table.to_csv(
        output_path,
        index=False,
    )

    display_columns = [
        "cluster_id",
        "ai_topic_name",
        "ai_severity",
        "ai_confidence",
        "evidence_count",
    ]

    print("\\nAI-generated cluster intelligence")
    print("---------------------------------")

    print(
        insight_table[display_columns].to_string(
            index=False
        )
    )

    print(f"\\nAI insights saved to: {output_path}")


if __name__ == "__main__":
    main()
'''

TEST_CODE = '''
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
'''

interpreter_path = (
    ROOT / "src/analysis/cluster_interpreter.py"
)

test_path = (
    ROOT / "tests/test_cluster_interpreter.py"
)

interpreter_path.write_text(
    INTERPRETER_CODE.strip() + "\n",
    encoding="utf-8",
)

test_path.write_text(
    TEST_CODE.strip() + "\n",
    encoding="utf-8",
)

print("Created: src/analysis/cluster_interpreter.py")
print("Created: tests/test_cluster_interpreter.py")
print("Phase 6 AI interpretation layer created.")