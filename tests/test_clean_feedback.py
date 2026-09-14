import pandas as pd
import pytest

from src.data.clean_feedback import (
    clean_feedback,
    validate_columns,
)


def valid_record() -> dict:
    return {
        "feedback_id": "FB-0001",
        "created_at": "2026-08-01",
        "source": "Survey",
        "customer_id": "CUST-1001",
        "customer_segment": "SMB",
        "product_area": "Payments",
        "feedback_text": " Payment failed   during checkout. ",
        "rating": 2,
        "app_version": "4.2.0",
        "region": "India",
        "monthly_revenue": 12000,
        "label": "Payment Failure",
        "sentiment_label": "Negative",
        "severity_label": "Critical",
    }


def test_removes_empty_feedback():
    first = valid_record()
    second = valid_record()
    second["feedback_id"] = "FB-0002"
    second["feedback_text"] = "   "

    cleaned, report = clean_feedback(
        pd.DataFrame([first, second])
    )

    assert len(cleaned) == 1
    assert report["empty_feedback_removed"] == 1


def test_normalizes_feedback_whitespace():
    cleaned, _ = clean_feedback(
        pd.DataFrame([valid_record()])
    )

    assert (
        cleaned.iloc[0]["feedback_text"]
        == "Payment failed during checkout."
    )


def test_removes_invalid_rating():
    record = valid_record()
    record["rating"] = 8

    cleaned, report = clean_feedback(
        pd.DataFrame([record])
    )

    assert cleaned.empty
    assert report["invalid_ratings_removed"] == 1


def test_removes_duplicate_feedback():
    record = valid_record()

    cleaned, report = clean_feedback(
        pd.DataFrame([record, record.copy()])
    )

    assert len(cleaned) == 1
    assert report["duplicates_removed"] == 1


def test_rejects_missing_required_columns():
    incomplete = pd.DataFrame(
        [{"feedback_id": "FB-0001"}]
    )

    with pytest.raises(ValueError):
        validate_columns(incomplete)
