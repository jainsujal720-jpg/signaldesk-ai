from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLEANING_CODE = '''
import argparse
from pathlib import Path

import pandas as pd

from src.config import settings


REQUIRED_COLUMNS = {
    "feedback_id",
    "created_at",
    "source",
    "customer_id",
    "customer_segment",
    "product_area",
    "feedback_text",
    "rating",
    "app_version",
    "region",
    "monthly_revenue",
    "label",
    "sentiment_label",
    "severity_label",
}


def validate_columns(data: pd.DataFrame) -> None:
    """Raise an error when required columns are missing."""
    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")


def clean_feedback(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean feedback records and return a quality report."""
    validate_columns(data)

    cleaned = data.copy()
    original_count = len(cleaned)

    cleaned.columns = [
        column.strip().lower() for column in cleaned.columns
    ]

    cleaned["feedback_text"] = (
        cleaned["feedback_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\\s+", " ", regex=True)
    )

    empty_feedback_count = (
        cleaned["feedback_text"].eq("").sum()
    )

    cleaned = cleaned[
        cleaned["feedback_text"].ne("")
    ].copy()

    cleaned["created_at"] = pd.to_datetime(
        cleaned["created_at"],
        errors="coerce",
    )

    invalid_date_count = cleaned["created_at"].isna().sum()
    cleaned = cleaned.dropna(subset=["created_at"])

    cleaned["rating"] = pd.to_numeric(
        cleaned["rating"],
        errors="coerce",
    )

    invalid_rating_count = (
        ~cleaned["rating"].between(1, 5)
    ).sum()

    cleaned = cleaned[
        cleaned["rating"].between(1, 5)
    ].copy()

    cleaned["monthly_revenue"] = pd.to_numeric(
        cleaned["monthly_revenue"],
        errors="coerce",
    ).fillna(0)

    cleaned["monthly_revenue"] = (
        cleaned["monthly_revenue"].clip(lower=0)
    )

    text_fingerprint = (
        cleaned["customer_id"].astype(str).str.lower()
        + "|"
        + cleaned["feedback_text"].str.lower()
        + "|"
        + cleaned["created_at"].astype(str)
    )

    duplicate_count = text_fingerprint.duplicated().sum()
    cleaned = cleaned[
        ~text_fingerprint.duplicated()
    ].copy()

    cleaned = cleaned.sort_values(
        ["created_at", "feedback_id"]
    ).reset_index(drop=True)

    report = {
        "original_records": original_count,
        "clean_records": len(cleaned),
        "removed_records": original_count - len(cleaned),
        "empty_feedback_removed": int(empty_feedback_count),
        "invalid_dates_removed": int(invalid_date_count),
        "invalid_ratings_removed": int(invalid_rating_count),
        "duplicates_removed": int(duplicate_count),
    }

    return cleaned, report


def process_feedback(
    input_path: Path,
    output_path: Path,
) -> dict:
    data = pd.read_csv(input_path)
    cleaned, report = clean_feedback(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.raw_data_dir / "flowpay_feedback.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            settings.processed_data_dir
            / "flowpay_feedback_clean.csv"
        ),
    )
    args = parser.parse_args()

    report = process_feedback(args.input, args.output)

    print("Data quality report")
    print("-------------------")

    for name, value in report.items():
        readable_name = name.replace("_", " ").title()
        print(f"{readable_name}: {value}")

    print(f"\\nClean dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
'''

TEST_CODE = '''
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
'''


def write_file(relative_path: str, content: str) -> None:
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        content.strip() + "\n",
        encoding="utf-8",
    )
    print(f"Created: {relative_path}")


write_file(
    "src/data/clean_feedback.py",
    CLEANING_CODE,
)

write_file(
    "tests/test_clean_feedback.py",
    TEST_CODE,
)

print("Phase 2 cleaning pipeline created successfully.")