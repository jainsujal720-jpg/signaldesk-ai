from pathlib import Path

FILES = {
    "requirements.txt": """
pandas==2.3.2
numpy==2.3.2
scikit-learn==1.7.1
openai==1.106.1
python-dotenv==1.1.1
streamlit==1.49.1
plotly==6.3.0
fastapi==0.116.1
uvicorn==0.35.0
pytest==8.4.2
""",
    ".env.example": """
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
""",
    "src/__init__.py": "",
    "src/data/__init__.py": "",
    "src/analysis/__init__.py": "",
    "src/dashboard/__init__.py": "",
    "tests/__init__.py": "",
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "src/config.py": '''
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )


settings = Settings()
''',
    "src/data/generate_feedback.py": '''
import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

from src.config import settings

TOPICS = {
    "Payment Failure": {
        "product_area": "Payments",
        "sentiment": "Negative",
        "severity": "Critical",
        "templates": [
            "My payment failed after I entered the OTP.",
            "The amount was deducted but the order was not confirmed.",
            "Checkout keeps loading and the payment never completes.",
            "The app crashed while I was paying by card.",
            "I was charged twice for the same transaction.",
        ],
    },
    "Login Problem": {
        "product_area": "Authentication",
        "sentiment": "Negative",
        "severity": "High",
        "templates": [
            "I cannot log in even though my password is correct.",
            "The OTP for login never arrives.",
            "The app keeps signing me out.",
            "Biometric login stopped working after the update.",
        ],
    },
    "Refund Delay": {
        "product_area": "Refunds",
        "sentiment": "Negative",
        "severity": "High",
        "templates": [
            "My refund has not arrived after ten days.",
            "The refund status has been pending for a week.",
            "I cannot see when my refund will be processed.",
            "Support confirmed a refund but I have not received it.",
        ],
    },
    "Invoice Export": {
        "product_area": "Reporting",
        "sentiment": "Neutral",
        "severity": "Medium",
        "templates": [
            "Please add an option to export invoices to Excel.",
            "We need bulk invoice downloads for accounting.",
            "Can you support PDF and CSV invoice exports?",
            "Invoice export would save our finance team several hours.",
        ],
    },
    "Slow Application": {
        "product_area": "Performance",
        "sentiment": "Negative",
        "severity": "Medium",
        "templates": [
            "The dashboard takes too long to load.",
            "The mobile application has become very slow.",
            "Transaction history freezes when I scroll.",
            "The latest version feels slower than the previous one.",
        ],
    },
    "Dashboard Usability": {
        "product_area": "Dashboard",
        "sentiment": "Neutral",
        "severity": "Low",
        "templates": [
            "The dashboard navigation is confusing.",
            "Important payment information is difficult to find.",
            "Please allow us to customize dashboard widgets.",
            "The analytics screen needs clearer filters.",
        ],
    },
    "Integration Request": {
        "product_area": "Integrations",
        "sentiment": "Positive",
        "severity": "Medium",
        "templates": [
            "Please add an integration with our accounting software.",
            "A Shopify integration would simplify our workflow.",
            "We would like API support for transaction reconciliation.",
            "Please connect FlowPay with more CRM platforms.",
        ],
    },
    "Positive Experience": {
        "product_area": "General",
        "sentiment": "Positive",
        "severity": "Low",
        "templates": [
            "FlowPay makes payment tracking much easier.",
            "The new dashboard is clean and helpful.",
            "Customer support resolved my problem quickly.",
            "The application is easy to use and reliable.",
        ],
    },
}

SOURCES = [
    "Support Ticket",
    "App Store Review",
    "Survey",
    "Feature Request",
    "Sales Note",
]

SEGMENTS = ["Free", "SMB", "Mid-Market", "Enterprise"]
REGIONS = ["India", "United Kingdom", "United States", "Singapore", "UAE"]

SEGMENT_REVENUE = {
    "Free": (0, 0),
    "SMB": (2000, 15000),
    "Mid-Market": (15001, 75000),
    "Enterprise": (75001, 300000),
}


def choose_topic(feedback_date: date, rng: random.Random) -> str:
    """Create a deliberate payment-issue spike after version 4.2."""
    release_date = date(2026, 7, 15)

    if feedback_date >= release_date:
        names = list(TOPICS)
        weights = [35, 10, 12, 12, 10, 7, 8, 6]
    else:
        names = list(TOPICS)
        weights = [12, 13, 13, 14, 13, 11, 12, 12]

    return rng.choices(names, weights=weights, k=1)[0]


def create_feedback_record(
    index: int,
    feedback_date: date,
    rng: random.Random,
) -> dict:
    topic_name = choose_topic(feedback_date, rng)
    topic = TOPICS[topic_name]
    segment = rng.choices(
        SEGMENTS,
        weights=[15, 40, 28, 17],
        k=1,
    )[0]

    minimum, maximum = SEGMENT_REVENUE[segment]
    monthly_revenue = (
        0 if maximum == 0 else rng.randint(minimum, maximum)
    )

    rating_by_sentiment = {
        "Negative": [1, 1, 2, 2, 3],
        "Neutral": [3, 3, 4],
        "Positive": [4, 4, 5, 5],
    }

    app_version = (
        "4.2.0"
        if feedback_date >= date(2026, 7, 15)
        else rng.choice(["4.0.0", "4.1.0"])
    )

    return {
        "feedback_id": f"FB-{index:04d}",
        "created_at": feedback_date.isoformat(),
        "source": rng.choice(SOURCES),
        "customer_id": f"CUST-{rng.randint(1000, 1350)}",
        "customer_segment": segment,
        "product_area": topic["product_area"],
        "feedback_text": rng.choice(topic["templates"]),
        "rating": rng.choice(
            rating_by_sentiment[topic["sentiment"]]
        ),
        "app_version": app_version,
        "region": rng.choice(REGIONS),
        "monthly_revenue": monthly_revenue,
        "label": topic_name,
        "sentiment_label": topic["sentiment"],
        "severity_label": topic["severity"],
    }


def generate_feedback(
    record_count: int = 750,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    start_date = date(2026, 5, 1)
    end_date = date(2026, 8, 31)
    date_range = (end_date - start_date).days

    records = []

    for index in range(1, record_count + 1):
        random_date = start_date + timedelta(
            days=rng.randint(0, date_range)
        )
        records.append(
            create_feedback_record(index, random_date, rng)
        )

    return sorted(records, key=lambda item: item["created_at"])


def save_feedback(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = settings.raw_data_dir / "flowpay_feedback.csv"
    records = generate_feedback(args.records, args.seed)
    save_feedback(records, output_path)

    print(f"Created {len(records)} records")
    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
''',
    "tests/test_generate_feedback.py": '''
from src.data.generate_feedback import generate_feedback


REQUIRED_FIELDS = {
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


def test_generates_requested_number_of_records():
    records = generate_feedback(record_count=100)
    assert len(records) == 100


def test_records_contain_required_fields():
    record = generate_feedback(record_count=1)[0]
    assert set(record) == REQUIRED_FIELDS


def test_generation_is_reproducible():
    first = generate_feedback(record_count=10, seed=42)
    second = generate_feedback(record_count=10, seed=42)
    assert first == second


def test_ratings_are_valid():
    records = generate_feedback(record_count=100)
    assert all(1 <= record["rating"] <= 5 for record in records)
''',
    "README.md": """
# SignalDesk AI

SignalDesk AI is a Voice of Customer intelligence platform that
converts unstructured feedback into evidence-backed product insights.

## Product problem

Product teams receive feedback through support tickets, surveys,
app reviews, feature requests and customer conversations. Important
patterns are difficult to identify manually and may remain unnoticed.

SignalDesk helps product teams:

- group semantically similar feedback;
- detect emerging customer problems;
- analyse sentiment and severity;
- calculate transparent opportunity scores;
- retrieve evidence behind every insight;
- ask questions across customer feedback;
- evaluate AI output against labelled data.

## Product principle

> Code calculates facts. AI interprets facts. The product manager
> makes the final decision.

## Current phase

Phase 1 creates a realistic synthetic dataset for the fictional
payments platform FlowPay. The dataset includes an intentional increase
in payment complaints following the release of version 4.2.0.

## Generate the dataset

```bash
python -m src.data.generate_feedback```

The generated file will be available at:

```text
data/raw/flowpay_feedback.csv
```

## Run tests

```bash
pytest
```

## Planned architecture

1. Data ingestion
2. Cleaning and validation
3. Embedding generation
4. Semantic clustering
5. Topic, sentiment and severity analysis
6. Trend and Opportunity Score calculation
7. Evidence-backed AI recommendations
8. Product intelligence dashboard
9. Ask Your Customers interface
10. Evaluation and human review
""",
}


def create_project() -> None:
    root = Path(__file__).resolve().parent

    for relative_path, content in FILES.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Skipped existing file: {relative_path}")
            continue

        destination.write_text(
            content.strip() + "\n" if content else "",
            encoding="utf-8",
        )
        print(f"Created: {relative_path}")

    print("\nSignalDesk Phase 1 foundation created successfully.")


if __name__ == "__main__":
    create_project()