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
