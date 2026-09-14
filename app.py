from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.product_intelligence import build_topic_insights
from src.config import settings


st.set_page_config(
    page_title="SignalDesk AI",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 14px;
            padding: 16px;
        }

        [data-testid="stMetricLabel"] {
            color: #9ca3af;
        }

        [data-testid="stMetricValue"] {
            color: #f9fafb;
        }

        .signal-header {
            padding: 20px 24px;
            border-radius: 16px;
            background:
                linear-gradient(120deg, #111827, #172554);
            border: 1px solid #263244;
            margin-bottom: 20px;
        }

        .signal-header h1 {
            margin: 0;
            color: white;
        }

        .signal-header p {
            margin: 6px 0 0 0;
            color: #bfdbfe;
        }

        .issue-card {
            border: 1px solid #334155;
            border-left: 5px solid #f97316;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 12px;
            background: #111827;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_feedback(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["created_at"] = pd.to_datetime(data["created_at"])
    return data


def filter_feedback(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    selected_segments = st.sidebar.multiselect(
        "Customer segment",
        sorted(data["customer_segment"].unique()),
        default=sorted(data["customer_segment"].unique()),
    )

    selected_sources = st.sidebar.multiselect(
        "Feedback source",
        sorted(data["source"].unique()),
        default=sorted(data["source"].unique()),
    )

    selected_regions = st.sidebar.multiselect(
        "Region",
        sorted(data["region"].unique()),
        default=sorted(data["region"].unique()),
    )

    start_date = data["created_at"].min().date()
    end_date = data["created_at"].max().date()

    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date,
    )

    filtered = data[
        data["customer_segment"].isin(selected_segments)
        & data["source"].isin(selected_sources)
        & data["region"].isin(selected_regions)
    ].copy()

    if len(selected_dates) == 2:
        selected_start, selected_end = selected_dates
        filtered = filtered[
            filtered["created_at"].dt.date.between(
                selected_start,
                selected_end,
            )
        ]

    return filtered


data_path = (
    settings.processed_data_dir
    / "flowpay_feedback_clean.csv"
)

if not data_path.exists():
    st.error(
        "Clean dataset not found. Run "
        "`python -m src.data.clean_feedback` first."
    )
    st.stop()

feedback = load_feedback(data_path)

st.markdown(
    """
    <div class="signal-header">
        <h1>📡 SignalDesk AI</h1>
        <p>
            Voice of Customer intelligence for evidence-backed
            product decisions
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

filtered = filter_feedback(feedback)

if filtered.empty:
    st.warning("No feedback matches the selected filters.")
    st.stop()

insights = build_topic_insights(filtered)

total_feedback = len(filtered)
unique_customers = filtered["customer_id"].nunique()
negative_rate = (
    filtered["sentiment_label"].eq("Negative").mean() * 100
)
emerging_count = int(insights["emerging_issue"].sum())

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Feedback analysed",
    f"{total_feedback:,}",
)
metric_columns[1].metric(
    "Unique customers",
    f"{unique_customers:,}",
)
metric_columns[2].metric(
    "Negative feedback",
    f"{negative_rate:.1f}%",
)
metric_columns[3].metric(
    "Emerging issues",
    emerging_count,
)

overview_tab, topic_tab, evidence_tab = st.tabs(
    [
        "Executive Overview",
        "Topic Explorer",
        "Customer Evidence",
    ]
)

with overview_tab:
    st.subheader("Product priorities")

    priority_table = insights[
        [
            "topic",
            "mentions",
            "unique_customers",
            "growth_percentage",
            "opportunity_score",
            "priority",
            "emerging_issue",
        ]
    ].copy()

    priority_table.columns = [
        "Topic",
        "Mentions",
        "Customers",
        "Growth %",
        "Opportunity Score",
        "Priority",
        "Emerging Issue",
    ]

    st.dataframe(
        priority_table,
        width="stretch",
        hide_index=True,
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        score_chart = px.bar(
            insights.sort_values("opportunity_score"),
            x="opportunity_score",
            y="topic",
            orientation="h",
            color="priority",
            title="Opportunity Score by topic",
            color_discrete_map={
                "Critical": "#ef4444",
                "High": "#f97316",
                "Medium": "#eab308",
                "Low": "#22c55e",
            },
        )

        score_chart.update_layout(
            xaxis_title="Opportunity Score",
            yaxis_title="",
            legend_title="Priority",
        )

        st.plotly_chart(
            score_chart,
            width="stretch",
        )

    with chart_right:
        topic_volume = (
            filtered.groupby("label")
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions")
        )

        volume_chart = px.bar(
            topic_volume,
            x="mentions",
            y="label",
            orientation="h",
            title="Feedback volume by topic",
            color="mentions",
            color_continuous_scale="Blues",
        )

        volume_chart.update_layout(
            xaxis_title="Feedback records",
            yaxis_title="",
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            volume_chart,
            width="stretch",
        )

    weekly = (
        filtered.assign(
            week=filtered["created_at"].dt.to_period("W").dt.start_time
        )
        .groupby(["week", "label"])
        .size()
        .reset_index(name="mentions")
    )

    trend_chart = px.line(
        weekly,
        x="week",
        y="mentions",
        color="label",
        markers=True,
        title="Weekly feedback trends",
    )

    trend_chart.update_layout(
        xaxis_title="Week",
        yaxis_title="Mentions",
        legend_title="Topic",
    )

    st.plotly_chart(
        trend_chart,
        width="stretch",
    )

with topic_tab:
    st.subheader("Investigate a product topic")

    selected_topic = st.selectbox(
        "Select topic",
        insights["topic"].tolist(),
    )

    selected_insight = insights[
        insights["topic"] == selected_topic
    ].iloc[0]

    topic_metrics = st.columns(5)

    topic_metrics[0].metric(
        "Mentions",
        int(selected_insight["mentions"]),
    )
    topic_metrics[1].metric(
        "Customers",
        int(selected_insight["unique_customers"]),
    )
    topic_metrics[2].metric(
        "Growth",
        f'{selected_insight["growth_percentage"]:.1f}%',
    )
    topic_metrics[3].metric(
        "Severity",
        f'{selected_insight["severity_score"]:.0f}/100',
    )
    topic_metrics[4].metric(
        "Opportunity",
        f'{selected_insight["opportunity_score"]:.1f}/100',
    )

    if selected_insight["emerging_issue"]:
        st.error(
            "🚨 Emerging issue detected: mentions increased "
            "substantially in the latest analysis period."
        )

    score_components = pd.DataFrame(
        {
            "Factor": [
                "Frequency",
                "Severity",
                "Trend",
                "Customer value",
                "Recency",
            ],
            "Score": [
                selected_insight["frequency_score"],
                selected_insight["severity_score"],
                selected_insight["trend_score"],
                selected_insight["customer_value_score"],
                selected_insight["recency_score"],
            ],
        }
    )

    component_chart = px.bar(
        score_components,
        x="Factor",
        y="Score",
        color="Score",
        range_y=[0, 100],
        title="Opportunity Score components",
        color_continuous_scale="Oranges",
    )

    component_chart.update_layout(
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        component_chart,
        width="stretch",
    )

with evidence_tab:
    st.subheader("Customer evidence")

    evidence_topic = st.selectbox(
        "Evidence topic",
        sorted(filtered["label"].unique()),
        key="evidence_topic",
    )

    evidence = filtered[
        filtered["label"] == evidence_topic
    ].sort_values(
        "created_at",
        ascending=False,
    )

    sentiment_filter = st.multiselect(
        "Sentiment",
        sorted(evidence["sentiment_label"].unique()),
        default=sorted(evidence["sentiment_label"].unique()),
    )

    evidence = evidence[
        evidence["sentiment_label"].isin(sentiment_filter)
    ]

    st.caption(
        f"Showing {len(evidence)} supporting feedback records."
    )

    st.dataframe(
        evidence[
            [
                "created_at",
                "feedback_text",
                "customer_segment",
                "source",
                "region",
                "sentiment_label",
                "severity_label",
                "app_version",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

st.caption(
    "SignalDesk uses deterministic calculations for metrics. "
    "AI explanations will be added in the next phase."
)
