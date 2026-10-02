from pathlib import Path
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocessing import FEATURE_COLUMNS
from model import train_models, predict_species
from analysis import (
    SPECIES_COLORS,
    FEATURE_LABELS,
    species_distribution,
    measurement_scatter,
    correlation_heatmap,
    feature_distribution,
    feature_summary,
    species_averages,
)


st.set_page_config(
    page_title="IrisVision | Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


def load_styles():
    css_path = BASE_DIR / "style.css"
    with open(css_path, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner="Training classification models...")
def load_models():
    return train_models()


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-caption">{escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle):
    st.markdown(
        f"""
        <div class="section-title">{escape(title)}</div>
        <div class="section-subtitle">{escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def probability_chart(probabilities):
    names = list(probabilities.keys())
    values = list(probabilities.values())

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[
                SPECIES_COLORS.get(name, "#A78BFA")
                for name in names
            ],
            text=[f"{value:.1f}%" for value in values],
            textposition="auto",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Classification Probabilities",
        template="plotly_dark",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        xaxis=dict(range=[0, 105], title="Probability (%)"),
        yaxis=dict(title=""),
        margin=dict(l=20, r=20, t=55, b=30),
        showlegend=False,
    )

    return fig


def confusion_matrix_chart(matrix, class_names):
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            colorscale="Purples",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Count: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        height=410,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        xaxis_title="Predicted Species",
        yaxis_title="Actual Species",
        margin=dict(l=20, r=20, t=60, b=35),
    )

    return fig


def prediction_page(data, results):
    section_heading(
        "Flower Species Prediction",
        "Enter four flower measurements to identify the Iris species.",
    )

    st.info(
        "Measurements are in centimeters. "
        "Try different values to explore how predictions change."
    )

    model_name = st.selectbox(
        "Classification Model",
        options=list(results.keys()),
    )

    with st.form("iris_prediction_form"):
        first_row = st.columns(2)
        second_row = st.columns(2)

        with first_row[0]:
            sepal_length = st.number_input(
                "Sepal Length (cm)",
                min_value=0.1,
                max_value=12.0,
                value=5.1,
                step=0.1,
            )

        with first_row[1]:
            sepal_width = st.number_input(
                "Sepal Width (cm)",
                min_value=0.1,
                max_value=8.0,
                value=3.5,
                step=0.1,
            )

        with second_row[0]:
            petal_length = st.number_input(
                "Petal Length (cm)",
                min_value=0.1,
                max_value=10.0,
                value=1.4,
                step=0.1,
            )

        with second_row[1]:
            petal_width = st.number_input(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=5.0,
                value=0.2,
                step=0.1,
            )

        submitted = st.form_submit_button(
            "🌸 Classify Flower",
            use_container_width=True,
        )

    if submitted:
        selected_model = results[model_name]["model"]

        scaler = (
            data["scaler"]
            if model_name == "Logistic Regression"
            else None
        )

        measurements = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]

        prediction = predict_species(
            selected_model,
            measurements,
            data["encoder"],
            scaler,
        )

        species = prediction["species"]
        confidence = prediction["confidence"]

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Iris Species</div>
                <div class="prediction-species">
                    {escape(species)}
                </div>
                <div class="prediction-confidence">
                    Model confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            probability_chart(prediction["probabilities"]),
            use_container_width=True,
        )

        st.caption(
            "Confidence represents the model's estimated class "
            "probability, not a guarantee of correctness."
        )

    st.markdown(
        """
        <div class="info-card">
            <h3>How classification works</h3>
            <p>
                IrisVision uses labeled flower measurements to learn
                patterns associated with Setosa, Versicolor and Virginica.
                The selected model then predicts the species of a new
                flower from its sepal and petal measurements.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analytics_page(df):
    section_heading(
        "Flower Data Analytics",
        "Explore species patterns and relationships between measurements.",
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            species_distribution(df),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            measurement_scatter(df),
            use_container_width=True,
        )

    st.plotly_chart(
        correlation_heatmap(df),
        use_container_width=True,
    )

    feature = st.selectbox(
        "Select a measurement to explore",
        options=FEATURE_COLUMNS,
        format_func=lambda value: FEATURE_LABELS[value],
    )

    st.plotly_chart(
        feature_distribution(df, feature),
        use_container_width=True,
    )

    st.subheader("Average Measurements by Species")
    st.dataframe(
        species_averages(df),
        use_container_width=True,
    )


def performance_page(data, results):
    section_heading(
        "Machine Learning Performance",
        "Evaluate classification quality on the held-out test dataset.",
    )

    selected = st.selectbox(
        "Select Model for Evaluation",
        options=list(results.keys()),
        key="performance_model",
    )

    metrics = results[selected]

    columns = st.columns(4)

    values = [
        ("Accuracy", f"{metrics['accuracy'] * 100:.1f}%"),
        ("Precision", f"{metrics['precision'] * 100:.1f}%"),
        ("Recall", f"{metrics['recall'] * 100:.1f}%"),
        ("F1 Score", f"{metrics['f1'] * 100:.1f}%"),
    ]

    for column, (label, value) in zip(columns, values):
        with column:
            metric_card(label, value, "Test dataset")

    st.plotly_chart(
        confusion_matrix_chart(
            metrics["confusion_matrix"],
            data["class_names"],
        ),
        use_container_width=True,
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": round(result["accuracy"], 3),
                "Precision": round(result["precision"], 3),
                "Recall": round(result["recall"], 3),
                "F1 Score": round(result["f1"], 3),
            }
            for name, result in results.items()
        ]
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Models use the same stratified 80/20 train-test split. "
        "Metrics describe performance on the held-out test samples."
    )


def dataset_page(df):
    section_heading(
        "Iris Dataset Explorer",
        "Inspect the original flower measurements and descriptive statistics.",
    )

    species_options = ["All Species"] + sorted(
        df["Species"].unique().tolist()
    )

    selected_species = st.selectbox(
        "Filter by Species",
        species_options,
    )

    filtered = df.copy()

    if selected_species != "All Species":
        filtered = filtered[
            filtered["Species"] == selected_species
        ]

    st.dataframe(
        filtered,
        hide_index=True,
        use_container_width=True,
        height=380,
    )

    st.download_button(
        "Download Filtered Dataset",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="iris_filtered.csv",
        mime="text/csv",
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        feature_summary(filtered),
        use_container_width=True,
    )


def main():
    load_styles()

    data, results = load_models()
    df = data["dataframe"]

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">🌸 IrisVision</div>
            <div class="sidebar-tagline">
                Flower Intelligence Platform
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.subheader("Project Overview")
        st.write("**Project:** Iris Flower Classification")
        st.write("**Domain:** Data Science")
        st.write("**Models:** Random Forest & Logistic Regression")
        st.write("**Dataset:** Iris.csv")

        st.divider()

        st.subheader("Dataset Summary")
        st.metric("Flower Samples", len(df))
        st.metric("Flower Species", df["Species"].nunique())
        st.metric("Input Features", len(FEATURE_COLUMNS))

        st.divider()
        st.caption("CodeAlpha Data Science Internship")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">
                Machine Learning • Botanical Analytics
            </div>
            <h1>🌸 Iris Flower Intelligence Dashboard</h1>
            <p>
                Discover flower species using machine learning,
                explore botanical measurements and evaluate
                classification performance through interactive analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview = st.columns(4)

    best_accuracy = max(
        result["accuracy"] for result in results.values()
    )

    overview_values = [
        ("Total Flowers", len(df), "Dataset records"),
        ("Species", df["Species"].nunique(), "Iris categories"),
        ("Measurements", len(FEATURE_COLUMNS), "Input features"),
        ("Highest Test Accuracy", f"{best_accuracy * 100:.1f}%",
         "Among trained models"),
    ]

    for column, (label, value, caption) in zip(
        overview, overview_values
    ):
        with column:
            metric_card(label, value, caption)

    tabs = st.tabs(
        [
            "🌸 Flower Prediction",
            "📊 Data Analytics",
            "🧠 Model Performance",
            "📁 Dataset Explorer",
        ]
    )

    with tabs[0]:
        prediction_page(data, results)

    with tabs[1]:
        analytics_page(df)

    with tabs[2]:
        performance_page(data, results)

    with tabs[3]:
        dataset_page(df)

    st.markdown(
        """
        <div class="dashboard-footer">
            IrisVision • Iris Flower Classification<br>
            Built with Python, Scikit-learn, Plotly and Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()