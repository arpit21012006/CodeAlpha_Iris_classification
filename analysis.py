import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SPECIES_COLORS = {
    "Setosa": "#38BDF8",
    "Versicolor": "#A78BFA",
    "Virginica": "#F472B6"
}

FEATURE_LABELS = {
    "SepalLengthCm": "Sepal Length",
    "SepalWidthCm": "Sepal Width",
    "PetalLengthCm": "Petal Length",
    "PetalWidthCm": "Petal Width"
}

CHART_TEMPLATE = "plotly_dark"


def apply_chart_style(fig, height=420):
    """Apply consistent styling to dashboard charts."""

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color="#CBD5E1",
            size=12
        ),
        margin=dict(l=25, r=25, t=65, b=35),
        title_font=dict(
            size=17,
            color="#F8FAFC"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False
    )

    return fig


def species_distribution(df):
    """Display the proportion of Iris species."""

    counts = (
        df["Species"]
        .value_counts()
        .rename_axis("Species")
        .reset_index(name="Count")
    )

    fig = px.pie(
        counts,
        names="Species",
        values="Count",
        hole=0.64,
        color="Species",
        color_discrete_map=SPECIES_COLORS,
        title="Iris Species Distribution"
    )

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(
            line=dict(color="#111827", width=2)
        )
    )

    fig.add_annotation(
        text=f"<b>{len(df)}</b><br>Flowers",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, color="#F8FAFC")
    )

    return apply_chart_style(fig)


def measurement_scatter(df):
    """Compare petal dimensions across species."""

    fig = px.scatter(
        df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        color="Species",
        color_discrete_map=SPECIES_COLORS,
        hover_data=[
            "SepalLengthCm",
            "SepalWidthCm"
        ],
        labels={
            "PetalLengthCm": "Petal Length (cm)",
            "PetalWidthCm": "Petal Width (cm)"
        },
        title="Petal Measurements by Species",
        opacity=0.85
    )

    fig.update_traces(
        marker=dict(size=10, line=dict(width=0))
    )

    return apply_chart_style(fig)


def correlation_heatmap(df):
    """Visualize correlations between flower measurements."""

    numeric_df = df[
        list(FEATURE_LABELS.keys())
    ]

    correlations = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlations.values,
            x=list(FEATURE_LABELS.values()),
            y=list(FEATURE_LABELS.values()),
            colorscale=[
                [0.0, "#312E81"],
                [0.5, "#1E293B"],
                [1.0, "#22D3EE"]
            ],
            zmin=-1,
            zmax=1,
            text=correlations.round(2).values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=12),
            colorbar=dict(
                title="Correlation"
            ),
            hovertemplate=(
                "%{x} / %{y}<br>"
                "Correlation: %{z:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Feature Correlation Matrix"
    )

    return apply_chart_style(fig)


def feature_distribution(df, feature="SepalLengthCm"):
    """Display feature distributions for each species."""

    if feature not in FEATURE_LABELS:
        raise ValueError("Invalid Iris measurement selected.")

    fig = px.box(
        df,
        x="Species",
        y=feature,
        color="Species",
        color_discrete_map=SPECIES_COLORS,
        points="all",
        labels={
            feature: f"{FEATURE_LABELS[feature]} (cm)"
        },
        title=f"{FEATURE_LABELS[feature]} Distribution"
    )

    fig.update_traces(
        jitter=0.25,
        pointpos=0,
        marker=dict(size=5, opacity=0.55)
    )

    return apply_chart_style(fig)


def feature_summary(df):
    """Generate descriptive statistics for flower measurements."""

    summary = df[
        list(FEATURE_LABELS.keys())
    ].describe().T

    summary.index = [
        FEATURE_LABELS[column]
        for column in summary.index
    ]

    return summary.round(2)


def species_averages(df):
    """Calculate average measurements for each species."""

    averages = df.groupby("Species")[
        list(FEATURE_LABELS.keys())
    ].mean()

    return averages.rename(
        columns=FEATURE_LABELS
    ).round(2)