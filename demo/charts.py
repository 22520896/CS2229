"""Plotly presentation only; all scores come from verified saved evidence."""

import plotly.graph_objects as go

from demo.evidence import COLORS


def theme(fig, y_title):
    fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(t=24, b=70, l=45, r=15),
        yaxis_title=y_title,
        legend=dict(orientation="h", y=-0.22),
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
    )
    return fig


def mechanism_chart(frame, mechanism, normalized):
    fig = go.Figure()
    for (model, source), block in frame.groupby(["model", "source"], sort=False):
        block = block.sort_values("x")
        fig.add_trace(
            go.Scatter(
                x=block.x,
                y=block.score,
                mode="lines+markers",
                name=f"{model} · {source}",
                line=dict(
                    color=COLORS.get(model, "#777777"),
                    dash="solid" if source == "Paper" else "dash",
                    width=2.5,
                ),
                marker=dict(
                    size=7, symbol="circle" if source == "Paper" else "diamond"
                ),
                hovertemplate="%{y:.4f}<extra>%{fullData.name}</extra>",
            )
        )
    title = {
        "rotation": "Phép xoay",
        "noise_features": "Feature nhiễu thêm (%)",
        "target_smoothing": "Hệ số covariance",
        "feature_removal": "Feature bị loại (%)",
    }[mechanism]
    fig.update_xaxes(title=title)
    if mechanism == "rotation":
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Không xoay", "Có xoay"])
    else:
        fig.update_xaxes(tickvals=sorted(frame.x.unique()))
    theme(fig, "Normalized test accuracy" if normalized else "Test accuracy")
    if not normalized:
        fig.update_yaxes(tickformat=".0%")
    return fig


def benchmark_chart(frame, show_range=True, log_x=False):
    fig = go.Figure()
    for model, block in frame.groupby("model", sort=False):
        block = block.sort_values("rank")
        color = COLORS.get(model, "#777777")
        defaults = block.loc[block["rank"].eq(0)]
        if log_x:
            block = block.loc[block["rank"].gt(0)]
        if block.empty:
            continue
        if log_x and not defaults.empty:
            fig.add_trace(
                go.Scatter(
                    x=[block["rank"].min(), block["rank"].max()],
                    y=[defaults["mean"].iloc[0]] * 2,
                    mode="lines",
                    name=model + " · default",
                    showlegend=False,
                    line=dict(color=color, dash="dot", width=1.2),
                )
            )
        if show_range:
            fig.add_trace(
                go.Scatter(
                    x=block["rank"],
                    y=block.high,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=block["rank"],
                    y=block.low,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.10),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=block["rank"],
                y=block["mean"],
                mode="lines",
                name=model,
                line=dict(color=color, width=2.5),
                hovertemplate="n=%{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_xaxes(
        title="Random search iterations"
        + (" · chấm = default" if log_x else " (0 = default)"),
        type="log" if log_x else "linear",
    )
    return theme(fig, "Normalized test score")


def confusion_chart(matrix):
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Dự đoán 0", "Dự đoán 1"],
            y=["Thật 0", "Thật 1"],
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_white", height=260, margin=dict(t=10, b=20, l=15, r=15)
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _rgba(color, alpha):
    return (
        "rgba("
        + ",".join(str(int(color[i : i + 2], 16)) for i in [1, 3, 5])
        + f",{alpha})"
    )
