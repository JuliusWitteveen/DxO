"""Visualization components for diagnostic flows and analytics."""

from typing import Dict, List, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class DiagnosticFlowVisualizer:
    """Generate visualizations of diagnostic flows."""

    def generate_sankey(self, case_state) -> go.Figure:
        """Create an interactive Sankey diagram of diagnostic flow."""
        nodes = ["Initial Vignette"]
        sources: List[int] = []
        targets: List[int] = []
        values: List[int] = []

        current_index = 0
        for entry in case_state.evidence_log:
            nodes.append(entry)
            sources.append(current_index)
            targets.append(len(nodes) - 1)
            values.append(1)
            current_index = len(nodes) - 1

        final_diag = getattr(case_state, "get_leading_diagnosis", lambda: "Diagnosis")()
        nodes.append(f"Diagnosis: {final_diag}")
        sources.append(current_index)
        targets.append(len(nodes) - 1)
        values.append(1)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(pad=15, thickness=20, label=nodes),
                    link=dict(source=sources, target=targets, value=values),
                )
            ]
        )
        fig.update_layout(title_text="Diagnostic Flow", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
        return fig


def _add_confidence_trace(fig: go.Figure, turns_data: List[Any]):
    """Adds the confidence evolution trace to the subplot."""
    turn_numbers = [getattr(t, "turn_number", i + 1) for i, t in enumerate(turns_data)]
    confidences = []
    for t in turns_data:
        if hasattr(t, "differential_at_turn") and t.differential_at_turn:
            confidences.append(max(t.differential_at_turn.values()))
        else:
            confidences.append(0)

    fig.add_trace(
        go.Scatter(
            x=turn_numbers,
            y=confidences,
            mode="lines+markers",
            name="Confidence",
            line=dict(width=3, color="blue"),
            marker=dict(size=8),
            hovertemplate="Turn %{x}<br>Confidence: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _add_cost_trace(fig: go.Figure, turns_data: List[Any]):
    """Adds the cost accumulation trace to the subplot."""
    turn_numbers = [getattr(t, "turn_number", i + 1) for i, t in enumerate(turns_data)]
    costs = [getattr(t, "cost_at_turn", 0) for t in turns_data]

    fig.add_trace(
        go.Scatter(
            x=turn_numbers,
            y=costs,
            mode="lines+markers",
            name="Cost",
            line=dict(width=3, color="red"),
            marker=dict(size=8),
            hovertemplate="Turn %{x}<br>Cost: $%{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )


def _add_differential_timeline_trace(fig: go.Figure, turns_data: List[Any]):
    """Adds the differential diagnosis timeline traces to the subplot."""
    all_diagnoses: Dict[str, List[tuple]] = {}
    for i, turn in enumerate(turns_data):
        if hasattr(turn, "differential_at_turn") and turn.differential_at_turn:
            turn_num = getattr(turn, "turn_number", i + 1)
            for diag, prob in turn.differential_at_turn.items():
                all_diagnoses.setdefault(diag, []).append((turn_num, prob))

    for diag, points in all_diagnoses.items():
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=diag[:30],
                hovertemplate=f"{diag}<br>Turn %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>",
            ),
            row=2,
            col=1,
        )


def _add_action_distribution_trace(fig: go.Figure, turns_data: List[Any]):
    """Adds the action distribution pie chart to the subplot."""
    action_counts = {"ask": 0, "test": 0, "diagnose": 0}
    for turn in turns_data:
        if hasattr(turn, "action_request") and hasattr(turn.action_request, "action_type"):
            action_type = turn.action_request.action_type
            if action_type in action_counts:
                action_counts[action_type] += 1

    if sum(action_counts.values()) > 0:
        fig.add_trace(
            go.Pie(
                labels=list(action_counts.keys()),
                values=list(action_counts.values()),
                marker=dict(colors=["#3498db", "#e74c3c", "#2ecc71"]),
            ),
            row=2,
            col=2,
        )


@st.cache_data
def create_diagnostic_journey_visualization(session):
    """Create visualizations summarizing the diagnostic journey."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Confidence Evolution",
            "Cost Accumulation",
            "Differential Diagnosis Timeline",
            "Action Distribution",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )

    if not session or not hasattr(session, "turns") or not session.turns:
        fig.add_annotation(
            text="No data available yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(height=700, showlegend=False)
        return fig

    turns = session.turns
    _add_confidence_trace(fig, turns)
    _add_cost_trace(fig, turns)
    _add_differential_timeline_trace(fig, turns)
    _add_action_distribution_trace(fig, turns)

    fig.update_xaxes(title_text="Turn", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Turn", row=1, col=2)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
    fig.update_xaxes(title_text="Turn", row=2, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)

    fig.update_layout(height=700, showlegend=True, title_text="Diagnostic Journey Overview")
    return fig


def create_confidence_timeline(session):
    """Plot confidence and cost trends across turns."""
    if not session or not hasattr(session, "turns") or not session.turns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(height=300, showlegend=False)
        return fig

    fig = go.Figure()
    for i, turn in enumerate(session.turns):
        colors = {"ask": "blue", "test": "orange", "diagnose": "green"}
        action_type = "unknown"
        action_content = "No action"
        if hasattr(turn, "action_request"):
            if hasattr(turn.action_request, "action_type"):
                action_type = turn.action_request.action_type
            if hasattr(turn.action_request, "content"):
                action_content = str(turn.action_request.content)[:100]

        leading_diag = "None"
        confidence = 0
        if hasattr(turn, "differential_at_turn") and turn.differential_at_turn:
            max_item = max(turn.differential_at_turn.items(), key=lambda x: x[1])
            leading_diag, confidence = max_item

        cost = getattr(turn, "cost_at_turn", 0)

        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=colors.get(action_type, "gray"),
                    symbol="circle",
                ),
                text=action_type.upper(),
                textposition="top center",
                name=f"Turn {i+1}",
                hovertemplate=(
                    f"<b>Turn {i+1}</b><br>"
                    f"Action: {action_type}<br>"
                    f"Request: {action_content}...<br>"
                    f"Cost: ${cost}<br>"
                    f"Leading Diagnosis: {leading_diag}<br>"
                    f"Confidence: {confidence:.1%}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Diagnostic Timeline",
        xaxis_title="Turn Number",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        height=300,
        hovermode="closest",
    )
    return fig