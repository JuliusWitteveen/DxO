from datetime import datetime
from typing import Dict, Any


def export_session_to_markdown(session_data: Dict[str, Any]) -> str:
    """Generate a Markdown report summarizing a diagnostic session.

    Args:
        session_data: Serialized session data as produced by
            :func:`InteractiveDxSession.to_dict`.

    Returns:
        A markdown-formatted string representing the session report.

    Raises:
        ValueError: If required session information is missing.
    """

    if not session_data:
        raise ValueError("No session data provided for export.")

    case_state = session_data.get("case_state") or {}
    turns = session_data.get("turns") or []
    if not turns:
        raise ValueError("Session data does not contain any turns to export.")

    report = f"# MAI-DxO Diagnostic Report\n\n"
    report += f"**Session ID:** `{session_data.get('session_id', 'N/A')}`\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    report += "## Case Summary\n"
    report += f"**Initial Presentation:**\n> {case_state.get('initial_vignette', 'N/A')}\n\n"

    last_action = turns[-1].get("action_request", {}) if turns else {}
    if last_action.get("action_type") == "diagnose":
        report += "## Final Diagnosis\n"
        report += f"**Diagnosis:** **{last_action.get('content', 'N/A')}**\n"
        report += f"**Reasoning:** {last_action.get('reasoning', 'N/A')}\n\n"

    report += "## Diagnostic Timeline\n"
    for turn in turns:
        turn_number = turn.get("turn_number", "?")
        action = turn.get("action_request", {})
        report += f"### Turn {turn_number}\n"
        report += (
            f"- **AI Action:** `{action.get('action_type', 'N/A').upper()}`\n"
        )
        content = action.get("content")
        if isinstance(content, list):
            report += "- **AI Request:**\n"
            for item in content:
                report += f"  - {item}\n"
        else:
            report += f"- **AI Request:** {content if content is not None else 'N/A'}\n"
        report += f"- **AI Reasoning:** *{action.get('reasoning', 'N/A')}*\n"
        physician_input = turn.get("physician_input")
        if physician_input:
            report += f"- **Physician Input:**\n```\n{physician_input}\n```\n"
        report += "\n"

    report += "## Final State\n"
    report += f"- **Total Cost:** ${case_state.get('cumulative_cost', 0):,}\n"
    report += f"- **Total Turns:** {len(turns)}\n\n"

    report += "## Differential Diagnosis Evolution\n"
    report += "| Turn | Diagnosis | Probability |\n"
    report += "|------|-----------|-------------|\n"

    # This part requires processing the timeline data
    timeline = []
    for turn in turns:
        turn_number = turn.get("turn_number", 0)
        for diagnosis, prob in turn.get("differential_at_turn", {}).items():
            timeline.append(
                {
                    "turn": turn_number,
                    "diagnosis": diagnosis,
                    "probability": prob,
                }
            )

    for item in timeline:
        report += f"| {item['turn']} | {item['diagnosis']} | {item['probability']:.0%} |\n"
    return report
