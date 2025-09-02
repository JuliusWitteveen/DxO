# CORE LOGIC SPEC — DxO (Non-negotiables)

Goal: preservation of the sequential diagnostic cycle:
1) Track differential diagnosis and update after each step
2) Choose next action (question/test) based on information gain and cost
3) Transparency panels: rationale per agent visible
4) Two modes: (A) Streamlit UI (human = oracle), (B) autonomous benchmark-run

Multi-agent responsibilities (minimum):
- Hypothesis: maintains and weighs DDx (differential)
- Test-Chooser: selects informative question/test with attention to cost
- Challenger: prevents reasoning errors (biases)
- Stewardship: monitors cost/value
- Checklist/QC: format/validation of intermediate output
- Consensus: combines to one next action
- Gatekeeper/Judge: only in autonomous mode

Do not change (without explicit permission):
- Sequential loop: Observe → Decide → Act → Update → Repeat
- The role boundaries and their IO-contract (inputs: state/evidence; outputs: proposal + rationale)
- Existence of UI mode and autonomous mode
- Transparency: rationales remain traceable per agent

Allowed:
- Module layouts, file names, helpers, parsing/structure
- Bug fixes, type annotations, more robust parsers
- UI improvements that do not alter core decision logic

Acceptance criteria (must pass):
- `streamlit run app.py` starts and a case can iterate
- `python example_autonomous.py` completes a full case
- Agents provide rationale strings; consensus delivers 1 valid “next action”
- Tests in `tests/` run successfully or are realistically supplemented