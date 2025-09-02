# CODex Findings for DxO

## Architecture Overview
DxO is a language‑model–driven diagnostic orchestrator composed of multiple "physician" agents that deliberate sequentially. It offers two run modes: a Streamlit interactive UI and a headless autonomous benchmark runner【F:README.md†L22-L27】. The core orchestration lives in `mai_dx/main.py`, where `MaiDxOrchestrator` manages agent initialization, parallel deliberation and consensus to decide the next diagnostic action【F:mai_dx/main.py†L239-L370】【F:mai_dx/main.py†L520-L638】.

### Repo Map
- `app.py` – Streamlit UI entry point.
- `example_autonomous.py` – CLI benchmark run.
- `config.py` – default prompts, model list and test costs.
- `llm_client_factory.py` – selects OpenAI Responses vs Chat Completions API.
- `mai_dx/` – core orchestrator, data structures, persistence and UI helpers.
- `scripts/install_dependencies.py` – optional dependency installer.
- `tests/` – unit tests for parsing and persistence.
- `swarms_utils/` – lightweight wrappers around external `swarms` utilities.

## Component Catalog
| Component | Purpose | Key API | Inputs → Outputs | Dependencies |
| --- | --- | --- | --- | --- |
| `LLMClient` (`llm_client_factory.py`) | Unified OpenAI client choosing Responses or Chat API | `generate(messages, system_prompt)` | chat messages → `(text, raw_response)` | `openai` SDK, `_use_responses_api` |
| `MaiDxOrchestrator` (`mai_dx/main.py`) | Coordinates multi‑agent diagnostic loop | `create_variant()`, `run()`, `_perform_turn()` | case state → action & deliberation | `swarms`, `pydantic`, prompts, costing |
| `InteractiveDxSession` (`mai_dx/interactive.py`) | Wraps orchestrator for Streamlit UI | `start()`, `step()`, `update_runtime_params()` | user inputs → session turns persisted | `MaiDxOrchestrator`, `persistence`, `costing` |
| `persistence` (`mai_dx/persistence.py`) | Encrypts/decrypts session files | `save_session()`, `load_session()`, `list_sessions()` | session dict ↔ encrypted JSON | `.env` (`MAI_DX_SECRET`), `cryptography` |
| `costing` (`mai_dx/costing.py`) | Lookup diagnostic test costs | `estimate_cost(tests, db)` | tests → cost int | `config.DEFAULT_TEST_COSTS` |
| UI module (`mai_dx/ui/*`) | Streamlit components for visualization & controls | e.g., `display_current_request()`, `render_settings_panel()` | session state → UI widgets | `streamlit`, `plotly` |

## Build/Run Path
1. **Interactive UI** (`streamlit run app.py`): `app.py` initializes the Streamlit session, collects settings, and instantiates `LLMClient`. User inputs spawn an `InteractiveDxSession`, which calls `_perform_turn` on `MaiDxOrchestrator` each time the physician responds.
2. **Autonomous CLI** (`python example_autonomous.py`): creates an orchestrator variant with `MaiDxOrchestrator.create_variant`, then executes `run` which loops through `_perform_turn`, interacts with a Gatekeeper for simulated patient responses, and finishes with Judge scoring.

### Main Control Loop
`MaiDxOrchestrator._perform_turn` builds shared context, runs panel agents in parallel (`ThreadPoolExecutor`), checks for stagnation, determines the next action via Consensus, and validates it【F:mai_dx/main.py†L590-L638】.

## Dataflow Overview
### Component Diagram
```mermaid
graph TD
  UI[Streamlit UI] -->|physician input| IS[InteractiveDxSession]
  IS -->|calls| ORCH[MaiDxOrchestrator]
  ORCH -->|runs| AGENTS[Panel Agents]
  ORCH -->|save/load| PERSIST[persistence]
  PERSIST -->|encrypted JSON| Sessions[(sessions/)]
  ORCH -->|LLM API| OpenAI[(OpenAI)]
  ORCH -->|Gatekeeper/Judge| LLMs[(External LLMs)]
```

### Sequence: Interactive Turn
```mermaid
sequenceDiagram
participant U as User
participant UI as Streamlit App
participant S as InteractiveDxSession
participant O as MaiDxOrchestrator
participant A as Agents
U->>UI: enter findings
UI->>S: step(physician_input)
S->>O: _perform_turn(case_state)
O->>A: parallel deliberation
A-->>O: analyses
O-->>S: Action + deliberation
S-->>UI: display next request
```

### Dataflow Summary
1. User provides findings → Streamlit UI → `InteractiveDxSession` updates `CaseState`.
2. `MaiDxOrchestrator` aggregates agent analyses → Consensus action.
3. Actions/tests update cost and differential; in autonomous mode the Gatekeeper supplies responses.
4. Session snapshots are encrypted and written to `sessions/` via `persistence.save_session`.

## Configuration & IO Map
- **Environment variables**: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MAI_DX_SECRET` for session encryption, `MAIDX_DEBUG` for verbose logs【F:README.md†L48-L55】【F:mai_dx/persistence.py†L46-L59】
- **Files**: `.env` for keys, `sessions/*.json` (encrypted), optional `logs/` for debug output.
- **Network**: OpenAI API via `openai` or `litellm`; external LLMs act as Gatekeeper/Judge during autonomous runs.
- **Flags/CLI**: mode selection (`no_budget`, `budgeted`, `question_only`) when creating orchestrator variants.

## Invariants & Contracts
- Diagnostic cycle must follow Observe → Decide → Act → Update (non‑negotiable per core spec).
- `Action` objects require `action_type` ∈ {`ask`,`test`,`diagnose`}, `content`, and `reasoning` enforced by `pydantic`【F:mai_dx/structures.py†L16-L34】.
- Differential entries must have probabilities between 0 and 1【F:mai_dx/main.py†L239-L244】【F:mai_dx/main.py†L656-L676】.
- Interactive session turn numbers must stay in sync with UI, otherwise `ValueError` is raised【F:mai_dx/interactive.py†L82-L88】.
- Sessions are encrypted with a persistent Fernet key; absence generates one and updates `.env`【F:mai_dx/persistence.py†L49-L59】.

## Risk Register
| Risk | Severity | Symptom/Quick Repro | Suggested Fix |
| --- | --- | --- | --- |
| **Incorrect enum name for Gatekeeper** (`AgentRole.GATEKeeper`) causes `AttributeError` when autonomous mode calls `_interact_with_gatekeeper` | High | Run `example_autonomous.py` → crash before first gatekeeper call | Fix enum reference to `AgentRole.GATEKEEPER`; add tests for gatekeeper path |
| **LLM call concurrency without timeouts** may hit rate limits or hang | Med | Simultaneous agent runs via `ThreadPoolExecutor` stall or fail under slow network | Add per‑request timeouts and retry/backoff logic |
| **Insecure fallback encryption** when `cryptography` missing stores sessions base64‑encoded | Med | Omit `cryptography` dependency and inspect saved file → readable | Require `cryptography` in production or warn user, disable saving without secure key |
| **API key persistence in plain text** (`app.py` writes to `.env`) | Med | Key stored unencrypted on disk | Use OS keyring or instruct users about security implications |
| **Limited test coverage of orchestrator logic** | Low | Modify `_perform_turn` and run tests → none fail | Add unit tests covering agent orchestration, budget modes, and gatekeeper/judge flows |

## Test & Tooling Baseline
- **Test framework**: `pytest` (`tests/` folder). Running `pytest -q` executes 8 tests focusing on parser robustness, persistence, and dependency shadowing; all pass【052f4b†L1-L2】.
- **Coverage gaps**: no tests for Streamlit UI, agent concurrency, cost budgeting, or autonomous run flows. Roughly covers <20% of modules.
- **Tooling**: no linting or type-check scripts detected; optional `scripts/install_dependencies.py` installs runtime packages.

---
Generated on 2025-09-02.
