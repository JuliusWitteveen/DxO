# DxO — Diagnostic Orchestrator (Streamlit)

> Python prototype of a multi-agent diagnostic system with a Streamlit GUI.  
> Built for hands-on experimentation and quick iteration.

---

## Contents
- [What is DxO?](#what-is-dxo)
- [Quick start (Windows, cmd.exe)](#quick-start-windows-cmdexe)
- [Run modes](#run-modes)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Using the Streamlit GUI](#using-the-streamlit-gui)
- [Saving, loading, and exports](#saving-loading-and-exports)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What is DxO?
DxO is a **language-model–driven diagnostic orchestrator**. Multiple “physician” agents propose and critique next actions while maintaining an evolving **differential diagnosis**. You can:
- **Drive the case interactively** in a Streamlit UI (you act as the “oracle” that provides findings/results).
- **Run headless/autonomous simulations** from the terminal to benchmark behavior.

Typical agents include: hypothesis generator, test selector, challenger, stewardship, checklist, and a consensus step. The UI exposes transparency panels and a differential timeline, and you can save/load sessions or export Markdown.

---

## Quick start (Windows, cmd.exe)

> All commands below are for **Command Prompt** (`cmd.exe`), not PowerShell or bash.

```bat
:: 1) Clone
git clone https://github.com/JuliusWitteveen/DxO.git
cd DxO

:: 2) Create & activate venv
python -m venv .venv
.\.venv\Scripts\activate

:: 3) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4) Configure keys (creates .env)
echo OPENAI_API_KEY=sk-REPLACE_ME> .env
:: Optional providers/flags (append lines as needed):
:: echo GEMINI_API_KEY=REPLACE_ME>> .env
:: echo MAI_DX_SECRET=REPLACE_ME_ANY_RANDOM_STRING>> .env

:: 5) Launch the Streamlit UI
streamlit run app.py
```

---

## Run modes

### 1) Interactive UI (Streamlit)

```bat
.\.venv\Scripts\activate
streamlit run app.py
```

Open the local URL that Streamlit prints (usually `http://localhost:8501`).

### 2) Autonomous/headless run

```bat
.\.venv\Scripts\activate
python example_autonomous.py
```

---

## Repository layout

```
DxO/
├─ app.py                      # Streamlit entry point (interactive mode)
├─ example_autonomous.py       # Headless simulation runner
├─ mai_dx/                     # Core orchestrator, agents, parsers, persistence, UI helpers
├─ tests/                      # Unit tests (e.g., robust parsing of tool/function outputs)
├─ scripts/                    # Optional helper scripts
├─ requirements.txt
└─ README.md
```

> Tip: code paths and names above reflect the current prototype; adjust here if you rename files.

---

## Configuration

DxO reads configuration from environment variables (preferably via a `.env` file in the repo root).

| Variable         | Required | Purpose                                                                                                                                    |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `OPENAI_API_KEY` | Yes      | Enables OpenAI model calls.                                                                                                                |
| `GEMINI_API_KEY` | No       | Enables Google Gemini paths if present.                                                                                                    |
| `MAI_DX_SECRET`  | No       | If set, DxO **encrypts saved sessions** using Fernet (symmetric encryption). If absent, sessions are saved unencrypted for easy debugging. |

Example `.env` (Windows, append with `>>`):

```bat
echo OPENAI_API_KEY=sk-REPLACE_ME> .env
echo GEMINI_API_KEY=REPLACE_ME>> .env
echo MAI_DX_SECRET=REPLACE_ME_ANY_RANDOM_STRING>> .env
```

**Model selection.** The default models are wired in the code (see comments where the client is instantiated). You can change the model name(s) there if you want to try different providers/versions that support tool/function-calling.

---

## Using the Streamlit GUI

1. **Start a new session**.
2. The **agent panel** proposes the next action (ask for info or order a test).
3. **You** enter findings/results (free text is fine).
4. DxO updates the **differential** and shows **transparency panels** (agent rationales).
5. Iterate until you’re satisfied; optionally **save** the session or **export** Markdown.

What to type? Short, clinical, factual snippets work well:

* *“22F, 2 days fever 38.5°C, sore throat, anterior nodes, no cough.”*
* *“Rapid strep: positive.”*
* *“No rash. Not pregnant.”*

---

## Saving, loading, and exports

* **Save/Load sessions** from the UI. File names include a timestamp.
* If `MAI_DX_SECRET` is set, session payloads are **encrypted**; otherwise they’re plain JSON for easy inspection.
* **Export Markdown** to capture the case summary, differential timeline, and agent rationales.

---

## Testing

```bat
.\.venv\Scripts\activate
pytest -q
```

---

## Troubleshooting

* **Streamlit not starting**: verify the venv is active and `pip install -r requirements.txt` ran without errors.
* **API errors**: confirm your `.env` has a valid `OPENAI_API_KEY` (and `GEMINI_API_KEY` if using Gemini).
* **Session files unreadable**: you likely saved with `MAI_DX_SECRET` set; keep the same secret to load them.
* **Function/Tool output parsing issues**: run tests with `pytest -q` to confirm the parser behavior.

---

## License

MIT (see `LICENSE` if present). This is a research prototype intended for local experimentation.

---

