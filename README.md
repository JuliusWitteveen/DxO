# MAI Diagnostic Orchestrator (MAI-DxO)

[![Paper](https://img.shields.io/badge/Paper-arXiv:2306.022405-red.svg)](https://arxiv.org/abs/2306.022405)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)

MAI-DxO is an interactive, AI-powered diagnostic tool that implements the concepts from Microsoft Research's "Sequential Diagnosis with Language Models" paper. It simulates a virtual panel of physician-agents to perform iterative medical diagnosis, now with a human-in-the-loop interface for real-time clinical evaluation.

This project provides two modes of operation:
1.  **Autonomous Mode**: The original simulation that runs a full diagnostic process based on a complete, hidden case file.
2.  **Interactive Mode**: A new turn-based Streamlit application where a human physician provides clinical findings in response to the AI's requests.

![MAI-DxO Interactive UI Screenshot](./mai_dxo_screenshot.png)

## ✨ Key Features

- **Interactive Turn-Based UI**: A Streamlit application that allows physicians to guide the diagnostic process.
- **8 AI Physician Agents**: Specialized roles for comprehensive, multi-faceted diagnosis.
- **Session Persistence**: Save, load, and resume diagnostic sessions.
- **Transparency Panels**: View the detailed reasoning of each AI agent at every turn.
- **Differential Diagnosis Timeline**: Visualize how the diagnostic probabilities evolve over time.
- **Markdown Export**: Generate a complete, shareable report of any session.
- **Autonomous Benchmark Mode**: Run the original simulation for research and evaluation.

## 🚀 Quick Start: Interactive Mode

### 1. Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator.git
cd Open-MAI-Dx-Orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or
python scripts/install_dependencies.py
```

### 2. Environment Variables

Create a `.env` file in the project root and add your API keys. The application will prompt for keys if they are not found.

```sh
# .env file
OPENAI_API_KEY="sk-..."
# GEMINI_API_KEY="..."
```

### 3. Run the App

```bash
streamlit run app.py
```

Your browser will open with the MAI-DxO interactive interface.

## 🔬 Autonomous Mode

To run the original autonomous benchmark simulation:

```bash
python3 example_autonomous.py
```

This will run a full diagnostic session in your terminal based on the pre-defined case in the script.

## ⚙️ How It Works: The Virtual Physician Panel

MAI-DxO employs a multi-agent system where each agent has a specific role:

- **🧠 Dr. Hypothesis**: Maintains the differential diagnosis.
- **🔬 Dr. Test-Chooser**: Selects the most informative diagnostic tests.
- **🤔 Dr. Challenger**: Prevents cognitive biases and diagnostic errors.
- **💰 Dr. Stewardship**: Ensures cost-effective care.
- **✅ Dr. Checklist**: Performs quality control checks.
- **🤝 Consensus Coordinator**: Synthesizes panel decisions into a single action.
- **🔑 Gatekeeper**: (Autonomous Mode) Acts as the clinical information oracle.
- **⚖️ Judge**: (Autonomous Mode) Evaluates the final diagnostic accuracy.

## 🤝 Contributing

We welcome contributions! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

If you use this work in your research, please cite both the original paper and this software implementation.
```bibtex
@misc{nori2023sequential,
      title={Sequential Diagnosis with Large Language Models}, 
      author={Harsha Nori and Mayank Daswani and Christopher Kelly and Scott Lundberg and Marco Tulio Ribeiro and Marc Wilson and Xiaoxuan Liu and Viknesh Sounderajah and Jonathan Carlson and Matthew P Lungren and Bay Gross and Peter Hames and Mustafa Suleyman and Dominic King and Eric Horvitz},
      year={2023},
      eprint={2306.022405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@software{mai_dx_orchestrator,
    title={Open-MAI-Dx-Orchestrator: An Interactive Implementation of Sequential Diagnosis with Language Models},
    author={The-Swarm-Corporation},
    year={2024},
    url={https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator}
}
```