import os
import json
from datetime import datetime
import streamlit as st
from dotenv import set_key, load_dotenv

from ui_controls import render_settings_panel
from llm_client_factory import LLMClient

from mai_dx.persistence import list_sessions, load_session
from mai_dx.ui import (
    is_api_key_set,
    initialize_session,
    process_clinical_response,
    save_current_session,
    load_saved_session,
    export_to_markdown,
    display_current_request,
    display_differential_diagnosis,
    display_session_history,
    DiagnosticFlowVisualizer,
    DeliberationTheater,
    AgentInspector,
    ReasoningTraceViewer,
    LimitationsDisplay,
    InteractiveExplorer,
    ConfidenceCalibration,
    create_diagnostic_journey_visualization,
    create_confidence_timeline,
)

st.set_page_config(layout="wide")
load_dotenv()

def initialize_app_session():
    """Initialize Streamlit's session state with default values."""
    if 'session' not in st.session_state:
        st.session_state.session = None
    if 'visualization_components' not in st.session_state:
        st.session_state.visualization_components = {
            'flow_viz': DiagnosticFlowVisualizer(),
            'theater': DeliberationTheater(),
            'inspector': AgentInspector(),
            'reasoning': ReasoningTraceViewer(),
            'limitations': LimitationsDisplay(),
            'explorer': InteractiveExplorer(),
            'confidence': ConfidenceCalibration(),
        }

def main():
    initialize_app_session()
    SETTINGS = render_settings_panel()

    st.markdown('<h1 class="main-header">🏥 MAI-DxO Interactive Diagnostic Tool</h1>', unsafe_allow_html=True)
    api_key_is_set = is_api_key_set()

    llm = None
    if api_key_is_set:
        llm = LLMClient(
            model=SETTINGS["model_name"],
            temperature=SETTINGS["temperature"],
            top_p=SETTINGS["top_p"],
            max_tokens=SETTINGS["max_tokens"],
            reasoning_effort=SETTINGS["reasoning_effort"],
        )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key Setup
        if not api_key_is_set:
            st.warning("⚠️ API Key Required")
            api_key = st.text_input("Enter OpenAI API Key:", type="password")
            if st.button("Set and Save API Key"):
                if api_key:
                    # Sla de sleutel op in het .env-bestand en de huidige sessie
                    set_key(".env", "OPENAI_API_KEY", api_key)
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("API Key saved locally!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API Key.")
        else:
            st.success("✅ API Key is Set")
            if st.button("🔄 Change API Key"):
                os.environ.pop("OPENAI_API_KEY", None)
                # Maak een leeg .env bestand of verwijder de sleutel
                with open(".env", "w") as f:
                    f.write("")
                st.rerun()

        # Mode Selection
        mode = st.selectbox(
            "Diagnostic Mode:",
            ["no_budget", "budgeted", "question_only"],
            index=0,
            help="no_budget: Full diagnostic capabilities\nbudgeted: Cost-constrained diagnosis\nquestion_only: Only asking questions, no tests"
        )

        # Session Management
        st.header("📁 Session Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Session", use_container_width=True):
                st.session_state.session = None
                st.rerun()

        with col2:
            if st.button("💾 Save Session", use_container_width=True):
                session_id = save_current_session()
                if session_id and SETTINGS.get("save_settings_with_session", True):
                    try:
                        session_file = os.path.join("sessions", f"{session_id}.json")
                        with open(session_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data["settings"] = SETTINGS
                        with open(session_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    except Exception as e:
                        st.error(f"Failed to include settings: {e}")

        # Load session
        try:
            saved_sessions = list_sessions()
            if saved_sessions:
                st.selectbox(
                    "Load Session:",
                    options=[""] + [s['id'] for s in saved_sessions],
                    format_func=lambda x: f"{x[:8]}... ({next((s['last_modified'] for s in saved_sessions if s['id'] == x), '')})" if x else "Select a session",
                    key="session_selector"
                )

                if st.session_state.session_selector and st.button("📂 Load", use_container_width=True):
                    if load_saved_session(st.session_state.session_selector):
                        loaded_session = load_session(st.session_state.session_selector)
                        if loaded_session.get("settings"):
                            st.session_state.dxo_settings = loaded_session["settings"]
                        st.rerun()
        except Exception as e:
            st.error(f"Error loading sessions: {str(e)}")

        # Export Options
        st.header("📤 Export")
        if st.button("📝 Export to Markdown", use_container_width=True):
            md_content = export_to_markdown()
            if md_content:
                st.download_button(
                    label="Download Report",
                    data=md_content,
                    file_name=f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

        # Display session info
        if st.session_state.session:
            st.header("📊 Session Info")
            if st.session_state.session.case_state:
                st.metric("Turns", len(st.session_state.session.turns))
                st.metric("Cost", f"${st.session_state.session.case_state.cumulative_cost}")

                if st.session_state.session.is_complete:
                    st.success("✅ Diagnosis Complete")
                else:
                    st.info("🔄 In Progress")

    # Main Content Area
    if not api_key_is_set:
        st.info("👈 Please enter your OpenAI API Key in the sidebar to begin")
        return

    # Initialize session if needed
    if st.session_state.session is None:
        st.header("🚀 Start New Diagnostic Session")

        # Case details input
        case_details = st.text_area(
            "Enter Initial Case Presentation:",
            height=150,
            placeholder="Example: 45-year-old male presenting with chest pain for 2 hours. The pain is substernal, crushing in nature, radiating to the left arm. BP 140/90, HR 95, RR 18, T 98.6°F. Patient has a history of hypertension and hyperlipidemia.",
            help="Provide the initial clinical presentation including patient demographics, chief complaint, vital signs, and relevant history."
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Start Diagnostic Session", type="primary", use_container_width=True):
                if case_details:
                    if initialize_session(SETTINGS["model_name"], mode, case_details):
                        st.rerun()
                else:
                    st.warning("Please enter initial case presentation")

        with col2:
            if st.button("Load Example Case", use_container_width=True):
                example_case = "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling and bleeding. Symptoms did not abate with antimicrobial therapy. No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable."
                if initialize_session(SETTINGS["model_name"], mode, example_case):
                    st.rerun()

        # Display example cases
        with st.expander("📚 Example Cases"):
            st.markdown("""
            **Case 1 - Acute Chest Pain:**
            > 55-year-old male with 3 hours of crushing chest pain, diaphoresis, and shortness of breath. BP 160/95, HR 110.

            **Case 2 - Pediatric Fever:**
            > 3-year-old with 5 days of high fever, bilateral conjunctivitis, strawberry tongue, and cervical lymphadenopathy.

            **Case 3 - Neurological:**
            > 68-year-old female with sudden onset right-sided weakness and slurred speech 2 hours ago. BP 180/100.
            """)

    # Active session interface
    if st.session_state.session and st.session_state.session.case_state:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical Interface", "🔍 Transparency", "📊 Analytics", "🧪 Advanced"])

        with tab1:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.header("Clinical Interaction")

                # Display current request
                display_current_request()

                # Response input if not complete
                if not st.session_state.session.is_complete:
                    clinical_response = st.text_area(
                        "Enter Clinical Findings:",
                        height=150,
                        placeholder="Enter the clinical findings, test results, or observations requested...",
                        key="clinical_response_input"
                    )

                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
                    with col_btn1:
                        if st.button("Submit Response", type="primary", use_container_width=True):
                            if clinical_response:
                                current_turn = len(st.session_state.session.turns)
                                result = process_clinical_response(clinical_response, current_turn)
                                if result:
                                    if result.get('diagnosis_complete'):
                                        st.success(f"🎯 Diagnosis Complete: {result.get('final_diagnosis')}")
                                        st.balloons()
                                    st.rerun()
                            else:
                                st.warning("Please enter clinical findings")

                    with col_btn2:
                        if st.button("Skip/Unknown", use_container_width=True):
                            skip_message = "Information not available / Unable to perform requested action"
                            current_turn = len(st.session_state.session.turns)
                            result = process_clinical_response(skip_message, current_turn)
                            if result:
                                st.rerun()
                else:
                    # Display final diagnosis
                    last_turn = st.session_state.session.turns[-1]
                    st.success(f"🎯 Final Diagnosis: {last_turn.action_request.content}")
                    st.info(f"Reasoning: {last_turn.action_request.reasoning}")

                # Display session history
                display_session_history()

            with col2:
                st.header("Diagnostic Status")

                # Current stats
                if st.session_state.session.case_state:
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Total Cost", f"${st.session_state.session.case_state.cumulative_cost}")
                        st.metric("Tests", len(st.session_state.session.case_state.tests_performed))
                    with col_stat2:
                        st.metric("Turns", len(st.session_state.session.turns))
                        st.metric("Questions", len(st.session_state.session.case_state.questions_asked))

                # Differential diagnosis
                display_differential_diagnosis()

                # Confidence timeline mini-chart
                if len(st.session_state.session.turns) > 1:
                    st.subheader("📈 Confidence Trend")
                    # Serialize turns to JSON to make it hashable for caching
                    session_dict = st.session_state.session.to_dict()
                    turns_json = json.dumps(session_dict.get("turns", []))
                    fig = create_confidence_timeline(turns_json)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("🔍 Transparency & Reasoning")

            if st.session_state.session.turns:
                # Select turn to inspect
                turn_options = [f"Turn {t.turn_number}" for t in st.session_state.session.turns]
                selected_turn_idx = st.selectbox(
                    "Select Turn to Inspect:",
                    range(len(turn_options)),
                    format_func=lambda x: turn_options[x],
                    index=len(turn_options)-1
                )

                selected_turn = st.session_state.session.turns[selected_turn_idx]

                # Display reasoning flow
                viz = st.session_state.visualization_components['reasoning']
                viz.show_reasoning_flow(selected_turn)

                # Display raw deliberations
                if selected_turn.deliberation:
                    st.subheader("🧠 Agent Deliberations")
                    for agent_name, response in selected_turn.deliberation.items():
                        with st.expander(f"{agent_name}", expanded=False):
                            st.text(response[:1000] + ("..." if len(response) > 1000 else ""))

        with tab3:
            st.header("📊 Diagnostic Analytics")

            # Diagnostic journey visualization
            session_dict = st.session_state.session.to_dict()
            turns_json = json.dumps(session_dict.get("turns", []))
            fig = create_diagnostic_journey_visualization(turns_json)
            st.plotly_chart(fig, use_container_width=True)

            # Confidence analysis
            viz = st.session_state.visualization_components['confidence']
            viz.show_confidence_analysis(st.session_state.session)

        with tab4:
            st.header("🧪 Advanced Features")

            # Explorer for what-if scenarios
            viz = st.session_state.visualization_components['explorer']
            viz.show_exploration_panel(st.session_state.session)

            # System limitations
            viz = st.session_state.visualization_components['limitations']
            viz.show_limitations_panel()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; color: #666;">
                Built with ❤️ using MAI-DxO Framework | 
                <a href="https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orcherstrator" target="_blank">GitHub</a> | 
                <a href="https://arxiv.org/abs/2306.022405" target="_blank">Paper</a>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()