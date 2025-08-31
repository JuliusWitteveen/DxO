import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from mai_dx.interactive import InteractiveDxSession
from mai_dx.export import export_session_to_markdown
from mai_dx.persistence import save_session, load_session, list_sessions, delete_session
from mai_dx.transparency import (
    DiagnosticFlowVisualizer,
    DeliberationTheater,
    AgentInspector,
    ReasoningTraceViewer,
    LimitationsDisplay,
    InteractiveExplorer,
    ConfidenceCalibration,
    create_diagnostic_journey_visualization,
    create_confidence_timeline
)
import plotly.graph_objects as go
from typing import Dict, List, Optional
import markdown2
import sys
from dotenv import load_dotenv, set_key

# Laad omgevingsvariabelen uit het .env-bestand bij het opstarten
load_dotenv()

# Page config
st.set_page_config(
    page_title="MAI-DxO Interactive Diagnostic Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.agent-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.diagnosis-card {
    background-color: #e8f4f8;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}
.status-active {
    background-color: #2ecc71;
    animation: pulse 1s infinite;
}
.status-complete {
    background-color: #3498db;
}
.status-error {
    background-color: #e74c3c;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
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
        'confidence': ConfidenceCalibration()
    }

def is_api_key_set():
    """Controleer of de API-sleutel is ingesteld in de omgevingsvariabelen."""
    return bool(os.getenv("OPENAI_API_KEY"))

def initialize_session(model_name: str = "gpt-4o", mode: str = "no_budget", case_details: str = ""):
    """Initialize a new diagnostic session using InteractiveDxSession"""
    try:
        config = {
            "model_name": model_name,
            "mode": mode,
            "max_iterations": 15,
            "initial_budget": 10000,
            "physician_visit_cost": 300
        }
        
        st.session_state.session = InteractiveDxSession(orchestrator_config=config)
        
        if case_details:
            st.session_state.session.start(case_details)
            return True
        return False
    except Exception as e:
        st.error(f"Failed to initialize session: {str(e)}")
        return False

def process_clinical_response(response: str):
    """Process the clinical response using InteractiveDxSession"""
    if not st.session_state.session:
        return None
    
    try:
        with st.spinner("Processing clinical response..."):
            st.session_state.session.step(response)
            
            # Check if diagnosis is complete
            if st.session_state.session.is_complete:
                last_turn = st.session_state.session.turns[-1]
                return {
                    'diagnosis_complete': True,
                    'final_diagnosis': last_turn.action_request.content,
                    'action': last_turn.action_request
                }
            else:
                last_turn = st.session_state.session.turns[-1]
                return {
                    'diagnosis_complete': False,
                    'next_request': f"{last_turn.action_request.action_type.capitalize()}: {last_turn.action_request.content}",
                    'action': last_turn.action_request
                }
        
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return None

def save_current_session():
    """Save current session using the persistence module"""
    if not st.session_state.session:
        st.warning("No session data to save")
        return
    
    try:
        session_data = st.session_state.session.to_dict()
        session_id = st.session_state.session.session_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_data["saved_at"] = timestamp
        save_session(session_data, session_id)

        st.success(f"Session saved at {timestamp} (ID: {session_id[:8]}...)")
        return session_id
    except Exception as e:
        st.error(f"Failed to save session: {str(e)}")
        return None

def load_saved_session(session_id: str):
    """Load a saved session using the persistence module"""
    try:
        session_data = load_session(session_id)
        st.session_state.session = InteractiveDxSession.from_dict(session_data)
        st.success("Session loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
        return False

def export_to_markdown():
    """Export session to markdown format"""
    if not st.session_state.session:
        return None
    
    try:
        session_data = st.session_state.session.to_dict()
        return export_session_to_markdown(session_data)
    except Exception as e:
        st.error(f"Failed to export: {str(e)}")
        return None

def display_current_request():
    """Display the current request from the AI panel"""
    if not st.session_state.session or not st.session_state.session.turns:
        return
    
    last_turn = st.session_state.session.turns[-1]
    
    if not st.session_state.session.is_complete:
        action = last_turn.action_request
        
        # Display request with appropriate styling
        if action.action_type == "ask":
            icon = "💬"
            color = "blue"
        elif action.action_type == "test":
            icon = "🔬"
            color = "orange"
        else:
            icon = "🎯"
            color = "green"
        
        st.markdown(f"""
        <div style="background-color: #f0f9ff; border-left: 4px solid {color}; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
            <h4 style="margin: 0; color: #333;">{icon} Clinical Request (Turn {last_turn.turn_number})</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{action.content}</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;"><em>Reasoning: {action.reasoning}</em></p>
        </div>
        """, unsafe_allow_html=True)

def display_differential_diagnosis():
    """Display current differential diagnosis with visual indicators"""
    if not st.session_state.session or not st.session_state.session.case_state:
        return
    
    differential = st.session_state.session.case_state.differential_diagnosis
    if not differential:
        st.info("No differential diagnosis formulated yet")
        return
    
    st.subheader("📊 Current Differential Diagnosis")
    
    # Sort by probability
    sorted_diff = sorted(differential.items(), key=lambda x: x[1], reverse=True)
    
    for i, (diagnosis, probability) in enumerate(sorted_diff[:5]):
        # Color coding based on probability
        if probability > 0.5:
            color = "#2ecc71"
            icon = "🟢"
        elif probability > 0.2:
            color = "#f39c12"
            icon = "🟡"
        else:
            color = "#e74c3c"
            icon = "🔴"
        
        # Progress bar visualization
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold;">{icon} {diagnosis}</span>
                <span style="color: {color}; font-weight: bold;">{probability:.1%}</span>
            </div>
            <div style="background-color: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background-color: {color}; width: {probability*100}%; height: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_session_history():
    """Display session history with expandable details"""
    if not st.session_state.session or not st.session_state.session.turns:
        return
    
    st.subheader("📋 Session History")
    
    # Display in reverse order (most recent first)
    for turn in reversed(st.session_state.session.turns[-5:]):
        action = turn.action_request
        
        # Determine icon and color based on action type
        if action.action_type == "ask":
            icon = "💬"
            color = "#3498db"
        elif action.action_type == "test":
            icon = "🔬"
            color = "#e67e22"
        else:
            icon = "🎯"
            color = "#27ae60"
        
        with st.expander(f"{icon} Turn {turn.turn_number} - {action.action_type.upper()}", expanded=(turn == st.session_state.session.turns[-1])):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Request:** {action.content}")
                if turn.physician_input:
                    st.markdown(f"**Your Response:** {turn.physician_input}")
                st.markdown(f"**AI Reasoning:** {action.reasoning}")
            
            with col2:
                if turn.differential_at_turn:
                    st.markdown("**Top Diagnoses:**")
                    for name, prob in list(sorted(turn.differential_at_turn.items(), key=lambda x: x[1], reverse=True))[:3]:
                        st.markdown(f"- {name}: {prob:.1%}")
                
                if hasattr(turn, 'cost_at_turn'):
                    st.metric("Total Cost", f"${turn.cost_at_turn}")

# Main UI
def main():
    st.markdown('<h1 class="main-header">🏥 MAI-DxO Interactive Diagnostic Tool</h1>', unsafe_allow_html=True)
    
    api_key_is_set = is_api_key_set()

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
        
        # Model Selection
        model_name = st.selectbox(
            "Select Model:",
            ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0
        )
        
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
                save_current_session()
        
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
                    if initialize_session(model_name, mode, case_details):
                        st.rerun()
                else:
                    st.warning("Please enter initial case presentation")
        
        with col2:
            if st.button("Load Example Case", use_container_width=True):
                example_case = "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling and bleeding. Symptoms did not abate with antimicrobial therapy. No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable."
                if initialize_session(model_name, mode, example_case):
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
                                result = process_clinical_response(clinical_response)
                                if result:
                                    if result.get('diagnosis_complete'):
                                        st.success(f"🎯 Diagnosis Complete: {result.get('final_diagnosis')}")
                                        st.balloons()
                                    st.rerun()
                            else:
                                st.warning("Please enter clinical findings")
                    
                    with col_btn2:
                        if st.button("Skip/Unknown", use_container_width=True):
                            result = process_clinical_response("Information not available / Unable to perform requested action")
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
                    fig = create_confidence_timeline(st.session_state.session)
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
            fig = create_diagnostic_journey_visualization(st.session_state.session)
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
                <a href="https://github.com/The-Swarm-Corporation/Open-MAI-Dx-Orchestrator" target="_blank">GitHub</a> | 
                <a href="https://arxiv.org/abs/2306.022405" target="_blank">Paper</a>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()