"""Configuration helpers for the Streamlit app."""

import streamlit as st
from dotenv import load_dotenv


def setup_page():
    """Load environment variables and apply Streamlit page configuration."""
    load_dotenv()
    st.set_page_config(
        page_title="MAI-DxO Interactive Diagnostic Tool",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
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
""",
        unsafe_allow_html=True,
    )
