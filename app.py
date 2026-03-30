import threshold
from MetabolicTest import MetabolicTest
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# File picker
uploaded_file = st.file_uploader("Select Excel Test File", type=["xlsx"])

if uploaded_file is not None:
    st.session_state['met_test'] = MetabolicTest(uploaded_file)
    st.toast(f"Loaded {uploaded_file.name}", icon="✅")
    threshold.load_threshold_tab()
    
# Page configuration
st.set_page_config(page_title="Threshold Auto-Detection", layout="wide")
