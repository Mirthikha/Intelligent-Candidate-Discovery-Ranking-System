import streamlit as st
import pandas as pd
from src.ranker import run_pipeline

st.set_page_config(page_title="Candidate Leaderboard", layout="wide")
st.title("🏆 Intelligent Candidate Discovery & Ranking System")

# Simple button trigger
if st.button("🔥 Run ranking Pipeline"):
    status_placeholder = st.empty()
    status_placeholder.warning("⚙️ Executing Backend Ranker Layers...")
    
    # Run the backend processing pipeline completely independently
    final_ranked_df = run_pipeline()
    st.session_state['processed_df'] = final_ranked_df
    
    status_placeholder.success("🎉 Processing complete!")

# Clean 5-Column Display
# 3. Clean, Fast Display Matching Your Uploaded Format Exactly
if 'processed_df' in st.session_state:
    st.subheader("📋 Top Ranked Candidates")
    display_df = st.session_state['processed_df'].copy()

    # Pass the final clean table to the UI
    st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)