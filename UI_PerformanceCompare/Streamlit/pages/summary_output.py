import streamlit as st


def render_summary_output():
    st.markdown('<div class="pc-title">Summary Output</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-subtitle">Reserved for cross-method benchmark summaries.</div>',
        unsafe_allow_html=True,
    )
    st.info("This page will aggregate all model runs after the image comparison view is stable.")
