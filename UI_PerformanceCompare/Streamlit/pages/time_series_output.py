import streamlit as st


def render_time_series_output():
    st.markdown('<div class="pc-title">Time Series Output</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pc-subtitle">Reserved for trajectory/time-series model comparison.</div>',
        unsafe_allow_html=True,
    )
    st.info("This page is intentionally empty for now. We are focusing on Image based output first.")
