import pathlib
import sys

import streamlit as st


APP_DIR = pathlib.Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from views.image_based_output import render_image_based_output
from views.summary_output import render_summary_output
from views.time_series_output import render_time_series_output


st.set_page_config(
    page_title="AI Pedsim | Performance Compare",
    page_icon="PC",
    layout="wide",
)


def inject_style():
    st.markdown(
        """
        <style>
        :root {
            --pc-border: rgba(49, 54, 63, 0.18);
            --pc-muted: rgba(49, 54, 63, 0.66);
            --pc-soft: rgba(49, 54, 63, 0.055);
            --pc-accent: #0f766e;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--pc-border);
        }

        [data-testid="stSidebar"] h1 {
            font-size: 1.15rem;
            margin-bottom: 0.3rem;
        }

        .pc-kicker {
            color: var(--pc-muted);
            font-size: 0.86rem;
            margin-bottom: 1.2rem;
        }

        .pc-title {
            font-size: 2rem;
            font-weight: 760;
            letter-spacing: 0;
            margin: 0 0 0.2rem;
        }

        .pc-subtitle {
            color: var(--pc-muted);
            margin-bottom: 1.1rem;
        }

        .pc-panel {
            border: 1px solid var(--pc-border);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.5);
        }

        .pc-section-label {
            color: var(--pc-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--pc-border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: var(--pc-soft);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_style()

    st.sidebar.title("Performance Compare")
    st.sidebar.markdown(
        '<div class="pc-kicker">AI output viewer only</div>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Menu",
        ["Image based output", "Time series output", "Summary output"],
        index=0,
    )

    if page == "Image based output":
        render_image_based_output()
    elif page == "Summary output":
        render_summary_output()
    else:
        render_time_series_output()


if __name__ == "__main__":
    main()
