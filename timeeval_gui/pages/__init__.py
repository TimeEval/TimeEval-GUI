from enum import Enum
import streamlit as st

from .gutentag import page as gutentag_page


def Pages():
    gutentag = st.sidebar.button("GutenTAG")
    eval = st.sidebar.button("Eval")

    if gutentag or not eval:
        gutentag_page()
