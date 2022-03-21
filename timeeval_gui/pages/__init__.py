from enum import Enum
import streamlit as st

from .gutentag import page as gutentag_page
from .eval import page as eval_page


def Pages():
    if st.sidebar.button("GutenTAG"):
        st.session_state['page'] = "gutentag"
    if st.sidebar.button("Eval"):
        st.session_state['page'] = "eval"

    if "page" in st.session_state and st.session_state.page == "gutentag":
        gutentag_page()
    elif "page" in st.session_state and st.session_state.page == "eval":
        eval_page()
    else:
        gutentag_page()
