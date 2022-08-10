import streamlit as st

from .gutentag import GutenTAGPage
from .eval import EvalPage
from .page import Page
from .results import ResultsPage


class Pages:
    def __init__(self, *pages: Page):
        assert len(pages) > 0, "You have to add at least one Page"
        self.pages = pages
        self.first_page = self.pages[0].name
        st.session_state.setdefault("page", self.first_page)

    def _render_sidebar(self):
        for page in self.pages:
            name = page.name
            if st.sidebar.button(name):
                st.session_state["page"] = name

    def _render_page(self):
        active_page = st.session_state["page"]
        for page in self.pages:
            if active_page == page.name:
                page.render()
                break

    def render(self):
        self._render_sidebar()
        self._render_page()

    @staticmethod
    def default() -> 'Pages':
        return Pages(GutenTAGPage(), EvalPage(), ResultsPage())
