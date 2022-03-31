import streamlit as st

from timeeval_gui.pages import Pages


def main():
    st.set_page_config(page_title="TimeEval - A Time Series Anomaly Detection Toolkit")
    Pages.default().render()


if __name__ == '__main__':
    main()
