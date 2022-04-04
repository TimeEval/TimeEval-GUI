from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import time
from .page import Page

from timeeval import Algorithm
from timeeval_experiments.algorithms import *
from timeeval_experiments import algorithms as timeeval_algorithms


algos: List[Algorithm] = [eval(f"{a}()") for a in dir(timeeval_algorithms) if "__" not in a]


class EvalPage(Page):

    def _get_name(self) -> str:
        return "Eval"

    def render(self):
        st.image("images/timeeval.png")
        st.title("Eval")

        col1, col2 = st.columns(2)
        with col1:
            st.write("## Algorithms")

            algorithms = st.multiselect("Algorithms", options=[a.name for a in algos])

            st.write("### Parameters")

            for algorithm in algorithms:
                st.write(f"#### {algorithm}")
                n_param = st.number_input(f"#Parameter Settings", key=f"{algorithm}#params", min_value=0)
                for p in range(n_param):
                    with st.expander(f"{algorithm} - Parameter Setting #{p + 1}"):
                        # todo: unfake
                        st.number_input("window_size", key=f"ws-{algorithm}-{p}", min_value=1)
                        st.number_input("n_clusters", key=f"nc-{algorithm}-{p}", min_value=1)

        with col2:
            st.write("## Datasets")

            datasets = st.multiselect("Datasets", options=["Traffic", "ECG", "ecg-10000-1"])

        st.write("## General Settings")

        with st.expander("Remote Configuration"):
            st.text_input("Scheduler Host")
            st.text_area("Worker Hosts")

        with st.expander("Resource Constraints"):
            st.number_input("Tasks per Host", min_value=1)
            st.number_input("CPU Limit")
            st.time_input("Train Timeout")
            st.time_input("Execute Timeout")
            st.number_input("Memory (GB)")

        if st.button("Start Experiment"):
            st.write("## Results")

            st.write("Progress 100% (ETA: 00:00:00)")
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)

            results = []
            for algorithm in algorithms:
                for dataset in datasets:
                    results.append({
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "ROC_AUC": np.random.randint(60, 100) / 100,
                        "PR_AUC": np.random.randint(30, 100) / 100,
                        "pre_execution_time": np.random.rand(),
                        "main_execution_time": np.random.rand(),
                        "post_execution_time": np.random.rand(),
                        "pre_train_time": np.random.rand(),
                        "main_train_time": np.random.rand(),
                        "post_train_time": np.random.rand(),
                    })

            df = pd.DataFrame(results)
            st.dataframe(df)
