from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import time
from .page import Page

from timeeval import Algorithm
from timeeval_experiments import algorithms as timeeval_algorithms

# keep this import!
from timeeval_experiments.algorithms import *


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

            algorithms: List[Algorithm] = st.multiselect("Algorithms", options=algos, format_func=lambda a: a.name)

            st.write("### Parameters")

            for algorithm in algorithms:
                algo_name = algorithm.name
                st.write(f"#### {algo_name}")
                n_param = st.number_input(f"#Parameter Settings", key=f"{algo_name}#params", min_value=0)
                for p in range(n_param):
                    with st.expander(f"{algo_name} - Parameter Setting #{p + 1}"):
                        for param in algorithm.param_schema:
                            st.text_input(algorithm.param_schema[param]["name"],
                                          value=algorithm.param_schema[param]["defaultValue"],
                                          help=algorithm.param_schema[param]["description"])

                        # todo: unfake
                        # st.number_input("window_size", key=f"ws-{algo_name}-{p}", min_value=1)
                        # st.number_input("n_clusters", key=f"nc-{algo_name}-{p}", min_value=1)

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
