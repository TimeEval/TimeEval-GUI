import numpy as np
import pandas as pd
import streamlit as st
import time


def page():
    st.title("Eval")

    st.write("## Algorithms")

    algorithms = st.multiselect("Algorithms", options=["K-Means", "DADS", "Series2Graph"])

    st.write("### Parameters")

    for algorithm in algorithms:
        st.write(f"#### {algorithm}")
        n_param = st.number_input(f"#Parameter Settings", key=f"{algorithm}#params", min_value=0)
        for p in range(n_param):
            with st.expander(f"{algorithm} - Parameter Setting #{p+1}"):
                # todo: unfake
                st.number_input("window_size", key=f"ws-{algorithm}-{p}", min_value=1)
                st.number_input("n_clusters", key=f"nc-{algorithm}-{p}", min_value=1)

    st.write("## Datasets")

    datasets = st.multiselect("Datasets", options=["Traffic", "ECG", "VLDB-Demo-TS"])

    st.write("## General Settings")

    st.write("### Remote Configuration")
    st.text_input("Scheduler Host")
    st.text_area("Worker Hosts")

    st.write("### Resource Constraints")
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
