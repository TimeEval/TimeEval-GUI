from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
import time

from streamlit.state import NoValue

from .page import Page

from timeeval import Algorithm
from timeeval.params import FixedParameters, FullParameterGrid, IndependentParameterGrid
from timeeval_experiments import algorithms as timeeval_algorithms

# keep this import!
from timeeval_experiments.algorithms import *


algos: List[Algorithm] = [eval(f"{a}()") for a in dir(timeeval_algorithms) if "__" not in a]

for algo in algos:
    st.session_state.setdefault(f"eval-{algo.name}-n_params", 0)


def inc_n_params(algo: str):
    value = st.session_state.get(f"eval-{algo}-n_params", 0)
    st.session_state[f"eval-{algo}-n_params"] = value + 1


def dec_n_params(algo: str):
    value = st.session_state.get(f"eval-{algo}-n_params", 0)
    if value > 0:
        st.session_state[f"eval-{algo}-n_params"] = value - 1


def parse_enum_param_type(tpe: str) -> List[str]:
    option_str = tpe.split("[")[1].split("]")[0]
    return option_str.split(",")


def parse_list_value(tpe: str, value: str) -> List[Union[float, int, str, bool]]:
    subtype = tpe.split("[")[1].split("]")[0].lower()
    cc_subtype = {
        "int": int,
        "float": float,
        "bool": bool
    }
    value_str = value.split("[")[1].split("]")[0]

    values = value_str.split(",")
    return [cc_subtype.get(subtype, str)(v) for v in values]


def param_input(param_config: Dict[str, Any], key: Optional[str] = None) -> Any:
    tpe = param_config["type"]
    label = "Value"

    def get_default_value(cc) -> Any:
        value = param_config["defaultValue"]
        if value is None:
            return NoValue()
        else:
            return cc(value)

    if tpe.lower() == "int":
        value = int(st.number_input(label,
                                    value=get_default_value(int),
                                    help=param_config["description"],
                                    step=1,
                                    key=key))
    elif tpe.lower() == "float":
        value = st.number_input(label,
                                value=get_default_value(float),
                                help=param_config["description"],
                                step=None,
                                format="%f",
                                key=key)
    elif tpe.lower().startswith("bool"):
        st.markdown("Value")
        value = st.checkbox("",
                            value=get_default_value(bool),
                            help=param_config["description"],
                            key=key)
    elif tpe.lower().startswith("enum"):
        try:
            default_index = parse_enum_param_type(tpe).index(param_config["defaultValue"])
        except ValueError:
            default_index = 0
        value = st.selectbox(label,
                             options=parse_enum_param_type(tpe),
                             index=default_index,
                             help=param_config["description"],
                             key=key)
    elif tpe.lower().startswith("list"):
        value = st.text_input(label + f" ({tpe.lower()})",
                              value=param_config["defaultValue"],
                              help=param_config["description"],
                              key=key)
        value = parse_list_value(tpe, value)
        print(value)
    # elif tpe.lower() == "string":
    else:
        value = st.text_input(label,
                              value=param_config["defaultValue"],
                              help=param_config["description"],
                              key=key)
    return value


class EvalPage(Page):

    def _get_name(self) -> str:
        return "Eval"

    def render(self):
        st.image("images/timeeval.png")
        st.title("Eval")

        # col1, col2 = st.columns(2)
        # with col1:
        st.write("## Algorithms")

        algo_names: List[str] = st.multiselect("Algorithms", options=[a.name for a in algos])
        algorithms = [a for a in algos if a.name in algo_names]

        st.write("### Parameters")

        for algorithm in algorithms:
            algo_name = algorithm.name
            with st.expander(algo_name):
                if not algorithm.param_schema:
                    st.info("Algorithm has no parameters.")
                    continue

                param_config_tpe = st.selectbox("Parameter configuration type",
                                                [FixedParameters, FullParameterGrid, IndependentParameterGrid],
                                                format_func=lambda x: x.__name__,
                                                help="FixedParameters - Single parameters setting with one value for each.\n"
                                                     "FullParameterGrid - Grid of parameters with a discrete number of "
                                                     "values for each. Yields the full cartesian product of all "
                                                     "available parameter combinations.\n"
                                                     "IndependentParameterGrid - Grid of parameters with a discrete "
                                                     "number of values for each. The parameters in the dict are "
                                                     "considered independent and explored one after the other.",
                                                key=f"eval-{algo_name}-config-tpe")
                st.write("---")

                if param_config_tpe.__name__ == FixedParameters.__name__:
                    n_params = st.session_state.get(f"eval-{algo_name}-n_params", 0)
                    displayed_params = []
                    param_grid = {}
                    for i in range(n_params):
                        param_col1, param_col2 = st.columns((2, 1))
                        with param_col1:
                            param = st.selectbox("Parameter",
                                                 options=[param for param in algorithm.param_schema if
                                                          param not in displayed_params],
                                                 format_func=lambda p: algorithm.param_schema[p]["name"],
                                                 key=f"{algo_name}-parameter-name-{i}")
                        with param_col2:
                            value = param_input(algorithm.param_schema[param], key=f"{algo_name}-parameter-value-{i}")
                        displayed_params.append(param)
                        param_grid[param] = value

                    bt_col1, bt_col2, _ = st.columns((1, 1, 18))
                    with bt_col1:
                        st.button("-",
                                  help="Remove a parameter configuration",
                                  on_click=dec_n_params, args=[algo_name],
                                  key=f"eval-{algo_name}-button-")
                    with bt_col2:
                        st.button("+",
                                  help="Add a parameter configuration",
                                  on_click=inc_n_params, args=[algo_name],
                                  key=f"eval-{algo_name}-button+")
                    algo.param_config = FixedParameters(param_grid)

                else:
                    st.error("Not implemented yet!")

        # with col2:
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
