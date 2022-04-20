import logging
from typing import List, Dict, Any, Optional, Union

import docker
import psutil
import streamlit as st
import timeeval_gui.st_redirect as rd
from durations import Duration
from streamlit.state import NoValue
from timeeval import Algorithm, ResourceConstraints, Metric, TimeEval
from timeeval.params import FixedParameters, FullParameterGrid, IndependentParameterGrid
from timeeval.resource_constraints import GB
from timeeval_experiments import algorithms as timeeval_algorithms

from .page import Page
from ..config import SKIP_DOCKER_PULL
from ..files import Files

# keep this import!
from timeeval_experiments.algorithms import *


algos: List[Algorithm] = [eval(f"{a}(skip_pull={SKIP_DOCKER_PULL})") for a in dir(timeeval_algorithms) if "__" not in a]
if SKIP_DOCKER_PULL:
    # filter out non-existent images from algorithm choices
    docker_client = docker.from_env()

    def image_exists(name: str, tag: str) -> bool:
        images = docker_client.images.list(name=f"{name}:{tag}")
        return len(images) > 0

    algos = [a for a in algos if image_exists(a.main.image_name, a.main.tag)]
    del docker_client

for algo in algos:
    st.session_state.setdefault(f"eval-{algo.name}-n_params", 0)


def inc_n_params(algo_name: str):
    value = st.session_state.get(f"eval-{algo_name}-n_params", 0)
    st.session_state[f"eval-{algo_name}-n_params"] = value + 1


def dec_n_params(algo_name: str):
    value = st.session_state.get(f"eval-{algo_name}-n_params", 0)
    if value > 0:
        st.session_state[f"eval-{algo_name}-n_params"] = value - 1


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
        v = param_config["defaultValue"]
        if v is None:
            return NoValue()
        else:
            return cc(v)

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
    else:  # elif tpe.lower() == "string":
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
                    algorithm.param_config = FixedParameters(param_grid)

                else:
                    st.error("Not implemented yet!")

        st.write("## Datasets")

        dmgr = Files().dmgr()
        available_datasets = dmgr.df().index.values
        datasets = st.multiselect("Datasets", options=available_datasets, format_func=lambda x: f"{x[0]}/{x[1]}")

        st.write("## General Settings")

        repetitions = st.slider("Repetitions", value=1, min_value=1, max_value=1000, step=1)
        metrics = st.multiselect("Metrics",
                                 options=[m for m in Metric if m not in {Metric.RANGE_F1, Metric.RANGE_RECALL, Metric.RANGE_PRECISION}],
                                 default=Metric.default_list(),
                                 format_func=lambda m: m.name)
        force_training_type_match = st.checkbox("Force training type match between algorithm and dataset", value=False)
        force_dimensionality_match = st.checkbox(
            "Force dimensionality match between algorithm and dataset (uni- or multivariate)",
            value=False
        )

        # with st.expander("Remote Configuration"):
        #     st.text_input("Scheduler Host")
        #     st.text_area("Worker Hosts")

        with st.expander("Resource Constraints"):
            rc = ResourceConstraints.default_constraints()
            rc.tasks_per_host = st.number_input("Parallel tasks (distributes CPUs and memory evenly across tasks)",
                                                value=rc.tasks_per_host,
                                                min_value=1,
                                                max_value=psutil.cpu_count())
            rc.train_timeout = Duration(st.text_input(
                "Train Timeout",
                value=rc.train_timeout.representation,
                help="Timeout for the training step of the algorithms as a duration (e.g., '2 minutes' or '1 hour').",
            ))
            rc.execute_timeout = Duration(st.text_input(
                "Execute Timeout",
                value=rc.execute_timeout.representation,
                help="Timeout for the execution step of the algorithms as a duration (e.g., '2 minutes' or '1 hour').",
            ))
            cpu_limit = rc.task_cpu_limit if rc.task_cpu_limit else 0.
            cpu_limit = st.number_input("CPU Limit (overwrites default constraints)",
                                        value=cpu_limit,
                                        min_value=0.,
                                        max_value=float(psutil.cpu_count()),
                                        help="Maximum amount of CPU shares to be used per task, where 2.5 = 2.5 CPU cores and 0 = no limit.")
            if cpu_limit > 0:
                rc.task_cpu_limit = cpu_limit
            else:
                rc.task_cpu_limit = None
            memory_limit = rc.task_memory_limit if rc.task_memory_limit else 0.
            memory_limit = st.number_input("Memory Limit (GB) (overwrites default constraints)",
                                           value=memory_limit,
                                           min_value=0.,
                                           max_value=float(psutil.virtual_memory().total / GB),
                                           help="Maximum amount of memory (in GB) to be used per task, where 0 = no limit.")
            if memory_limit > 0:
                rc.task_memory_limit = int(memory_limit * GB)
            else:
                rc.task_memory_limit = None
            limits = rc.get_compute_resource_limits()
            st.info(f"Resulting resource limits: cpu={limits[1]:.2f}, mem={limits[0] / GB:.0f} GB")

        if st.button("Start Experiment"):
            timeeval = TimeEval(
                dmgr, datasets, algorithms,
                results_path=Files().results_folder(),
                distributed=False,
                repetitions=repetitions,
                resource_constraints=rc,
                metrics=metrics,
                skip_invalid_combinations=True,
                force_training_type_match=force_training_type_match,
                force_dimensionality_match=force_dimensionality_match,
                disable_progress_bar=True
            )

            # reset logging backend
            logging.root.handlers = []
            logging.basicConfig(
                filename=timeeval.results_path / "timeeval.log",
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
            )

            st.info("Running evaluation - please wait...")
            st_out = st.empty()
            with rd.stdouterr(to=st_out):
                timeeval.run()
            st.success(f"... evaluation done!")

            st.write("## Results")

            df_results = timeeval.get_results(aggregated=True, short=True)
            st.dataframe(df_results)
