import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from timeeval import DatasetManager, Datasets
import plotly.graph_objects as go

from .page import Page
from ..files import Files


@st.cache(show_spinner=True, max_entries=1)
def load_results(results_path: Path) -> pd.DataFrame:
    res = pd.read_csv(results_path / "results.csv")
    res["dataset_name"] = res["dataset"]
    res["overall_time"] = res["execute_main_time"].fillna(0) + res["train_main_time"].fillna(0)
    res["algorithm-index"] = res.algorithm + "-" + res.index.astype(str)
    res = res.drop_duplicates()
    return res


@st.cache(show_spinner=True, max_entries=1)
def create_dmgr(data_path: Path) -> Datasets:
    return DatasetManager(data_path, create_if_missing=False)


@st.cache(show_spinner=True, max_entries=100, hash_funcs={pd.DataFrame: pd.util.hash_pandas_object, "builtins.function": lambda _: None})
def plot_boxplot(df, n_show: Optional[int] = None, title="Box plots", ax_label="values", metric="ROC_AUC", _fmt_label=lambda x: x, log: bool = False) -> go.Figure:
    df_asl = df.copy()
    df_asl["dataset_name"] = df_asl["dataset_name"].str.split(".").str[0]
    df_asl = df_asl.pivot(index="algorithm-index", columns="dataset_name", values=metric)
    df_asl = df_asl.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df_asl["median"] = df_asl.median(axis=1)
    df_asl = df_asl.sort_values(by="median", ascending=True)
    df_asl = df_asl.drop(columns="median").T

    fig = go.Figure()
    for i, c in enumerate(df_asl.columns):
        fig.add_trace(go.Box(
            x=df_asl[c],
            name=_fmt_label(c),
            boxpoints=False,
            visible="legendonly" if n_show is not None and n_show < i < len(df_asl.columns) - n_show else None
        ))
    fig.update_layout(
        title={"text": title, "xanchor": "center", "x": 0.5},
        xaxis_title=ax_label,
        legend_title="Algorithms"
    )
    if log:
        fig.update_xaxes(type="log")
    return fig


def load_scores_df(algorithm_name, dataset_id, df, result_path, repetition=1):
    params_id = df.loc[(df["algorithm"] == algorithm_name) & (df["collection"] == dataset_id[0]) & (df["dataset"] == dataset_id[1]), "hyper_params_id"].item()
    path = (
        result_path /
        algorithm_name /
        params_id /
        dataset_id[0] /
        dataset_id[1] /
        str(repetition) /
        "anomaly_scores.ts"
    )
    return pd.read_csv(path, header=None)


def plot_scores(algorithm_name, collection_name, dataset_name, df, dmgr, result_path, **kwargs):
    if not isinstance(algorithm_name, list):
        algorithms = [algorithm_name]
    else:
        algorithms = algorithm_name
    # construct dataset ID
    if collection_name == "GutenTAG" and not dataset_name.endswith("supervised"):
        dataset_id = (collection_name, f"{dataset_name}.unsupervised")
    else:
        dataset_id = (collection_name, dataset_name)

    # load dataset details
    df_dataset = dmgr.get_dataset_df(dataset_id)

    # check if dataset is multivariate
    dataset_dim = df.loc[(df["collection"] == collection_name) & (df["dataset_name"] == dataset_name), "dataset_input_dimensionality"].unique().item()
    dataset_dim = dataset_dim.lower()

    auroc = {}
    df_scores = pd.DataFrame(index=df_dataset.index)
    skip_algos = []
    algos = []
    for algo in algorithms:
        algos.append(algo)
        # get algorithm metric results
        try:
            auroc[algo] = df.loc[(df["algorithm"] == algo) & (df["collection"] == collection_name) & (df["dataset_name"] == dataset_name), "ROC_AUC"].item()
        except ValueError:
            st.warning(f"No ROC_AUC score found! Probably {algo} was not executed on {dataset_name}.")
            auroc[algo] = -1
            skip_algos.append(algo)
            continue

        # load scores
        try:
            df_scores[algo] = load_scores_df(algo, dataset_id, df, result_path).iloc[:, 0]
        except (ValueError, FileNotFoundError):
            st.warning(f"No anomaly scores found! Probably {algo} was not executed on {dataset_name}.")
            df_scores[algo] = np.nan
            skip_algos.append(algo)
    algorithms = [a for a in algos if a not in skip_algos]

    fig = plot_scores_plotly(algorithms, auroc, df_scores, df_dataset, dataset_dim, dataset_name.split(".")[0])
    st.plotly_chart(fig)


def plot_scores_plotly(algorithms, auroc, df_scores, df_dataset, dataset_dim, dataset_name, **kwargs) -> go.Figure:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create plot
    fig = make_subplots(2, 1)
    if dataset_dim == "multivariate":
        for i in range(1, df_dataset.shape[1] - 1):
            fig.add_trace(go.Scatter(x=df_dataset["timestamp"], y=df_dataset.iloc[:, i], name=df_dataset.columns[i]), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=df_dataset["timestamp"], y=df_dataset.iloc[:, 1], name="timeseries"), 1, 1)
    fig.add_trace(go.Scatter(x=df_dataset["timestamp"], y=df_dataset["is_anomaly"], name="label"), 2, 1)

    for algo in algorithms:
        fig.add_trace(go.Scatter(x=df_dataset["timestamp"], y=df_scores[algo], name=f"{algo}={auroc[algo]:.4f}"), 2, 1)
    fig.update_xaxes(matches="x")
    fig.update_layout(
        title=f"Results of {','.join(np.unique(algorithms))} on {dataset_name}",
        height=400
    )
    return fig


class ResultsPage(Page):
    def _get_name(self) -> str:
        return "Results"

    def _overall_results(self, res: pd.DataFrame):
        st.header("Experiment run results")
        st.dataframe(res)

    def _error_summary(self, res: pd.DataFrame):
        st.header("Errors")

        index_columns = ["algo_training_type", "algo_input_dimensionality", "algorithm"]
        df_error_counts = res.pivot_table(index=index_columns, columns=["status"], values="repetition", aggfunc="count")
        df_error_counts = df_error_counts.fillna(value=0).astype(np.int64)
        if "Status.ERROR" in df_error_counts:
            sort_by = ["algo_input_dimensionality", "Status.ERROR"]
        else:
            sort_by = ["algo_input_dimensionality"]
        df_error_counts = df_error_counts.reset_index().sort_values(by=sort_by,
                                                                    ascending=False).set_index(index_columns)

        df_error_counts["ALL"] = \
            df_error_counts.get("Status.ERROR", 0) + \
            df_error_counts.get("Status.OK", 0) + \
            df_error_counts.get("Status.TIMEOUT", 0)

        for tpe in ["SEMI_SUPERVISED", "SUPERVISED", "UNSUPERVISED"]:
            if tpe in df_error_counts.index:
                st.write(tpe)
                st.dataframe(df_error_counts.loc[tpe])

    def _plot_experiment(self, res: pd.DataFrame, dmgr: Datasets, results_path: Path):
        st.header("Plot Single Experiment")
        col1, col2, col3 = st.columns(3)
        with col1:
            collection = st.selectbox("Collection", options=res["collection"].unique())
        with col2:
            dataset = st.selectbox("Dataset", res[res.collection == collection]["dataset_name"].unique(), format_func=lambda x: x.split(".")[0])
        with col3:
            options = res[(res.collection == collection) & (res.dataset_name == dataset) & (res.status.isin(["Status.OK", "OK"]))]["algorithm"].unique()
            options = [None] + list(options)
            algorithm_name = st.selectbox("Algorithm", options, index=0)
        if algorithm_name is not None:
            plot_scores(algorithm_name, collection, dataset, res, dmgr, results_path)

    def _df_overall_scores(self, res: pd.DataFrame) -> pd.DataFrame:
        aggregations = ["min", "mean", "median", "max"]
        df_overall_scores = res.pivot_table(index="algorithm-index", values="ROC_AUC", aggfunc=aggregations)
        df_overall_scores.columns = aggregations
        df_overall_scores = df_overall_scores.sort_values(by="mean", ascending=False)
        return df_overall_scores

    def _quality_summary(self, res: pd.DataFrame):
        df_lut = self._df_overall_scores(res)

        st.header("Quality Summary")
        if len(res.algorithm.unique()) > 2 and st.checkbox("Show only best and worse", key="nshow-check-quality", value=True):
            n_show = st.number_input("Show worst and best n algorithms", key="nshow_roc", min_value=2, max_value=df_lut.shape[0], value=min(df_lut.shape[0], 10))
        else:
            n_show = None
        fmt_label = lambda c: f"{c} (ROC_AUC={df_lut.loc[c, 'mean']:.2f})"

        fig = plot_boxplot(res, n_show=n_show, title="AUC_ROC box plots", ax_label="AUC_ROC score", metric="ROC_AUC", _fmt_label=fmt_label)
        st.plotly_chart(fig)

    def _runtime_summary(self, res: pd.DataFrame):
        df_lut = self._df_overall_scores(res)

        st.header("Runtime Summary")
        if len(res.algorithm.unique()) > 2 and st.checkbox("Show only best and worse", key="nshow-check-rt", value=True):
            n_show = st.number_input("Show worst and best n algorithms", key="nshow_rt", min_value=2, max_value=df_lut.shape[0], value=min(df_lut.shape[0], 10))
        else:
            n_show = None
        fmt_label = lambda c: f"{c} (ROC_AUC={df_lut.loc[c, 'mean']:.2f})" if c in df_lut.index else c

        fig = plot_boxplot(res, n_show=n_show, title="Overall runtime box plots", ax_label="Overall runtime (in seconds)", metric="overall_time", _fmt_label=fmt_label, log=True)
        st.plotly_chart(fig)

    def render(self):
        st.title(self.name)
        files = Files()

        col1, col2 = st.columns(2)

        with col1:
            results_dir = st.text_input(
                "Choose experiment run results parent folder",
                placeholder="/home/user/results",
                value=files.results_folder()
            )

        with col2:
            experiments = [exp for exp in os.listdir(results_dir) if os.path.isdir(Path(results_dir) / exp) and ("results.csv" in os.listdir(Path(results_dir) / exp))]
            results_path = st.selectbox("Choose experiment run results folder", experiments)

        data_path = st.text_input(
            "Choose location of datasets folder",
            placeholder="/home/user/data",
            value=files.timeseries_folder()
        )
        if results_dir != "" and results_path != "" and data_path != "" and len(experiments) > 0:
            results_path = Path(results_dir) / results_path
            data_path = Path(data_path)
            res = load_results(results_path)
            dmgr = create_dmgr(data_path)

            self._overall_results(res)
            self._error_summary(res)
            self._quality_summary(res)
            self._runtime_summary(res)
            self._plot_experiment(res, dmgr, results_path)
