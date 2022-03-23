from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from timeeval import DatasetManager

from .page import Page


def plot_boxplot(df, n_show: Optional[int] = None, title="Box plots", ax_label="values", fmt_label=lambda x: x) -> plt.Figure:
    if n_show is not None:
        n_show = n_show // 2
        title = title + f" (worst {n_show} and best {n_show} algorithms)"
        df_boxplot = pd.concat([df.iloc[:, :n_show], df.iloc[:, -n_show:]])
    else:
        title = title + f" (sorted by {ax_label})"
        df_boxplot = df
    labels = df_boxplot.columns
    labels = [fmt_label(c) for c in labels]
    values = [df_boxplot[c].dropna().values for c in df_boxplot.columns]
    fig = plt.figure()
    ax = fig.gca()
    ax.boxplot(values, sym="", vert=True, meanline=True, showmeans=True, showfliers=False,
               manage_ticks=True)
    ax.set_ylabel(ax_label)
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=-45, ha="left", rotation_mode="anchor")
    # add vline to separate bad and good algos
    ymin, ymax = ax.get_ylim()
    if n_show is not None:
        ax.vlines([n_show + 0.5], ymin, ymax, colors="black", linestyles="dashed")
    fig.tight_layout()
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


def plot_scores(algorithm_name, dataset_name, df, dmgr, result_path, **kwargs):
    if not isinstance(algorithm_name, list):
        algorithms = [algorithm_name]
    else:
        algorithms = algorithm_name
    # construct dataset ID
    dataset_id = ("GutenTAG", f"{dataset_name}.unsupervised")

    # load dataset details
    df_dataset = dmgr.get_dataset_df(dataset_id)

    # check if dataset is multivariate
    dataset_dim = df.loc[df["dataset_name"] == dataset_name, "dataset_input_dimensionality"].unique().item()
    dataset_dim = dataset_dim.lower()

    auroc = {}
    df_scores = pd.DataFrame(index=df_dataset.index)
    skip_algos = []
    algos = []
    for algo in algorithms:
        algos.append(algo)
        # get algorithm metric results
        try:
            auroc[algo] = df.loc[(df["algorithm"] == algo) & (df["dataset_name"] == dataset_name), "ROC_AUC"].item()
        except ValueError:
            st.warning(f"No ROC_AUC score found! Probably {algo} was not executed on {dataset_name}.")
            auroc[algo] = -1
            skip_algos.append(algo)
            continue

        # load scores
        training_type = df.loc[df["algorithm"] == algo, "algo_training_type"].values[0].lower().replace("_", "-")
        try:
            df_scores[algo] = load_scores_df(algo, ("GutenTAG", f"{dataset_name}.{training_type}"), df, result_path).iloc[:, 0]
        except (ValueError, FileNotFoundError):
            st.warning(f"No anomaly scores found! Probably {algo} was not executed on {dataset_name}.")
            df_scores[algo] = np.nan
            skip_algos.append(algo)
    algorithms = [a for a in algos if a not in skip_algos]

    return plot_scores_plt(algorithms, auroc, df_scores, df_dataset, dataset_dim, dataset_name, **kwargs)


def plot_scores_plt(algorithms, auroc, df_scores, df_dataset, dataset_dim, dataset_name, **kwargs):
    import matplotlib.pyplot as plt

    # Create plot
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
    if dataset_dim == "multivariate":
        for i in range(1, df_dataset.shape[1] - 1):
            axs[0].plot(df_dataset.index, df_dataset.iloc[:, i], label=f"channel-{i}")
    else:
        axs[0].plot(df_dataset.index, df_dataset.iloc[:, 1], label=f"timeseries")
    axs[1].plot(df_dataset.index, df_dataset["is_anomaly"], label="label")

    for algo in algorithms:
        axs[1].plot(df_scores.index, df_scores[algo], label=f"{algo}={auroc[algo]:.4f}")
    axs[0].legend()
    axs[1].legend()
    fig.suptitle(f"Results of {','.join(np.unique(algorithms))} on {dataset_name}")
    fig.tight_layout()
    return fig


class Results(Page):
    def _get_name(self) -> str:
        return "Results"

    def _preprocess_results(self, res: pd.DataFrame) -> pd.DataFrame:
        res["dataset_name"] = res["dataset"].str.split(".").str[0]
        res["overall_time"] = res["execute_main_time"].fillna(0) + res["train_main_time"].fillna(0)
        res = res.drop_duplicates()
        return res

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

    def _plot_experiment(self, res: pd.DataFrame, dmgr: DatasetManager, results_path: Path):
        st.header("Plot Single Experiment")
        col1, col2, col3 = st.columns(3)
        with col1:
            collection = st.selectbox("Collection", options=res["collection"].unique())
        with col2:
            dataset = st.selectbox("Dataset", res[res.collection == collection]["dataset_name"].unique())
        with col3:
            algorithm_name = st.selectbox("Algorithm", res[(res.collection == collection) & (res.dataset_name == dataset) & (res.status == "Status.OK")]["algorithm"].unique())
        if st.button("Plot"):
            fig = plot_scores(algorithm_name, dataset, res, dmgr, results_path)
            st.pyplot(fig)

    def _df_overall_scores(self, res: pd.DataFrame) -> pd.DataFrame:
        aggregations = ["min", "mean", "median", "max"]
        df_overall_scores = res.pivot_table(index="algorithm", values="ROC_AUC", aggfunc=aggregations)
        df_overall_scores.columns = aggregations
        df_overall_scores = df_overall_scores.sort_values(by="mean", ascending=False)
        return df_overall_scores

    def _quality_summary(self, res: pd.DataFrame):
        df_asl = res.pivot(index="algorithm", columns="dataset_name", values="ROC_AUC")
        df_asl = df_asl.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df_asl["mean"] = df_asl.agg("mean", axis=1)
        df_asl = df_asl.sort_values(by="mean", ascending=True)
        df_asl = df_asl.drop(columns="mean").T

        st.header("Quality Summary")
        if st.checkbox("Show only best and worse", key="nshow-check-quality"):
            n_show = st.number_input("Show worst and best n algorithms", key="nshow_roc", min_value=2)
        else:
            n_show = None
        fmt_label = lambda c: f"{c} (ROC_AUC={self._df_overall_scores(res).loc[c, 'mean']:.2f})"

        fig = plot_boxplot(df_asl, n_show=n_show, title="AUC_ROC box plots", ax_label="AUC_ROC score", fmt_label=fmt_label)
        st.pyplot(fig)

    def _runtime_summary(self, res: pd.DataFrame):
        df_arl = res.pivot(index="algorithm", columns="dataset_name", values="overall_time")
        df_arl = df_arl.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df_arl["mean"] = df_arl.agg("mean", axis=1)
        df_arl = df_arl.sort_values(by="mean", ascending=True)
        df_arl = df_arl.drop(columns="mean").T

        st.header("Runtime Summary")
        if st.checkbox("Show only best and worse", key="nshow-check-rt"):
            n_show = st.number_input("Show worst and best n algorithms", key="nshow_rt", min_value=2)
        else:
            n_show = None
        fmt_label = lambda c: f"{c} (ROC_AUC={self._df_overall_scores(res).loc[c, 'mean']:.2f})" if c in self._df_overall_scores(res).index else c
        fig = plot_boxplot(df_arl, n_show=n_show, title="Overall runtime box plots", ax_label="Overall runtime (in seconds)", fmt_label=fmt_label)
        st.pyplot(fig)

    def render(self):
        st.title(self.name)

        results_path = st.text_input("Choose experiment run results folder", placeholder="/home/user/results")
        data_path = st.text_input("Choose location of datasets folder", placeholder="/home/user/data")
        if results_path != "" and data_path != "":
            results_path = Path(results_path)
            data_path = Path(data_path)
            res = pd.read_csv(results_path / "results.csv")
            res = self._preprocess_results(res)
            dmgr = DatasetManager(data_path)

            self._overall_results(res)
            self._error_summary(res)
            self._quality_summary(res)
            self._runtime_summary(res)
            self._plot_experiment(res, dmgr, results_path)
