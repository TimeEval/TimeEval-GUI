from typing import Tuple, Dict

import streamlit as st

from timeeval_gui.timeseries_config import TimeSeriesConfig
from timeeval_gui.utils import get_base_oscillations, get_anomaly_types, get_anomaly_params


def general_area(ts_config: TimeSeriesConfig) -> TimeSeriesConfig:
    ts_config.set_name(st.text_input("Name"))
    ts_config.set_length(st.number_input("Length", min_value=10))
    return ts_config


def select_base_oscillation(key="base-oscillation") -> Tuple[str, str]:
    bos = get_base_oscillations()

    value = st.selectbox("Base-Oscillation", bos.items(), format_func=lambda x: x[1], key=key)
    return value


def select_anomaly_type(key="anomaly-type") -> Tuple[str, str]:
    bos = get_anomaly_types()

    value = st.selectbox("Anomaly Type", bos.items(), format_func=lambda x: x[1], key=key)
    return value


def channel_area(c, ts_config: TimeSeriesConfig) -> TimeSeriesConfig:
    base_oscillation = select_base_oscillation(f"base-oscillation-{c}")
    frequency = st.number_input("Frequency (#repetitions/100)", key=f"frequency-{c}")
    variance = st.number_input("Variance", key=f"variance-{c}")
    ts_config.add_base_oscillation(base_oscillation[0], frequency=frequency, variance=variance)

    return ts_config


def anomaly_area(a, ts_config: TimeSeriesConfig) -> TimeSeriesConfig:
    position = st.selectbox("Position", key=f"anomaly-position-{a}", options=["beginning", "middle", "end"])
    length = int(st.number_input("Length", key=f"anomaly-length-{a}", min_value=1))
    channel = st.selectbox("Channel", key=f"anomaly-channel-{a}", options=list(range(len(ts_config.config["base-oscillations"]))))

    n_kinds = st.number_input("Number of Anomaly Types", key=f"anomaly-types-{a}", min_value=1)
    kinds = []
    for t in range(int(n_kinds)):
        st.write(f"##### Type {t}")
        anomaly_type, _ = select_anomaly_type(f"anomaly-type-{a}-{t}")
        parameters = parameter_area(a, t, anomaly_type)
        kinds.append({"kind": anomaly_type, "parameters": parameters})

    ts_config.add_anomaly(position=position, length=length, channel=channel, kinds=kinds)
    return ts_config


def parameter_area(a, t, anomaly_type: str) -> Dict:
    param_conf = {}
    parameters = get_anomaly_params(anomaly_type)
    for name, p in parameters:
        key = f"{a}-{t}-{name}"
        if p == str:
            param_conf[name] = st.text_input(name.upper(), key=key)
        elif p == bool:
            param_conf[name] = st.checkbox(name.upper(), key=key)
        elif p == int:
            param_conf[name] = st.number_input(name.upper(), key=key, step=1)
        elif p == float:
            param_conf[name] = st.number_input(name.upper(), key=key)
    return param_conf


def page():
    st.title("GutenTAG")

    timeseries_config = TimeSeriesConfig()

    st.write("## General Settings")
    timeseries_config = general_area(timeseries_config)

    st.write("## Channels")
    n_channels = st.number_input("Number of Channels", min_value=1)
    for c in range(n_channels):
        with st.expander(f"Channel {c}"):
            timeseries_config = channel_area(c, timeseries_config)

    st.write("## Anomalies")
    n_anomalies = st.number_input("Number of Anomalies", min_value=0)
    for a in range(n_anomalies):
        with st.expander(f"Anomaly {a}"):
            timeseries_config = anomaly_area(a, timeseries_config)

    if st.button("Build Timeseries"):
        timeseries = timeseries_config.generate_timeseries()
        timeseries.generate()
        st.pyplot(timeseries.build_figure_base_oscillation())

    st.button("Export")
