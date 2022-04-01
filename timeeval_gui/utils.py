from typing import Dict, Tuple, List, Type

from timeeval_gui.files import Files


def get_base_oscillations() -> Dict[str, str]:
    return {
        "sine": "Sine",
        "random-walk": "Random Walk",
        "ecg": "ECG",
        "polynomial": "Polynomial",
        "cylinder-bell-funnel": "Cylinder Bell Funnel",
        "random-mode-jump": "Random Mode Jump",
        "formula": "Formula"
    }


def get_anomaly_types() -> Dict[str, str]:
    return {
        "amplitude": "Amplitude",
        "extremum": "Extremum",
        "frequency": "Frequency",
        "mean": "Mean",
        "pattern": "Pattern",
        "pattern-shift": "Pattern Shift",
        "platform": "Platform",
        "trend": "Trend",
        "variance": "Variance",
        "mode-correlation": "Mode Correlation",
    }


def map_types(t: str) -> Type:
    return {
        "boolean": bool,
        "string": str,
        "integer": int,
        "number": float
    }.get(t, str)


def get_anomaly_params(anomaly: str) -> List[Tuple[str, Type]]:
    params = []
    param_config = Files().anomaly_kind_configuration_schema()

    for param_name, param in param_config["definitions"].get(f"{anomaly}-params", {}).get("properties", {}).items():
        params.append((param_name, map_types(param.get("type"))))

    return params
