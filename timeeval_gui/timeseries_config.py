from typing import Any, List, Dict

from gutenTAG.anomalies import Anomaly
from gutenTAG.base_oscillations import BaseOscillationInterface, BaseOscillation
from gutenTAG.generator import TimeSeries
from gutenTAG.generator.parser import ConfigParser


class TimeSeriesConfig:
    def __init__(self):
        self.config: Dict[str, Any] = {
            "base-oscillations": [],
            "anomalies": []
        }

    def set_name(self, name: str):
        self.config["name"] = name

    def set_length(self, length: int):
        self.config["length"] = length

    def add_base_oscillation(self, kind: str, **kwargs):
        self.config["base-oscillations"].append(dict(kind=kind, **kwargs))

    def add_anomaly(self, **kwargs):
        self.config["anomalies"].append(kwargs)

    def generate_base_oscillations(self) -> List[BaseOscillationInterface]:
        parser = ConfigParser()
        return parser._build_base_oscillations(self.config)

    def generate_anomalies(self) -> List[Anomaly]:
        parser = ConfigParser()
        anomalies = parser._build_anomalies(self.config)
        return anomalies

    def generate_timeseries(self, supervised: bool = False, semi_supervised: bool = False) -> TimeSeries:
        return TimeSeries(self.generate_base_oscillations(), self.generate_anomalies(), self.name,
                          supervised=supervised,
                          semi_supervised=semi_supervised)

    def __getattr__(self, item):
        return self.config[item]
