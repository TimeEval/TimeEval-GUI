from pathlib import Path
from typing import Dict, Hashable, Any, Optional

import requests
import yaml
from gutenTAG.generator import TimeSeries
from gutenTAG.generator.timeseries import TrainingType
from gutenTAG.utils.global_variables import UNSUPERVISED_FILENAME, SUPERVISED_FILENAME, SEMI_SUPERVISED_FILENAME

from timeeval_gui.config import GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL, TIMEEVAL_FILES_PATH


class Files:
    _instance: Optional['Files'] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Files, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if TIMEEVAL_FILES_PATH.is_absolute():
            self._files_path = TIMEEVAL_FILES_PATH
        else:
            self._files_path = (Path.cwd() / TIMEEVAL_FILES_PATH).absolute()
        self._files_path.mkdir(parents=True, exist_ok=True)
        self._anomaly_kind_schema_path = self._files_path / "cache" / "anomaly-kind.guten-tag-generation-config.schema.yaml"
        self._anomaly_kind_schema_path.parent.mkdir(exist_ok=True)
        self._ts_path = self._files_path / "timeseries"
        self._ts_path.mkdir(exist_ok=True)

    def anomaly_kind_configuration_schema(self) -> Dict[Hashable, Any]:
        # load parameter configuration only once
        if not self._anomaly_kind_schema_path.exists():
            self._load_anomaly_kind_configuration_schema()
        with self._anomaly_kind_schema_path.open("r") as fh:
            return yaml.load(fh, Loader=yaml.FullLoader)

    def store_ts(self, timeseries: TimeSeries) -> None:
        path = self._ts_path / timeseries.dataset_name
        path.mkdir(exist_ok=True)

        timeseries.to_csv(path / UNSUPERVISED_FILENAME, TrainingType.TEST)
        if timeseries.supervised:
            timeseries.to_csv(path / SUPERVISED_FILENAME, TrainingType.TRAIN_ANOMALIES)
        if timeseries.semi_supervised:
            timeseries.to_csv(path / SEMI_SUPERVISED_FILENAME, TrainingType.TRAIN_NO_ANOMALIES)

    def _load_anomaly_kind_configuration_schema(self) -> None:
        result = requests.get(GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL)
        with self._anomaly_kind_schema_path.open("w") as fh:
            fh.write(result.text)
