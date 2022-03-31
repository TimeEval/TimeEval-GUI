from pathlib import Path
from typing import Dict, Hashable, Any, Optional

import requests
import yaml

from timeeval_gui.config import GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL, TIMEEVAL_FILES_PATH


class Files:
    _instance: Optional['Files'] = None

    @staticmethod
    def instance() -> 'Files':
        if Files._instance is None:
            Files._instance = Files()
        return Files._instance

    def __init__(self):
        if TIMEEVAL_FILES_PATH.is_absolute():
            self._files_path = TIMEEVAL_FILES_PATH
        else:
            self._files_path = (Path.cwd() / TIMEEVAL_FILES_PATH).absolute()
        self._anomaly_kind_schema_path = TIMEEVAL_FILES_PATH / "cache" / "anomaly-kind.guten-tag-generation-config.schema.yaml"
        self._files_path.mkdir(parents=True, exist_ok=True)
        (self._files_path / "cache").mkdir(exist_ok=True)

    def anomaly_kind_configuration_schema(self) -> Dict[Hashable, Any]:
        # load parameter configuration only once
        if not self._anomaly_kind_schema_path.exists():
            self._load_anomaly_kind_configuration_schema()
        with self._anomaly_kind_schema_path.open("r") as fh:
            return yaml.load(fh, Loader=yaml.FullLoader)

    def _load_anomaly_kind_configuration_schema(self) -> None:
        result = requests.get(GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL)
        print(result)
        with self._anomaly_kind_schema_path.open("w") as fh:
            fh.write(result.text)
