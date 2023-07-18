"""Module used to define Neptune class and logger funcations."""
import logging
import sys
import time
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

import neptune
from neptune.exceptions import NeptuneException
from requests.exceptions import Timeout

from ig import FEATURES_DIRECTORY, MODEL_CONFIGURATION_DIRECTORY, MODELS_DIRECTORY


def init_logger(
    logging_directory: Union[str, Path],
    file_name: Optional[str] = "InfoRun",
) -> None:
    """Init logging function."""
    if isinstance(logging_directory, str):
        logging_directory = Path(logging_directory)
    if not logging_directory.exists():
        logging_directory.mkdir(exist_ok=True, parents=True)
    log_file_path = logging_directory / f"{file_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)-12s: %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=log_file_path,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.getLogger("hyperopt").setLevel(logging.CRITICAL)


def get_logger(name: str) -> Logger:
    """Get logger by name."""
    logger = logging.getLogger(name)

    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.INFO)

    return logger


log: Logger = get_logger("Neptune")


class ModelLogWriter:
    """Log Model Output to a separate log file."""

    def __init__(self, logger_path: str) -> None:
        """Init ModelLogWriter."""
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        self.logger = logging.getLogger("Model/outpus")
        self.logger.propagate = False

        file_handler = logging.FileHandler(logger_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        sys.stdout = self  # type: ignore

    def write(self, buf: str) -> None:
        """Write Method."""
        for line in buf.rstrip().splitlines():
            self.logger.info(line.rstrip())

    def flush(self) -> None:
        """Flush Method."""

    def reset_stdout(self, original_stdout: TextIO) -> None:
        """Reset stdout to system."""
        sys.stdout = original_stdout


class NeptuneLogs:
    """Neptune Class."""

    def __init__(self, configuration: Dict[str, Any], folder_name: str) -> None:
        """Init function for Neptune class."""
        self.configuration = configuration
        self.folder_name = folder_name
        self.runs: Any = None

    @property
    def use_neptune(self) -> bool:
        """Determine whether we use neptune."""
        return self.configuration.get("neptune_logs", False)

    def init_neptune(
        self,
        tags: List[str],
        training_path: Union[Path, str],
        test_path: Optional[Union[Path, str]],
    ) -> None:
        """Init neptune instance."""
        if tags is None:
            tags = []
        if self.use_neptune:
            self.runs = neptune.init_run(
                tags=["IG MODEL"] + tags,
                # mode="offline"
            )
            self.runs["params"] = {
                "run_type": "train",
                "model_name": "IG model",
                "experiment": self.folder_name,
                "training_path": str(training_path),
                "validation_path": str(test_path),
            }

    def upload_configuration_files(self, configuration_path: Path) -> None:
        """Upload configuration file to neptune."""
        if self.runs:
            self._upload_file("configurations/configuration.yml", str(configuration_path))
            for model_config_path in self.configuration["models"]:
                self._upload_file(
                    f"configurations/model_configuration/{model_config_path}",
                    str(MODEL_CONFIGURATION_DIRECTORY / model_config_path),
                )

            for features_path in self.configuration["feature_paths"]:
                self._upload_file(
                    f"configurations/features/{features_path}",
                    str(FEATURES_DIRECTORY / features_path),
                )

    def upload_experiment(self, experiment_path: Path, neptune_sub_folder: str = "outputs") -> None:
        """Upload a given experiment folder to neptune."""
        if self.runs:
            destination = experiment_path.relative_to(MODELS_DIRECTORY / self.folder_name)

            for file in self._iter_dir(experiment_path / "eval"):
                self._upload_file(f"{neptune_sub_folder}/{destination}/eval/{file.name}", str(file))
            for file in self._iter_dir(experiment_path / "checkpoint"):
                self._upload_file(
                    (
                        f"{neptune_sub_folder}/{destination}/"
                        "checkpoint/"
                        f"{file.relative_to((experiment_path /'checkpoint'))}"
                    ),
                    str(file),
                )

            self._upload_file(
                f"{neptune_sub_folder}/{destination}/configuration.yml",
                str(experiment_path / "configuration.yml"),
            )
            self._upload_file(
                f"{neptune_sub_folder}/{destination}/features.txt",
                str(experiment_path / "features.txt"),
            )

    def close_neptune(self, time_out: int = 10) -> None:
        """Close Neptune connection."""
        if self.runs:
            time_count = 0
            try:
                self.runs.stop()
                log.info("Stop Neptune")
            except ValueError as v:
                log.info("Still waiting for Neptune...")
                time.sleep(10)
                time_count += 1
                if time_count > time_out:
                    raise Timeout("Timeout for closing neptune connection") from v

    def _iter_dir(self, path: Path, prefix_to_remove: Optional[List[str]] = None) -> List[Path]:
        """Iterate over folder."""
        if not prefix_to_remove:
            prefix_to_remove = []
        if not isinstance(prefix_to_remove, list):
            raise TypeError("prefix_to_remove should be a list")

        return [
            file
            for file in path.rglob("*")
            if (file.suffix not in prefix_to_remove and file.is_file())
        ]

    def _upload_file(self, destination: str, file: str, max_attempts: int = 2) -> None:
        """Upload file to neptune safely."""
        for i in range(max_attempts):
            try:
                self.runs[destination].upload(file)
                succeed = True
                break
            except NeptuneException:
                log.warning("Error in attempt %s to log %s to %s ", i, file, destination)
                time.sleep(2)
                continue
        if not succeed:
            log.warning("Fail to log %s to %s", file, destination)
