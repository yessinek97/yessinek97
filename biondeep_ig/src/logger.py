"""Module used to define Neptune class and logger funcations."""
import logging
import shutil
import sys
import time

import neptune.new as neptune
from neptune.api_exceptions import ServerError
from requests.exceptions import HTTPError
from requests.exceptions import Timeout

from biondeep_ig import FEATURES_DIRECTORY
from biondeep_ig import MODEL_CONFIGURATION_DIRECTORY
from biondeep_ig import MODELS_DIRECTORY


def init_logger(folder_name=None, file_name=None):
    """Init logging function."""
    log_file_path = None
    if folder_name:
        model_folder_path = MODELS_DIRECTORY / folder_name
        if not model_folder_path.exists():
            model_folder_path.mkdir(exist_ok=True, parents=True)
        log_file_name = f"{file_name}.log" if file_name else "InfoRun.log"
        log_file_path = model_folder_path / log_file_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)-12s: %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=log_file_path,
    )
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


def get_logger(name):
    """Get logger by name."""
    logger = logging.getLogger(name)

    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    return logger


def move_log_file(folder_name):
    """Move the log file to the Experiment Folder."""
    shutil.move(MODELS_DIRECTORY / "InfoRun.log", MODELS_DIRECTORY / folder_name / "InfoRun.log")


log = get_logger("Neptune")


class ModelLogWriter:
    """Log Model Output to a sperate log file."""

    def __init__(self, logger_path):
        """Init ModelLogWriter."""
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        self.logger = logging.getLogger("Model/outpus")
        self.logger.propagate = False

        file_handler = logging.FileHandler(logger_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        sys.stdout = self

    def write(self, buf):
        """Write Method."""
        for line in buf.rstrip().splitlines():
            self.logger.info(line.rstrip())

    def flush(self):
        """Flush Method."""

    def reset_stdout(self, orginal_stdout):
        """Reset stdout to system."""
        sys.stdout = orginal_stdout


class NeptuneLogs:
    """Neptune Class."""

    def __init__(self, configuration, folder_name):
        """Init function for Neptune class."""
        self.configuration = configuration
        self.folder_name = folder_name
        self.runs = None

    @property
    def use_neptune(self):
        """Determine whether we use neptune."""
        return self.configuration.get("neptune_logs", False)

    def init_neptune(self, tags, training_path, test_path):
        """Init neptune instance."""
        if tags is None:
            tags = []
        if self.use_neptune:
            self.runs = neptune.init(
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

    def upload_configuration_files(self, configuration_path):
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

    def upload_experiment(self, experiment_path, neptune_sub_folder="outputs"):
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

    def close_neptune(self, time_out=10):
        """Close Neptune connection."""
        if self.runs:
            time_count = 0
            try:
                self.runs.stop()
                log.info("Stop Neptune")
            except ValueError:
                log.info("Still waiting for Neptune...")
                time.sleep(10)
                time_count += 1
                if time_count > time_out:
                    raise Timeout("Timeout for closing neptune connection")

    def _iter_dir(self, path, prefix_to_remove=None):
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

    def _upload_file(self, destination, file, max_attempts=2):
        """Upload file to neptune safely."""
        for i in range(max_attempts):
            try:
                self.runs[destination].upload(file)
                succeed = True
                break
            except (ServerError, HTTPError):
                log.warning(f"Error in attempt {i} to log '{file}' to '{destination}'")
                time.sleep(2)
                continue
        if not succeed:
            log.warning(f"Fail to log '{file}' to '{destination}'")
