# type: ignore
"""This module includes unit tests for inference pipeline."""
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from ig.inference import inference


def test_inference(
    models_path: Path,
    test_data_path: str,
    folder_name: str = "training_fixtures",
    id_name: str = "id",
) -> None:
    """This function tests the inference command."""
    runner = CliRunner()
    params = [
        "--test_data_path",
        test_data_path,
        "--folder_name",
        folder_name,
        "--id_name",
        id_name,
    ]
    _ = runner.invoke(inference, params)
    output_file_name = Path(test_data_path).stem
    prediction_path = models_path / folder_name / "Inference"
    output_file_path = prediction_path / f"{output_file_name}.csv"
    output_df = pd.read_csv(output_file_path)
    output_log_file_path = prediction_path / "Inference.log"
    log_file = open(output_log_file_path).readlines()
    assert output_log_file_path.exists(), "Check Inference generated log file !"
    assert log_file != [], "Check Inference generated log file values !"
    assert (output_file_path).exists(), "Check the Inference output file!"
    assert (
        output_df[id_name].all() == pd.read_csv(test_data_path)[id_name].all()
    ), "Check the Inference command provided id_name!"
