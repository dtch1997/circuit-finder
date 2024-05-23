import simple_parsing
import pathlib

from dataclasses import replace
from circuit_finder.experiments.run_leap_experiment import (
    LeapExperimentConfig,
    run_leap_experiment,
)

from circuit_finder.constants import ProjectDir
from circuit_finder.data.ioi import ABBA_DATASETS, BABA_DATASETS

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    args = parser.parse_args()

    for dataset_path in ABBA_DATASETS + BABA_DATASETS:
        save_dir = str(ProjectDir / "results" / pathlib.Path(dataset_path).stem)
        config = replace(args.config, dataset_path=dataset_path, save_dir=save_dir)
        print(config)
        run_leap_experiment(config)
