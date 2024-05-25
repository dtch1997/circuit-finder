import simple_parsing
import pathlib

from dataclasses import replace
from circuit_finder.experiments.run_leap_experiment import (
    LeapExperimentConfig,
    run_leap_experiment,
)

from circuit_finder.constants import ProjectDir
from circuit_finder.data.ioi import ABBA_DATASETS, BABA_DATASETS

IOI_DATASETS = ABBA_DATASETS + BABA_DATASETS

ANIMAL_DIET_DATASETS = [
    "datasets/animal_diet_long_prompts.json",
    "datasets/animal_diet_short_prompts.json",  # NOTE: has a bug
]

DOCSTRING_DATASETS = [
    "datasets/docstring_prompts.json",
]

CAPITAL_CITIES_DATASETS = [
    "datasets/capital_cities_pythia-70m-deduped_prompts.json",
]

SPORTS_PLAYERS_DATASETS = [
    "datasets/sports_players_pythia-410m-deduped_prompts.json",
]

ALL_DATASETS = [
    *ANIMAL_DIET_DATASETS,
    *DOCSTRING_DATASETS,
    *CAPITAL_CITIES_DATASETS,
    *SPORTS_PLAYERS_DATASETS,
    *IOI_DATASETS,
]


def get_datasets(sweep_name: str) -> list[str]:
    """List of dataset paths"""
    if sweep_name == "ioi":
        return ABBA_DATASETS + BABA_DATASETS
    elif sweep_name == "animal_diet":
        return ANIMAL_DIET_DATASETS
    elif sweep_name == "docstring":
        return DOCSTRING_DATASETS
    elif sweep_name == "capital_cities":
        return CAPITAL_CITIES_DATASETS
    elif sweep_name == "sports_players":
        return SPORTS_PLAYERS_DATASETS
    elif sweep_name == "all":
        return ALL_DATASETS
    else:
        raise ValueError(f"Unknown dataset sweep: {sweep_name}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(LeapExperimentConfig, dest="config")
    parser.add_argument("--sweep-name", type=str, default="all")
    args = parser.parse_args()

    for dataset_path in get_datasets(args.sweep_name):
        dataset_name = pathlib.Path(dataset_path).stem
        save_dir = ProjectDir / "results" / dataset_name
        if save_dir.exists():
            print(f"Found existing results for {dataset_name}. Skipping...")
            continue
        config = replace(args.config, dataset_path=dataset_path, save_dir=str(save_dir))
        print(config)
        try:
            run_leap_experiment(config)
        except Exception as e:
            print(e)
            pass
