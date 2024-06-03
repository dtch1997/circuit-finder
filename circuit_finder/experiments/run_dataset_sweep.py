from circuit_finder.data.ioi import ABBA_DATASETS, BABA_DATASETS

IOI_DATASETS = ABBA_DATASETS[:2] + BABA_DATASETS[:2]

ANIMAL_DIET_DATASETS = [
    "datasets/animal_diet_long_prompts.json",
    "datasets/animal_diet_short_prompts.json",  # NOTE: has a bug
]

DOCSTRING_DATASETS = [
    "datasets/docstring_prompts.json",
]

GREATERTHAN_DATASETS = [
    "datasets/greaterthan_gpt2-small_prompts.json",
]

CAPITAL_CITIES_DATASETS = [
    "datasets/capital_cities_pythia-70m-deduped_prompts.json",
]

SPORTS_PLAYERS_DATASETS = [
    "datasets/sports-players/sports_players_pythia-410m-deduped_prompts.json",
]

SUBJECT_VERB_AGREEMENT_DATASETS = [
    "datasets/subject_verb_agreement.json",
]

GENDER_BIAS_DATASETS = [
    "datasets/gender_bias.json",
]

ALL_DATASETS = [
    *ANIMAL_DIET_DATASETS,
    *DOCSTRING_DATASETS,
    # *CAPITAL_CITIES_DATASETS,
    # *SPORTS_PLAYERS_DATASETS,
    *GENDER_BIAS_DATASETS,
    *GREATERTHAN_DATASETS,
    *SUBJECT_VERB_AGREEMENT_DATASETS,
    ABBA_DATASETS[0],
    BABA_DATASETS[0],
    ABBA_DATASETS[2],
    BABA_DATASETS[2],
]

# Selected based on two criteria
# - Whether GPT-2 small can reliably pick the correct answer
# - Whether the logit diff is significant
SELECTED_DATASETS = [
    *DOCSTRING_DATASETS,
    *GENDER_BIAS_DATASETS,
    # *GREATERTHAN_DATASETS,
    *SUBJECT_VERB_AGREEMENT_DATASETS,
    # ABBA_DATASETS[0],
    # BABA_DATASETS[0],
    ABBA_DATASETS[2],
    ABBA_DATASETS[3],
    BABA_DATASETS[2],
    BABA_DATASETS[3],
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
