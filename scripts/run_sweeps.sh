
# C4 ablations
pdm run python circuit_finder/experiments/run_leap_experiment_batched.py --contrast_pairs False --ablate_act_type unstructured
# with error ablations
pdm run python circuit_finder/experiments/run_leap_experiment_batched.py --contrast_pairs False --ablate_act_type unstructured --error_ablate_type value 

# Dataset ablations
pdm run python circuit_finder/experiments/run_leap_experiment_batched.py --contrast_pairs True --ablate_act_type structured
# without error ablations
pdm run python circuit_finder/experiments/run_leap_experiment_batched.py --contrast_pairs True --ablate_act_type structured --error_ablate_type value