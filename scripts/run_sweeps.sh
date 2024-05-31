# Corrupt ablations
# Here we ablate to the corrupt activations. 
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix corrupt_keep_error --contrast_pairs True --ablate_act_type corrupt
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix corrupt_ablate_error --contrast_pairs True --ablate_act_type corrupt --error_ablate_type value

# Tokenwise ablations
# Here we ablate to the mean of the clean activations over a reference dataset. 
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix tokenwise_keep_error --contrast_pairs True --ablate_act_type tokenwise
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix tokenwise_ablate_error --contrast_pairs True --ablate_act_type tokenwise --error_ablate_type value

# Dataset ablations
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix D_keep_error --contrast_pairs True --ablate_act_type structured
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix D_ablate_error --contrast_pairs True --ablate_act_type structured --error_ablate_type value

# C4 ablations
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix c4_keep_error  --contrast_pairs False --ablate_act_type unstructured
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix c4_ablate_error --contrast_pairs False --ablate_act_type unstructured --error_ablate_type value 


# Clean ablations
# Here we ablate to the clean activations, meaned over the batch. 
# It doesn't make sense to use batch_size == 1 here. 
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix clean_keep_error_bs8 --contrast_pairs True --ablate_act_type clean --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix clean_ablate_error_bs8 --contrast_pairs True --ablate_act_type clean --error_ablate_type value --batch_size 8
# As above but with batch_size = 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix corrupt_keep_error_bs8 --contrast_pairs True --ablate_act_type corrupt --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix corrupt_ablate_error_bs8 --contrast_pairs True --ablate_act_type corrupt --error_ablate_type value --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix tokenwise_keep_error_bs8 --contrast_pairs True --ablate_act_type tokenwise --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix tokenwise_ablate_error_bs8 --contrast_pairs True --ablate_act_type tokenwise --error_ablate_type value --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix c4_keep_error_bs8  --contrast_pairs False --ablate_act_type unstructured --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix c4_ablate_error_bs8 --contrast_pairs False --ablate_act_type unstructured --error_ablate_type value --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix D_keep_error_bs8 --contrast_pairs True --ablate_act_type structured --batch_size 8
pdm run python circuit_finder/paper/run_leap_experiment_batched.py --save_dir_prefix D_ablate_error_bs8 --contrast_pairs True --ablate_act_type structured --error_ablate_type value --batch_size 8
