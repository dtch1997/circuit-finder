# type: ignore
from typing import Sequence

prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    # "Then, Jeremy and Scott were working at the hospital.{} decided to give a snack to",
    # "When Tom and James went to the park,{} gave the ball to",
    # "When Dan and Sid went to the shops,{} gave an apple to",
    # "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" John", " Mary"),
    # (" Jeremy", " Scott"),
    # (" Tom", " James"),
    # (" Dan", " Sid"),
    # (" Martin", " Amy"),
]

corrupt_prompt_for_prompt_format = [
    "When Michael and Anderson went to the shops, Jason gave the bag to",
    # "Then, Michael and Anderson were working at the hospital. Rachel decided to give a snack to",
    # "When Rachel and Laura went to the shops, Sidney gave an apple to",
    # "After Reshel and Lawrence went to the park, Kevin gave a drink to"
]


def get_ioi_data() -> tuple[Sequence[str], Sequence[str], Sequence[tuple[str, str]]]:
    # Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
    clean_prompts = [
        prompt.format(name)
        for (prompt, names) in zip(prompt_format, name_pairs)
        for name in names[::-1]
    ]
    # Define 8 corrupt prompts
    corrupt_prompts = []
    for i in range(len(prompt_format)):
        corrupt_prompts.extend([corrupt_prompt_for_prompt_format[i]] * 2)

    # Define the answers for each prompt, in the form (correct, incorrect)
    answers = [names[::i] for names in name_pairs for i in (1, -1)]
    return clean_prompts, corrupt_prompts, answers
