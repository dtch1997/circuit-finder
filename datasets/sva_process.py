"""Process the subject-verb agreement dataset"""

import sys
import json

sys.path.append("/workspace/circuit-finder")
from circuit_finder.constants import ProjectDir

if __name__ == "__main__":
    prompts = []
    with open(ProjectDir / "datasets" / "subject_verb_agreement/nounpp_train.txt") as f:
        lines = f.readlines()
        for line in lines:
            raw_prompt_dict = json.loads(line)

            prompt = {
                "clean": raw_prompt_dict["clean_prefix"],
                "corrupt": raw_prompt_dict["patch_prefix"],
                "answers": [raw_prompt_dict["clean_answer"]],
                "wrong_answers": [raw_prompt_dict["patch_answer"]],
            }
            prompts.append(prompt)

    with open(ProjectDir / "datasets" / "subject_verb_agreement.json", "w") as f:
        json.dump({"prompts": prompts}, f)
