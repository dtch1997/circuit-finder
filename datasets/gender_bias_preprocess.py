"""Process the subject-verb agreement dataset"""

import sys
import json
import pandas as pd

sys.path.append("/workspace/circuit-finder")
from circuit_finder.constants import ProjectDir
from circuit_finder.pretrained import load_model

if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Load the dataset
    df = pd.read_csv(ProjectDir / "datasets" / "gender_bias/gpt2.csv")
    df.head()

    prompts = []
    # Process the dataframe
    for i, row in df.iterrows():
        answer = model.to_single_str_token(row["clean_answer_idx"])
        wrong_answer = model.to_single_str_token(row["corrupted_answer_idx"])

        prompt = {
            "clean": row["clean"],
            "corrupt": row["corrupted"],
            "answers": [answer],
            "wrong_answers": [wrong_answer],
        }
        prompts.append(prompt)

    with open(ProjectDir / "datasets" / "gender_bias.json", "w") as f:
        json.dump({"prompts": prompts}, f)
