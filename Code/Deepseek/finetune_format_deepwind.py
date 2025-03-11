import pandas as pd
import json

DEEPWIND_VERSION = "3a"
TEST_TRAIN = "train"
RESULT_LENGTH = "Result Short"

# Define the system content
SYSTEM_CONTENT= {
    "1": "Classify the following wind farm project into success or failure.",
    "2": "Classify the following wind farm project into success or failure.",
    "3a": "Classify the following wind farm project into success or failure. First, assess the potential project risks and strengths associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project is more likey to succeed or fail. Give this response as a single word, 'Success' or 'Fail'.",
    "CoT": "Classify the following wind farm project into success or failure. First, assess the potential project risks associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project will succeed or fail.",
}

# Load the CSV file
# Define the output JSONL file
if TEST_TRAIN == "train":
    df = pd.read_csv("String format/train_prompts.csv")
    output_file = f"Training Prompts/FT_prompts_Deepwind_train_prompts_v{DEEPWIND_VERSION}.jsonl"
elif TEST_TRAIN == "test":
    df = pd.read_csv("String format/test_prompts.csv")
    output_file = f"Testing Prompts/FT_prompts_Deepwind_test_prompts_v{DEEPWIND_VERSION}.jsonl"


with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        data = {
            "conversations": [
                {"content": SYSTEM_CONTENT[DEEPWIND_VERSION], "role": "system"},
                {"content": row["Prompt"], "role": "user"},
                {"content": row[RESULT_LENGTH], "role": "assistant"},
            ]
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Formatted data saved to {output_file}")