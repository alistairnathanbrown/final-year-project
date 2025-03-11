import pandas as pd
import json

GPT_FT_VERSION = "1"
TEST_TRAIN = "test"
RESULT_LENGTH = "Result Short"

# Define the system content
SYSTEM_CONTENT= {
    "1": "Classify the following wind farm project into success or failure.",
    "2?": "Classify the following wind farm project into success or failure. First, assess the potential project risks associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project will succeed or fail.",
    "3?": "Analyse the characteristics of the following wind farm project. Identify relevant technical and economic factors based on historical trends. Then, compare these to similar projects. Finally, determine whether the project characteristics align more closely with successful or unsuccessful projects.",
    "4?": "Classify the following wind farm project based on technical and economic indicators. Compare it to historical wind farm data and determine which category (successful or fail) it most closely resembles.",
}

# Load the CSV file
# Define the output JSONL file
if TEST_TRAIN == "train":
    df = pd.read_csv("String format/train_prompts.csv")
    output_file = f"Training Prompts/FT_prompts_FT_GPT_train_prompts_v{GPT_FT_VERSION}.jsonl"
elif TEST_TRAIN == "test":
    df = pd.read_csv("String format/test_prompts.csv")
    output_file = f"Testing Prompts/FT_prompts_FT_GPT_test_prompts_v{GPT_FT_VERSION}.jsonl"


with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        data = {
            "messages": [
                {"role": "system", "content": SYSTEM_CONTENT[GPT_FT_VERSION]},
                {"role": "user", "content":  row["Prompt"]},
                {"role": "assistant", "content": row[RESULT_LENGTH]},
            ]
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Formatted data saved to {output_file}")