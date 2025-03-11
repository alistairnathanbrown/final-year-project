import pandas as pd
import json

WINDLLAMA_VERSION = "5b2"
TEST_TRAIN = "test"
RESULT_LENGTH = "Result Short"

# Define the system content
SYSTEM_CONTENT= {
    "1": "Evaluate the probability of success for the wind farm and classify it as successful or failed.",
    "2": "Evaluate the probability of success for the wind farm and classify it as successful or failed.",
    "3": "Evaluate the probability of success for the wind farm and classify it as successful or failed.",
    "4a": "Classify the wind farm project as successful or failed.",
    "4b": "Classify the following wind farm project into success or failure.",
    "4c": "Classify the following wind farm project into success or failure.",
    "5a": "Classify the following wind farm project into success or failure. First, assess the potential project risks associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project will succeed or fail.",
    "5b1": "Analyse the characteristics of the following wind farm project. Identify relevant technical and economic factors based on historical trends. Then, compare these to similar projects. Finally, determine whether the project characteristics align more closely with successful or unsuccessful projects.",
    "5b2": "Classify the following wind farm project based on technical and economic indicators. Compare it to historical wind farm data and determine which category (successful or fail) it most closely resembles.",
}

# Load the CSV file
# Define the output JSONL file
if TEST_TRAIN == "train":
    df = pd.read_csv("String format/train_prompts.csv")
    output_file = f"Training Prompts/FT_prompts_Ollama_v{WINDLLAMA_VERSION}.jsonl"
elif TEST_TRAIN == "test":
    df = pd.read_csv("String format/test_prompts.csv")
    output_file = f"Model Testing/Test Prompts/FT_prompts_Ollama_test_prompts_v{WINDLLAMA_VERSION}.jsonl"


with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        data = {
            "conversations": [
                {"content": SYSTEM_CONTENT[WINDLLAMA_VERSION], "role": "system"},
                {"content": row["Prompt"], "role": "user"},
                {"content": row[RESULT_LENGTH], "role": "assistant"},
            ]
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Formatted data saved to {output_file}")