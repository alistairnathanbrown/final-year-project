import pandas as pd
import json

# Define the system content
SYSTEM_CONTENT = "Evaluate the probability of success for the wind farm and classify it as successful or failed."

# Load the CSV file
df = pd.read_csv("String format/train_prompts.csv")

# Define the output JSONL file
output_file = "FT_prompts.jsonl"

# Open file and write JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        data = {
            "messages": [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": row["Prompt"]},
                {"role": "assistant", "content": row["Result"]},
            ]
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Formatted data saved to {output_file}")