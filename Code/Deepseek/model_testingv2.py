import ollama
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import os
import re

# Configuration
DEEPWIND_VERSION = "3a"
SYSTEM_PROMPT = "Classify the following wind farm project into success or failure. First, assess the potential project risks and strengths associated with the wind farm. Then, evaluate both the physical and economic factors and compare these to similar projects. Finally, conclude on if the project is more likey to succeed or fail. Give this response as a single word, 'Success' or 'Fail'."  # Modify this as needed
SUCCESS_RESPONSE = "Success"
FAIL_RESPONSE = "Fail"
CSV_FILE_PATH = f"String format/test_prompts.csv"
JSONL_OUTPUT_PATH = f"Model Testing/Response Files/FT_results_Deepwind_v{DEEPWIND_VERSION}.jsonl"
CLASSIFICATION_REPORT_PATH = f"Model Testing/Result Files/classification_report_v{DEEPWIND_VERSION}.csv"
TIMEOUT_SECONDS = 500

# Load existing processed entries to allow resumption
processed_entries = set()
if os.path.exists(JSONL_OUTPUT_PATH):
    with open(JSONL_OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                processed_entries.add(entry["Entry ID"])
            except json.JSONDecodeError:
                continue

# Load CSV datar
df = pd.read_csv(CSV_FILE_PATH)

# Function to query the model
def query_ollama(user_prompt):
    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": user_prompt, "role": "user"},
    ]
    try:
        response = ollama.chat(
            model=f"deepwind-v{DEEPWIND_VERSION}",
            messages=messages,
            stream=False
        )['message']['content']
        return response
    except Exception as e:
        return None

# Function to extract response type using </think> tag
def extract_response_type(model_response):
    if not model_response:
        return "Unclassified"

    # Look for the last occurrence of </think>
    match = re.search(r"</think>\s*(.*)", model_response, re.IGNORECASE | re.DOTALL)

    if match:
        response_text = match.group(1).strip().lower()  # Extract text after </think>
        if "success" in response_text:
            return SUCCESS_RESPONSE
        elif "fail" in response_text or "failure" in response_text:
            return FAIL_RESPONSE

    return

# Step 1: Collect model responses and save them to JSONL
with tqdm(total=len(df), desc="Processing entries", unit="entry") as pbar:
    for _, row in df.iterrows():
        entry_id = row["Entry ID"]
        if entry_id in processed_entries:
            pbar.update(1)
            continue  # Skip already processed entries

        user_prompt = row["Prompt"]

        # Use ThreadPoolExecutor to enforce a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(query_ollama,  user_prompt)
            try:
                model_response = future.result(timeout=TIMEOUT_SECONDS)  # Timeout set to 5 minutes
            except concurrent.futures.TimeoutError:
                model_response = None  # Mark as unclassified due to timeout

        # Save response to JSONL file
        with open(JSONL_OUTPUT_PATH, 'a', encoding='utf-8') as f:
            json.dump({"Entry ID": entry_id, "Model Response": model_response}, f)
            f.write("\n")

        pbar.update(1)

# Step 2: Compute classification metrics by analyzing the JSONL file
success_tp = success_fp = success_fn = 0
failure_tp = failure_fp = failure_fn = 0
unclassified_count = 0

# Read responses from JSONL and compare with expected results
with open(JSONL_OUTPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line.strip())
            entry_id = entry["Entry ID"]
            model_response = entry["Model Response"]

            # Fetch expected response from CSV
            expected_response = df.loc[df["Entry ID"] == entry_id, "Result Short"].values[0]

            # Extract response type using </think> logic
            response_type = extract_response_type(model_response)

            # Update classification counts
            if response_type == "Unclassified":
                unclassified_count += 1
            elif response_type == SUCCESS_RESPONSE:
                if expected_response == SUCCESS_RESPONSE:
                    success_tp += 1
                else:
                    success_fp += 1
            elif response_type == FAIL_RESPONSE:
                if expected_response == FAIL_RESPONSE:
                    failure_tp += 1
                else:
                    failure_fp += 1

            if expected_response == SUCCESS_RESPONSE and response_type != expected_response:
                success_fn += 1
            if expected_response == FAIL_RESPONSE and response_type != expected_response:
                failure_fn += 1

        except (json.JSONDecodeError, KeyError, IndexError):
            continue  # Skip faulty lines

# Compute precision, recall, and F1 score
def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    support = tp + fn  # Total actual instances of this class
    return precision, recall, f1, support

success_precision, success_recall, success_f1, success_support = compute_f1(success_tp, success_fp, success_fn)
failure_precision, failure_recall, failure_f1, failure_support = compute_f1(failure_tp, failure_fp, failure_fn)

# Compute macro & weighted averages
total_support = success_support + failure_support
macro_precision = (success_precision + failure_precision) / 2
macro_recall = (success_recall + failure_recall) / 2
macro_f1 = (success_f1 + failure_f1) / 2

weighted_precision = (success_precision * success_support + failure_precision * failure_support) / total_support if total_support > 0 else 0
weighted_recall = (success_recall * success_support + failure_recall * failure_support) / total_support if total_support > 0 else 0
weighted_f1 = (success_f1 * success_support + failure_f1 * failure_support) / total_support if total_support > 0 else 0

overall_accuracy = (success_tp + failure_tp) / len(df) if len(df) > 0 else 0

# Create classification report DataFrame
classification_report = pd.DataFrame({
    "precision": [success_precision, failure_precision, "", macro_precision, weighted_precision],
    "recall": [success_recall, failure_recall, "", macro_recall, weighted_recall],
    "f1-score": [success_f1, failure_f1, overall_accuracy, macro_f1, weighted_f1],
    "support": [success_support, failure_support, total_support, total_support, total_support]
}, index=["Success", "Failure", "Accuracy", "Macro Avg", "Weighted Avg"])

# Print results in table format
tqdm.write("\nClassification Report:\n")
print(classification_report)

# Save results to CSV
classification_report.to_csv(CLASSIFICATION_REPORT_PATH)
tqdm.write(f"\nResults saved to {CLASSIFICATION_REPORT_PATH}")