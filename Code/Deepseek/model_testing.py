import ollama
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm

DEEPWIND_VERSION = "3a"
SUCCESS_RESPONSE = "Success"
FAIL_RESPONSE = "Fail"
file_path = f"Testing Prompts/FT_prompts_Deepwind_test_prompts_v{DEEPWIND_VERSION}.jsonl"

def load_json_lines(file_path):
    data = []
    assistant_responses = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())

                assistant_texts = [conv["content"] for conv in entry.get("conversations", []) if
                                   conv.get("role") == "assistant"]
                assistant_responses.extend(assistant_texts)

                entry["conversations"] = [conv for conv in entry.get("conversations", []) if
                                          conv.get("role") != "assistant"]

                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data, assistant_responses

data, assistant_responses = load_json_lines(file_path)

success_tp = 0  # True Positives (Correct Success)
success_fp = 0  # False Positives (Predicted Success, but incorrect)
success_fn = 0  # False Negatives (Should be Success but wasn't)

failure_tp = 0  # True Positives (Correct Failure)
failure_fp = 0  # False Positives (Predicted Failure, but incorrect)
failure_fn = 0  # False Negatives (Should be Failure but wasn't)

unclassified_count = 0

index = 0
total = len(data)

def query_ollama(entry):
    try:
        response = ollama.chat(
            model=f"deepwind-v{DEEPWIND_VERSION}",
            messages=entry['conversations'],
            stream=False
        )['message']['content']
        return response
    except Exception as e:
        return None

with tqdm(total=total, desc="Processing entries", unit="entry") as pbar:
    for entry in range(total):

        expected_response = assistant_responses[index]

        # Use a ThreadPoolExecutor to enforce a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(query_ollama, data[entry])
            try:
                response = future.result(timeout=60)  # Timeout set to 60 seconds
            except concurrent.futures.TimeoutError:
                response = None  # Mark as unclassified due to timeout

        tqdm.write(response)

        if response is None:
            unclassified_count += 1
            tqdm.write("Unclassified Response")

        elif response == SUCCESS_RESPONSE:
            if response == expected_response:
                success_tp += 1  # Correctly predicted success
                # tqdm.write("Correct Success")
            else:
                success_fp += 1  # Wrongly predicted success
                tqdm.write("Incorrect Success")

        elif response == FAIL_RESPONSE:
            if response == expected_response:
                failure_tp += 1  # Correctly predicted failure
                # tqdm.write("Correct Failure")
            else:
                failure_fp += 1  # Wrongly predicted failure
                # tqdm.write("Incorrect Failure")

        else:
            unclassified_count += 1
            tqdm.write("Unclassified Response")

        if expected_response == SUCCESS_RESPONSE and response != expected_response:
            success_fn += 1
        if expected_response == FAIL_RESPONSE and response != expected_response:
            failure_fn += 1

        index += 1
        pbar.update(1)


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

overall_accuracy = (success_tp + failure_tp) / total if total > 0 else 0

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
csv_file = f"Model Testing/Result Files/classification_report_v{DEEPWIND_VERSION}.csv"
classification_report.to_csv(csv_file)

tqdm.write(f"\nResults saved to {csv_file}")