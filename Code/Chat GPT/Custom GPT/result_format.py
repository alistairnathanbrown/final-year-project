import pandas as pd


def process_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    string_a = "The project will be successful."
    string_b = "The project will fail."

    # Add the 'Result' column based on 'Cancelled' column
    df['Result'] = df['Cancelled'].apply(lambda x: string_a if x == 0 else string_b)

    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)


# Example usage
input_csv = "String format/train_prompts.csv"  # Replace with your input file path
output_csv = "output.csv"  # Replace with your desired output file path
process_csv(input_csv, output_csv)