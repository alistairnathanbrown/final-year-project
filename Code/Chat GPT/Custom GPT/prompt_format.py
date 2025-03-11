import pandas as pd


def format_project_description(row):
    # Determine if the project is onshore or offshore
    if row['Installation Type'].lower() == 'onshore':
        location_sentence = "The project is onshore"
    else:
        location_sentence = (
            f"The project is {row['Distance to Shore For Offshore (km)']} km from shore at a depth of {row['Ocean Depth For Offshore (m)']} meters"
        )

    # Determine if the project is single or multi-phase
    phase_status = 'multi' if row['Mutiple Phase'] == 1 else 'single'

    # Format the owner field (split by 'and' if multiple owners exist)
    if isinstance(row['Owner'], str):
        owner = ' and '.join(row['Owner'].split(';'))
    else:
        owner = 'Unknown'

    # Construct the description string
    # description = (
    #     f"Evaluate the probability of success for a {row['Capacity (MW)']} MW {row['Installation Type']} wind farm in {row['Country']} "
    #     f"(Inflation {row['Inflation in Project Country (HCPI)']}, Energy Inflation {row['Energy Inflation in Project Country (EPI)']}, "
    #     f"GDP Growth {row['Country GDP Growth Rate']}, Government Debt {row['Government Debt as Percentage of GDP']}%, "
    #     f"Credit Rating {row['Country Credit Rating']}). "
    #     f"It is a {phase_status} phase project in {row['Start year']}, with a cost of {row['Estimated Project Cost (GBP)']} GBP, "
    #     f"owned by {owner}. {location_sentence} The average annual wind speed is {row['Annual Wind Speed (m/s)']} m/s."
    # )

    try:
        start_year = int(row['Start year'])
        description = (
            f"Evaluate the probability of success for a {row['Capacity (MW)']} MW {row['Installation Type']} wind farm in {row['Country']}. "
            f"It is a {phase_status} phase project beginning in {row['Start year']}, with a cost of {row['Estimated Project Cost (GBP)']} GBP "
            f"and is owned by {owner}. {location_sentence}, and the average annual wind speed is {row['Annual Wind Speed (m/s)']:.3f} m/s. "
            f"{row['Country']} has a credit rating of {int(row['Country Credit Rating'])}/21, with a GDP growth of {row['Country GDP Growth Rate']:.3f}% "
            f"and {row['Government Debt as Percentage of GDP']:.3f}% of its GDP in debt. "
            f"The inflation rate is {row['Inflation in Project Country (HCPI)']:.3f} and the energy inflation rate is {row['Energy Inflation in Project Country (EPI)']:.3f}."
        )
    except ValueError:
        description = (
            f"Evaluate the probability of success for a {row['Capacity (MW)']} MW {row['Installation Type']} wind farm in {row['Country']}. "
            f"It is a {phase_status} phase project beginning in an unknwon year, with a cost of {row['Estimated Project Cost (GBP)']} GBP "
            f"and is owned by {owner}. {location_sentence}."
        )


    return description


def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter=',')  # Assuming comma-separated file

    # Strip column names of any extra spaces
    df.columns = df.columns.str.strip()

    # Apply the formatting function to each row
    df['Project Description'] = df.apply(format_project_description, axis=1)

    # Save back to the CSV file
    df.to_csv(file_path, index=False, sep=',')

    print("Processing complete. Updated CSV saved.")


# Example usage
# file_path = "test_set.csv"  # Replace with your actual CSV file path
file_path = "train_set.csv"  # Replace with your actual CSV file path
process_csv(file_path)
