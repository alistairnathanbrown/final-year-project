import os
import pandas as pd
from datetime import datetime

# Paths
input_folder = "countries"  # Change this to your folder path
credit_scale_file = "Credit Scale.csv"  # Change this to your file path
output_file = "output.csv"

# Load credit scale
credit_scale_df = pd.read_csv(credit_scale_file)
credit_scale = {}
for agency in ['Moody\'s', 'S&P', 'DBRS']:
    credit_scale[agency] = dict(zip(credit_scale_df[agency], credit_scale_df['Scale']))


# Define the output structure
years = list(range(1970, 2025))
out_df = pd.DataFrame(columns=['Country'] + years)

# Process each country's file
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        country = file.replace(".csv", "")
        file_path = os.path.join(input_folder, file)

        # Load country data
        try:
            df = pd.read_csv(file_path, dayfirst=True)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Ensure proper date conversion
            print(f"Successfully loaded {file}")
            print(df.head())  # Debugging: Print first few rows to check data
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        df = df.dropna(subset=['Date'])  # Drop rows where date conversion failed
        df = df.sort_values(by='Date', ascending=True)  # Process from earliest to latest


        # Convert ratings to standard scale
        def convert_rating(row):
            agency = row['Agency']
            rating = row['Rating']
            scale = credit_scale.get(agency, {}).get(rating, None)
            if scale is None:
                print(f"Warning: No scale found for Rating: {rating}, Agency: {agency}")
            return scale


        df['Scale'] = df.apply(convert_rating, axis=1)
        print("Converted ratings:")
        print(df[['Rating', 'Agency', 'Scale']].head())  # Debugging: Check rating conversions
        df.dropna(subset=['Scale'], inplace=True)

        # Process ratings in chronological order
        year_ratings = {}
        last_value = None
        for _, row in df.iterrows():
            if pd.notna(row['Date']):
                year = row['Date'].year
                last_value = row['Scale']  # Update last known value
                year_ratings[year] = last_value

        # Fill in missing years by propagating values forward
        country_ratings = {}
        last_value = None
        for year in years:
            if year in year_ratings:
                last_value = year_ratings[year]
            country_ratings[year] = last_value

        # Append to output DataFrame
        new_row = pd.DataFrame([[country] + [country_ratings.get(y, '') for y in years]], columns=out_df.columns)
        if not new_row.isna().all(axis=None):  # Ensure new_row is not entirely empty
            out_df = pd.concat([out_df, new_row], ignore_index=True)

        print(f"Finished processing {file}")

# Save results
out_df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")