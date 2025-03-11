import requests
import pandas as pd
import time
import os

# File paths
input_file = "Europe Data Set/RDS3CSV.csv"
output_file = "outputRDS3.csv"

# Read the input CSV file
df = pd.read_csv(input_file)

# Ensure proper data types for the relevant columns
df["Latitude"] = pd.to_numeric(df["Latitude"], errors='coerce')
df["Longitude"] = pd.to_numeric(df["Longitude"], errors='coerce')
df["Start year"] = pd.to_numeric(df["Start year"], errors='coerce')

# Filter out rows with NaN in the "Start year" column
df = df.dropna(subset=["Start year"])

# Check the output file to determine where to resume
processed_ids = set()
if os.path.exists(output_file):
    # Read the output file and collect processed IDs
    output_df = pd.read_csv(output_file)
    processed_ids = set(output_df["Entry ID"].unique())
    print(f"Resuming from {len(processed_ids)} processed entries.")

# Base URL for the NASA POWER API
base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Cache for storing previous API results and counter for cache clearing
cache = {}
request_count = 0
clear_cache_threshold = 100  # Clear cache after every 100 requests

# Batch processing parameters
batch_size = 50

# Prepare for batch saving
output_batch = []

# Counter
counter = 1

# Loop through the DataFrame in batches
for batch_start in range(0, len(df), batch_size):
    batch_end = min(batch_start + batch_size, len(df))
    batch = df.iloc[batch_start:batch_end]  # Select a batch of rows

    # Process each row in the current batch
    for index, row in batch.iterrows():
        entry_id = row["Entry ID"]
        if entry_id in processed_ids:
            print(f"Skipping already processed entry {entry_id}.")
            continue

        latitude = row["Latitude"]
        longitude = row["Longitude"]
        start_year = int(row["Start year"])
        location_key = (latitude, longitude, start_year)

        # Check if result is in cache
        if location_key in cache:
            print(f"Using cached data for entry {entry_id} with {latitude}, {longitude}, Year: {start_year}")
            average_wind_speed = cache[location_key]
        else:
            print(f"Processing entry {entry_id} with {latitude}, {longitude}, Year: {start_year}")

            # Determine the years to request
            if start_year == '-':
                years_to_check = list(range(2000, 2023))
            elif start_year > 2022:
                years_to_check = [2022]  # Use only 2022 if the year is beyond the API's range
            else:
                years_to_check = [year for year in range(start_year - 1, start_year + 2) if year <= 2022]

            # Initialize wind speed list for averaging
            wind_speeds = []

            for year in years_to_check:
                # Define the parameters for the API request
                params = {
                    "start": year,
                    "end": year,
                    "latitude": latitude,
                    "longitude": longitude,
                    "community": "ag",
                    "parameters": "WS50M",
                    "format": "json",
                    "header": "true"
                }

                # Make the API request
                response = requests.get(base_url, params=params)
                request_count += 1  # Increment request count

                if response.status_code == 200:
                    data = response.json()
                    wind_data = data["properties"]["parameter"]["WS50M"]
                    annual_key = f"{year}13"
                    wind_speed = wind_data.get(annual_key, None)

                    if wind_speed is not None:
                        wind_speeds.append(wind_speed)
                else:
                    print(f"Failed to retrieve data for {latitude}, {longitude} in {year}. Status code: {response.status_code}")

                # Pause to avoid hitting API rate limits
                time.sleep(1)

            # Calculate the average wind speed
            if wind_speeds:
                average_wind_speed = sum(wind_speeds) / len(wind_speeds)
                cache[location_key] = average_wind_speed  # Store in cache
                print(f"Average Wind Speed: {average_wind_speed}")
            else:
                print(f"No valid wind speed data for {latitude}, {longitude} around {start_year}")
                average_wind_speed = None

        # Append the processed row to the output batch
        output_batch.append({
            "Entry ID": entry_id,
            "Longitude": longitude,
            "Latitude": latitude,
            "Start year": start_year,
            "Average Wind Speed": average_wind_speed
        })

        counter += 1

        # Check if cache should be cleared
        if request_count >= clear_cache_threshold:
            print("Clearing cache to prevent memory overflow.")
            cache.clear()
            request_count = 0  # Reset the request count

    # Save the current batch to the output CSV
    if output_batch:
        save_mode = 'a' if os.path.exists(output_file) else 'w'
        output_df = pd.DataFrame(output_batch)
        output_df.to_csv(output_file, mode=save_mode, index=False, header=(save_mode == 'w'))
        print(f"Saved batch {batch_start} to {batch_end} to {output_file}")
        output_batch = []  # Clear the batch after saving

print(f"All data saved to {output_file}")