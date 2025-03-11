import requests
import pandas as pd
import time

# Load data from CSV file
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

# Base URL for the NASA POWER API
base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Add a new column for the annual wind speed, initially set to None
df["Annual Wind Speed"] = None

# Cache for storing previous API results and counter for cache clearing
cache = {}
request_count = 0
clear_cache_threshold = 100  # Clear cache after every 100 requests

# Batch processing parameters
batch_size = 50

# Counter
counter = 1

# Loop through the DataFrame in batches
for batch_start in range(0, len(df), batch_size):
    batch_end = min(batch_start + batch_size, len(df))
    batch = df.iloc[batch_start:batch_end]  # Select a batch of rows

    # Process each row in the current batch
    for index, row in batch.iterrows():
        latitude = row["Latitude"]
        longitude = row["Longitude"]
        start_year = int(row["Start year"])
        location_key = (latitude, longitude, start_year)

        # Check if result is in cache
        if location_key in cache:
            print(f"Using cached data for entry {counter} with {latitude}, {longitude}, Year: {start_year}")
            df.at[index, "Annual Wind Speed"] = cache[location_key]
            counter += 1
            continue

        print(f"Processing entry {counter} with {latitude}, {longitude}, Year: {start_year}")
        counter += 1

        # Define the parameters for the API request
        params = {
            "start": start_year,
            "end": start_year + 1,
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
            annual_key = f"{start_year}13"
            wind_data = data["properties"]["parameter"]["WS50M"]
            annual_wind_speed = wind_data.get(annual_key, None)

            # Update DataFrame and cache
            if annual_wind_speed is not None:
                df.at[index, "Annual Wind Speed"] = annual_wind_speed
                print(f"Wind speed: {annual_wind_speed}")
                cache[location_key] = annual_wind_speed  # Store in cache
            else:
                print(f"No annual wind speed for {latitude}, {longitude} in {start_year}")
        else:
            print(f"Failed to retrieve data for {latitude}, {longitude}. Status code: {response.status_code}")
            df.at[index, "Annual Wind Speed"] = "None"

        # Check if cache should be cleared
        if request_count >= clear_cache_threshold:
            print("Clearing cache to prevent memory overflow.")
            cache.clear()
            request_count = 0  # Reset the request count

        # Pause to avoid hitting API rate limits
        time.sleep(1)

    # After processing each batch, save the updated rows to the output CSV
    print(f"Saving batch {batch_start} to {batch_end} to {output_file}")
    df.iloc[batch_start:batch_end].to_csv(output_file, mode='a', index=False, header=(batch_start == 0))

print(f"All data saved to {output_file}")
