import requests
import pandas as pd
import time

# Load data from CSV file
input_file = "Test Data Set 2.csv"  # Replace with the path to your input CSV file
output_file = "output.csv"  # Replace with the path for the output CSV file

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

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    latitude = row["Latitude"]
    longitude = row["Longitude"]
    start_year = int(row["Start year"])
    location_key = (latitude, longitude, start_year)

    # Check if result is in cache
    if location_key in cache:
        print(f"Using cached data for {latitude}, {longitude}, Year: {start_year}")
        df.at[index, "Annual Wind Speed"] = cache[location_key]
        continue

    print(f"Processing {latitude}, {longitude}, Year: {start_year}")

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

    # Check if cache should be cleared
    if request_count >= clear_cache_threshold:
        print("Clearing cache to prevent memory overflow.")
        cache.clear()
        request_count = 0  # Reset the request count

    # Pause to avoid hitting API rate limits
    time.sleep(1)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
