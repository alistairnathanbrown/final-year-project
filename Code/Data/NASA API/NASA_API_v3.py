import requests
import pandas as pd
import time

# Load data from CSV file
input_file = "Test Data Set 2.csv"  # Replace with the path to your input CSV file
output_file = "output.csv"  # Replace with the path for the output CSV file

# Read the input CSV file
df = pd.read_csv(input_file)

# Ensure proper data types for the relevant columns
df["Latitude"] = pd.to_numeric(df["Latitude"], errors='coerce')  # Convert latitude to numeric (float)
df["Longitude"] = pd.to_numeric(df["Longitude"], errors='coerce')  # Convert longitude to numeric (float)
df["Start year"] = pd.to_numeric(df["Start year"], errors='coerce')  # Convert start year to numeric (int)

# Filter out rows with NaN in the "Start year" column
df = df.dropna(subset=["Start year"])

# Base URL for the NASA POWER API
base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Add a new column for the annual wind speed, initially set to None
df["Annual Wind Speed"] = None

# Cache for storing previous API results
cache = {}

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    latitude = row["Latitude"]
    longitude = row["Longitude"]
    start_year = int(row["Start year"])  # Ensure it's an integer
    location_key = (latitude, longitude, start_year)  # Key for caching

    # Check if the result is already in the cache
    if location_key in cache:
        print(f"Using cached data for {latitude}, {longitude}, Year: {start_year}")
        df.at[index, "Annual Wind Speed"] = cache[location_key]
        continue

    print(f"Processing {latitude}, {longitude}, Year: {start_year}")

    # Define the parameters for the API request
    params = {
        "start": start_year,
        "end": start_year + 1,  # For a one-year period
        "latitude": latitude,
        "longitude": longitude,
        "community": "ag",
        "parameters": "WS50M",  # 50-meter wind speed
        "format": "json",
        "header": "true"
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Access the annual wind speed value with the key "YYYY13" (e.g., "201013")
        annual_key = f"{start_year}13"
        wind_data = data["properties"]["parameter"]["WS50M"]
        annual_wind_speed = wind_data.get(annual_key, None)

        # Update the DataFrame and cache with the retrieved annual wind speed
        if annual_wind_speed is not None:
            df.at[index, "Annual Wind Speed"] = annual_wind_speed
            print(f"Wind speed: {annual_wind_speed}")
            cache[location_key] = annual_wind_speed  # Store result in cache
        else:
            print(f"No annual wind speed for {latitude}, {longitude} in {start_year}")
    else:
        print(f"Failed to retrieve data for {latitude}, {longitude}. Status code: {response.status_code}")

    # Pause to avoid hitting API rate limits
    time.sleep(1)

# Save the updated DataFrame, which includes all original columns plus the new "Annual Wind Speed" column, to a new CSV file
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
