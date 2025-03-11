import requests
import pandas as pd
import time

# List of triples containing latitude, longitude, and start year
coordinates = [
    {"lat": 28.4624, "lon": -0.0576, "start_year": 2014},  # Example coordinates with start year
    {"lat": 34.0522, "lon": -118.2437, "start_year": 2011},  # Los Angeles, 2011
    {"lat": 48.8566, "lon": 2.3522, "start_year": 2012}  # Paris, 2012
]

# Base URL for the NASA POWER API
base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Container to store all results
data_list = []

# Loop through each coordinate and year set to make API requests
for coord in coordinates:
    # Define the parameters for the API request
    params = {
        "start": coord["start_year"],
        "end": coord["start_year"] + 1,  # For a one-year period
        "latitude": coord["lat"],
        "longitude": coord["lon"],
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
        annual_key = f"{coord['start_year']}13"
        wind_data = data["properties"]["parameter"]["WS50M"]
        annual_wind_speed = wind_data.get(annual_key, None)

        if annual_wind_speed is not None:
            # Store the annual wind speed with coordinates and year
            data_list.append({
                "latitude": coord["lat"],
                "longitude": coord["lon"],
                "year": coord["start_year"],
                "annual_wind_speed": annual_wind_speed
            })
        else:
            print(f"No annual wind speed for {coord['lat']}, {coord['lon']} in {coord['start_year']}")
    else:
        print(f"Failed to retrieve data for {coord}. Status code: {response.status_code}")

    # Pause to avoid hitting API rate limits
    time.sleep(1)

# Convert data to a pandas DataFrame
df = pd.DataFrame(data_list)
print(df)
