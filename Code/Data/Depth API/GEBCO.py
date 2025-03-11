import requests
import pandas as pd
from datetime import datetime
import time

username = ''
password = ''

input_csv = 'Errors.csv'
output_csv = 'output.csv'

# Define the base URL for the API
base_url = f'https://{username}:{password}@api.meteomatics.com'

current_time = datetime.utcnow().strftime('%Y-%m-%dT%HZ')

# Function to retrieve ocean depth data
def get_ocean_depth(lat, lon, project_name):
    # Construct the API request URL
    url = f"{base_url}/{current_time}/ocean_depth:m/{lat},{lon}/csv"

    response = requests.get(url)

    # Check for successful response
    if response.status_code == 200:
        # Extract the ocean depth value from the response text
        lines = response.text.splitlines()
        depth_value = lines[1].split(';')[1]  # The depth value is the second part after the ';'
        print(f"Ocean depth for lat {lat}, lon {lon}, {project_name}: {depth_value}")
        return float(depth_value)  # Return as a float for numerical processing
    else:
        print(f"Error fetching data for lat {lat}, lon {lon}: {response.status_code}")
        return None

# Read the input CSV
df = pd.read_csv(input_csv)

# Prepare a list to store results
results = []

# Iterate through the rows of the input DataFrame
for idx, row in df.iterrows():
    project_name = row['Project Name']  # Assuming 'Project Name' exists in the input CSV
    lat = row['Latitude']  # Assuming 'Latitude' exists
    lon = row['Longitude']  # Assuming 'Longitude' exists

    # Get the ocean depth
    ocean_depth = get_ocean_depth(lat, lon, project_name)

    # If ocean depth is successfully retrieved, append the result
    if ocean_depth is not None:
        results.append({
            'Project Name': project_name,
            'Latitude': lat,
            'Longitude': lon,
            'Ocean Depth': ocean_depth
        })

    # Delay to respect the API rate limit (50 requests per minute)
    if (idx + 1) % 50 == 0:
        print("Reached 50 requests, waiting for 60 seconds...")
        time.sleep(60)  # Wait for 60 seconds after every 50 requests

# Convert the results list into a DataFrame
output_df = pd.DataFrame(results)

# Save the DataFrame to the output CSV
output_df.to_csv(output_csv, index=False)

print(f"Results have been saved to {output_csv}")