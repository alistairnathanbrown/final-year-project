import csv
from collections import Counter

def extract_and_sort_companies(input_file, output_file):
    company_counter = Counter()

    # Read the input file
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for entry in row:
                companies = entry.split(';')
                companies = [company.strip() for company in companies]  # Remove extra whitespace
                company_counter.update(companies)

    # Sort companies by number of occurrences in descending order
    sorted_companies = company_counter.most_common()

    # Write the results to the output file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Company Name', 'Occurrences'])  # Add headers
        writer.writerows(sorted_companies)

# Usage example:
input_file = 'companies/companies.csv'
output_file = 'companies/unique_companies_sorted.csv'
extract_and_sort_companies(input_file, output_file)