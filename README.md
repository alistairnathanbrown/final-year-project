# Final Year Project: Probability of Default for Energy Projects

This repository contains the source code and data for my final-year research project, **"Using Generative AI to Calculate the Probability of Default for Energy Projects."** The project aims to assess the credit risk of large debt-dependent construction projects, particularly offshore wind farms, by leveraging generative AI and machine learning techniques.

## Repository Structure

inal-year-project/ │── data/ │ ├── raw/ # Raw datasets (wind farm data, financial data, etc.) │ ├── processed/ # Cleaned and preprocessed datasets │ ├── api_responses/ # Cached API responses for reproducibility │── src/ │ ├── data_collection/ # Scripts for data gathering and API interactions │ │ ├── fetch_wind_data.py # Fetch wind project data │ │ ├── fetch_macro_data.py # Fetch macroeconomic indicators (inflation, rates, etc.) │ │ ├── fetch_credit_data.py # Fetch credit ratings and financials │ ├── preprocessing/ # Scripts for cleaning, feature engineering, and transformation │ ├── models/ │ │ ├── train_deepseek.py # Training script for DeepSeek model │ │ ├── train_llama.py # Training script for LLaMA model │ │ ├── test_models.py # Model evaluation and testing scripts │ ├── notebooks/ # Jupyter Notebooks for exploratory analysis and validation │── results/ │ ├── model_outputs/ # Results from model training and inference │ ├── figures/ # Plots and visualizations │── requirements.txt # Python dependencies │── README.md # Project documentation │── LICENSE # License for the repository

## Project Overview

### Objective
This research explores the application of generative AI in assessing the **Probability of Default (PD)** for large-scale energy projects. Traditional credit risk models are well-studied in the consumer loan space but lack application in project finance. By integrating machine learning with domain-specific features like inflation, energy yield, construction cost overruns, and credit ratings, this study aims to improve risk assessment for lenders and investors.

### Data Sources
- **Wind Farm Database:** Contains project-specific details (location, capacity, status).
- **Macroeconomic Indicators:** Inflation, interest rates, commodity prices (collected via APIs).
- **Credit Ratings & Financials:** Firm-level financial risk factors.
- **Construction Risk Data:** Cost overruns and delays.

### Model Implementation
- **DeepSeek and LLaMA** are trained and tested to assess their performance in predicting default risk.
- Custom feature engineering is applied to encode time-series and categorical variables.
- The project leverages **LLM fine-tuning** to enhance performance on sparse datasets.
