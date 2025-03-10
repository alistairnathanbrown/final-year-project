# Final Year Project: Probability of Default for Energy Projects

This repository contains the source code and data for my final-year research project, **"Using Generative AI to Calculate the Probability of Default for Energy Projects."** The project aims to assess the credit risk of large debt-dependent construction projects, particularly offshore wind farms, by leveraging generative AI and machine learning techniques.

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
