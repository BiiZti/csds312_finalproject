# Social Media Addiction Analysis Project

This project analyzes the relationship between social media addiction and different demographic groups using machine learning approaches.

## Project Structure

```
├── data/
│   ├── social_media_usage/    # Social media usage dataset
│   └── time_spent/           # Time spent on social media dataset
├── visualizations/           # Generated plots and visualizations
├── Final Project.py         # Data download script
├── analysis.py             # Main analysis script
└── requirements.txt        # Project dependencies
```

## Team Members and Responsibilities

- Random Forest Classifier: Lakshmi Kunjan and Lalithya Gangula
- Logistic Regression: Alvisa Krasniqi and Salma Bhar
- XGBoost: Shenguo Wu and Nathan Kim

## Setup Instructions

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure Kaggle API:
   - Place your `kaggle.json` file in the project root directory
   - Run the data download script:
     ```bash
     python "Final Project.py"
     ```

3. Run the analysis:
   ```bash
   python analysis.py
   ```

## Analysis Pipeline

1. Data Loading and Cleaning
   - Handle missing values
   - Remove duplicates
   - Feature engineering

2. Exploratory Data Analysis (EDA)
   - Distribution analysis
   - Correlation analysis
   - Demographic analysis
   - Time series analysis

3. Model Training and Evaluation
   - Random Forest Classifier
   - Logistic Regression
   - XGBoost
   - Model comparison and insights

## Results

The analysis results, including visualizations and model comparisons, will be saved in the `visualizations/` directory. 