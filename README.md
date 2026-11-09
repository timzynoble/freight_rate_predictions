# Freight Rate Prediction Assessment

This repository contains the end-to-end machine learning solution for predicting freight shipping rates using historical load data, as part of the Micro1 Data Science Assessment.

## Project Structure

```text
freight-rate-prediction/
├── README.md                          # Setup and execution instructions
├── requirements.txt                   # Exact Python dependencies
├── src/                               # Source code modules
│   ├── preprocess.py                  # Data cleaning and feature engineering
│   ├── train.py                       # Time-based validation split & CatBoost training
│   └── predict.py                     # Generates validation and December predictions
├── data/                              # Input CSV datasets
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
├── outputs/                           # Final generated submission files
│   ├── validation_predictions.csv
│   └── december-chart-inputs.csv
└── reports/                           # Final report and visual charts
    ├── technical_report.pdf           # (or .docx)
    └── candidate_december.png         # December prediction chart