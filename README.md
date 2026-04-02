# Gaussian Model Generator

A Streamlit web application for PtMeOH Gaussian surrogate modeling from Aspen Plus Excel data. [file:1]

## Features

- Welcome modal dialog with guided onboarding. [file:1]
- Sequential workflow with visible, logically locked tabs. [file:1]
- Database Analyzer for Aspen Excel inspection and variable confirmation. [file:1]
- Data Cleaning & Preparation with representative external test split. [file:1]
- Training & Validation using 5-fold cross-validation. [file:1]
- Gaussian Process Regression with Matérn 2.5 default kernel and optional RBF benchmark. [file:1]
- External Test & Packing workflow. [file:1]
- Exportable Excel files, plots, PDF reports, Python model package, text summary, and serialized model. [file:1]
- Final left sidebar download center. [file:1]

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected workflow

1. Upload Aspen Excel data. [file:1]
2. Confirm the detected hydrogen-flow input column. [file:1]
3. Select and confirm one output column. [file:1]
4. Clean data and create the representative external test split. [file:1]
5. Run 5-fold cross-validation and retrain the final production GP. [file:1]
6. Run external testing and package the accepted model. [file:1]

## Notes

- Version 1 is optimized for a 1D PtMeOH surrogate where hydrogen flow is the model input. [file:1]
- The architecture is intentionally prepared for future multi-input and multi-output extensions. [file:1]
