# Hyperspectral UAV Survey for Chemical Classification

This repository contains the Python source code for the machine learning classification models (Random Forest and XGBoost) used in the manuscript:

**"[Damage Evolution and Recovery Dynamics of Vegetation Exposed to Toxic Chemical Gases: Development of a Real-World Applicable Hyperspectral UAV Survey Protocol]"**  
Submitted to *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (J-STARS)*.

## Overview
This codebase provides automated pipelines to train, evaluate, and extract feature importances for hyperspectral data collected via UAVs. The models are designed to classify organic chemicals using spectral reflectance data. 

* `rf_classification.py`: Trains a Random Forest classifier.
* `xgb_classification.py`: Trains an XGBoost classifier with cross-validation and early stopping.

## Requirements
The scripts require Python 3.8+ and the following libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`

You can install the dependencies using:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

## Data Availability
As stated in the manuscript, the experimental hyperspectral data are available from the corresponding author upon reasonable request. 

For testing purposes, place your spectral data (`.csv`) and a text file containing the target classes (`.txt`) in the same directory. Ensure the data is encoded in UTF-8.

## Usage
Both scripts can be executed via the command line. They require three arguments: the input dataset, the text file containing the selected classes, and the output directory path.

**To run the Random Forest model:**
```bash
python rf_classification.py --input data.csv --label classes.txt --output ./results_rf/
```

**To run the XGBoost model:**
```bash
python xgb_classification.py --input data.csv --label classes.txt --output ./results_xgb/
```

## Outputs
After execution, the scripts will generate the following in the specified output directory:
* `[Model]_detailed_results.xlsx`: Contains overall accuracy, cross-validation metrics, class-wise precision/recall/F1-scores, and confusion matrices.
* `plots/feature_importance_all.png`: A visualization of the top 30 most important wavelengths.
