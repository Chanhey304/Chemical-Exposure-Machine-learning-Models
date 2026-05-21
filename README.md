# Chemical-Exposure-Machine-learning-Models

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
