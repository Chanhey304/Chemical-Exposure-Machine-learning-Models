"""
XGBoost Classification for Hyperspectral UAV Survey Data
Associated with the manuscript: [Damage Evolution and Recovery Dynamics of Vegetation Exposed to Toxic Chemical Gases: Development of a Real-World Applicable Hyperspectral UAV Survey Protocol]
Journal: IEEE J-STARS

This script loads hyperspectral data, trains an XGBoost model to classify
chemical substances using early stopping and cross-validation, and outputs 
detailed performance metrics including feature importances and confusion matrices.

Usage:
    python xgb_classification.py --input data.csv --label classes.txt --output ./results/
"""

import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             cohen_kappa_score, precision_recall_fscore_support, roc_auc_score)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperspectral Data Classification using XGBoost")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--label', type=str, required=True, help="Path to the label text file")
    parser.add_argument('--output', type=str, required=True, help="Directory to save the results")
    return parser.parse_args()

def format_confusion_matrix(cm, class_names, title):
    total = cm.sum(axis=1)
    acc = np.round(np.diag(cm) / total * 100, 2)
    total_acc = np.round(np.sum(np.diag(cm)) / np.sum(cm) * 100, 2)
    
    df_cm = pd.DataFrame(
        cm,
        columns=[f'Predicted_{c}' for c in class_names],
        index=[f'Actual_{c}' for c in class_names]
    )
    df_cm['Accuracy(%)'] = acc
    df_cm['Sum_of_sample(Actual)'] = total
    
    total_row = pd.DataFrame(
        [[None]*len(class_names) + [total_acc, np.sum(cm)]],
        index=['Total_Accuracy(%)'],
        columns=df_cm.columns
    )
    
    final_df = pd.concat([df_cm, total_row])
    final_df.columns.name = title
    return final_df

def get_confusion_metrics(cm, class_idx):
    TP = cm[class_idx, class_idx]
    FP = np.sum(cm[:, class_idx]) - TP
    FN = np.sum(cm[class_idx, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    return TP, TN, FP, FN

def main():
    args = parse_arguments()
    input_file = args.input
    label_file = args.label
    output_dir = args.output

    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load label file
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            selected_classes = [line.strip() for line in f if line.strip()]
        print(f"Loaded classes: {selected_classes}")
    except Exception as e:
        raise SystemExit(f"Program terminated: Error reading label file - {str(e)}")

    # Load and preprocess data
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except Exception as e:
        raise SystemExit(f"Program terminated: Error reading input data - {str(e)}")

    df = df[df['Label'].isin(selected_classes)]

    X = df.drop('Label', axis=1)
    y = df['Label']
    wavelengths = X.columns.astype(float)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 1. Main Train/Test Split (70/30) - Test is strictly for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
    )

    # 2. Train/Validation Split from Train set (80/20) for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Calculate class weights based on the training subset
    class_weights_tr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_tr),
        y=y_tr
    )
    w_tr  = np.array([class_weights_tr[label] for label in y_tr])
    w_val = np.array([class_weights_tr[label] for label in y_val])

    # 3. XGBoost Training Preparation
    dtr  = xgb.DMatrix(X_tr,  label=y_tr,  weight=w_tr)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_encoded)),
        'eval_metric': 'mlogloss',
        'eta': 0.01,
        'max_depth': 2,
        'min_child_weight': 5,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.3,
        'reg_lambda': 3.0,
        'seed': 42
    }

    evals = [(dtr, 'train'), (dval, 'eval')]
    
    print("\nTraining XGBoost model...")
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # 4. Predictions and Accuracy Evaluation
    # Full Train evaluation (70% total train set)
    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    y_train_pred = np.argmax(booster.predict(dtrain_full), axis=1)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Test evaluation (30% hold-out set)
    y_test_pred = np.argmax(booster.predict(dtest), axis=1)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\nModel Performance Summary:")
    print(f"  Training Accuracy (70% subset): {train_acc:.4f}")
    print(f"  Testing Accuracy (30% subset):  {test_acc:.4f}")

    # Cross-validation strictly on the Train set (70%)
    print("\nRunning Cross-Validation on Training set...")
    cv_acc = []
    cv_kappa = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_tr_idx, fold_val_idx in skf.split(X_train, y_train):
        X_fold_tr = X_train.iloc[fold_tr_idx]
        X_fold_val = X_train.iloc[fold_val_idx]
        y_fold_tr = y_train[fold_tr_idx]
        y_fold_val = y_train[fold_val_idx]

        cw = compute_class_weight('balanced', classes=np.unique(y_fold_tr), y=y_fold_tr)
        w_fold_tr = np.array([cw[label] for label in y_fold_tr])
        w_fold_val = np.array([cw[label] for label in y_fold_val])

        d_fold_tr = xgb.DMatrix(X_fold_tr, label=y_fold_tr, weight=w_fold_tr)
        d_fold_val = xgb.DMatrix(X_fold_val, label=y_fold_val, weight=w_fold_val)

        fold_model = xgb.train(
            params=params,
            dtrain=d_fold_tr,
            num_boost_round=5000,
            evals=[(d_fold_tr, 'train'), (d_fold_val, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        y_fold_val_pred = np.argmax(fold_model.predict(d_fold_val), axis=1)
        cv_acc.append(accuracy_score(y_fold_val, y_fold_val_pred))
        cv_kappa.append(cohen_kappa_score(y_fold_val, y_fold_val_pred))

    print(f"  CV Mean Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

    # 5. Feature Importance Calculation and Visualization
    importance_types = ['weight', 'gain', 'cover']
    importance_scores = {}
    for typ in importance_types:
        score_dict = booster.get_score(importance_type=typ)
        scores = np.zeros(len(wavelengths))
        for i, col in enumerate(X.columns):
            if col in score_dict:
                scores[i] = score_dict[col]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        importance_scores[typ] = scores

    combined = np.mean([importance_scores[t] for t in importance_types], axis=0)
    feature_df = pd.DataFrame({
        'Wavelength': wavelengths,
        'Weight_Score': importance_scores['weight'],
        'Gain_Score': importance_scores['gain'],
        'Cover_Score': importance_scores['cover'],
        'Combined_Score': combined
    })

    plt.figure(figsize=(20, 15))
    for i, col in enumerate(['Weight_Score', 'Gain_Score', 'Cover_Score', 'Combined_Score']):
        top = feature_df.nlargest(30, col)
        plt.subplot(2, 2, i+1)
        plt.bar(range(30), top[col])
        plt.xticks(range(30), top['Wavelength'].round(1), rotation=45)
        plt.title(f'Top 30 {col}')
        plt.xlabel('Wavelength')
        plt.ylabel('Score')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "feature_importance_all.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nFeature importance plots saved to: {plot_path}")

    # 6. Confusion Matrix and Detailed Metrics
    y_train_orig = le.inverse_transform(y_train)
    y_test_orig = le.inverse_transform(y_test)
    y_train_pred_orig = le.inverse_transform(y_train_pred)
    y_test_pred_orig = le.inverse_transform(y_test_pred)

    cm_train = confusion_matrix(y_train_orig, y_train_pred_orig, labels=selected_classes)
    cm_test = confusion_matrix(y_test_orig, y_test_pred_orig, labels=selected_classes)

    train_prec, train_rec, train_f1, train_supp = precision_recall_fscore_support(
        y_train_orig, y_train_pred_orig, labels=selected_classes, zero_division=0
    )
    test_prec, test_rec, test_f1, test_supp = precision_recall_fscore_support(
        y_test_orig, y_test_pred_orig, labels=selected_classes, zero_division=0
    )

    detailed_metrics = pd.DataFrame({
        'Class': selected_classes,
        'Train_Precision': train_prec, 'Train_Recall': train_rec, 'Train_F1_Score': train_f1, 'Train_Support': train_supp,
        'Test_Precision': test_prec, 'Test_Recall': test_rec, 'Test_F1_Score': test_f1, 'Test_Support': test_supp
    })

    try:
        y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_train_pred_proba = booster.predict(dtrain_full)
        y_test_pred_proba = booster.predict(dtest)
        roc_auc_train = roc_auc_score(y_train_bin, y_train_pred_proba, multi_class='ovr')
        roc_auc_test = roc_auc_score(y_test_bin, y_test_pred_proba, multi_class='ovr')
    except Exception:
        roc_auc_train = roc_auc_test = "Not applicable"

    kappa_train = cohen_kappa_score(y_train_orig, y_train_pred_orig)
    kappa_test = cohen_kappa_score(y_test_orig, y_test_pred_orig)

    general_metrics = pd.DataFrame({
        'Metric': ['Cohen_Kappa', 'ROC_AUC'],
        'Training': [kappa_train, roc_auc_train],
        'Test': [kappa_test, roc_auc_test]
    })

    confusion_metrics = []
    for idx, class_name in enumerate(selected_classes):
        train_TP, train_TN, train_FP, train_FN = get_confusion_metrics(cm_train, idx)
        test_TP, test_TN, test_FP, test_FN = get_confusion_metrics(cm_test, idx)
        confusion_metrics.append({
            'Class': class_name,
            'Train_True_Positive': train_TP, 'Train_True_Negative': train_TN, 
            'Train_False_Positive': train_FP, 'Train_False_Negative': train_FN,
            'Test_True_Positive': test_TP, 'Test_True_Negative': test_TN, 
            'Test_False_Positive': test_FP, 'Test_False_Negative': test_FN
        })

    # 7. Save Results to Excel
    results_file = os.path.join(output_dir, 'XGB_detailed_results.xlsx')
    print(f"Saving detailed results to: {results_file}")
    
    with pd.ExcelWriter(results_file) as writer:
        pd.DataFrame({
            'Training_Accuracy': [train_acc],
            'Testing_Accuracy': [test_acc],
            'CV_Accuracy_Mean': [np.mean(cv_acc)],
            'CV_Accuracy_Std': [np.std(cv_acc)],
            'CV_Kappa_Mean': [np.mean(cv_kappa)],
            'CV_Kappa_Std': [np.std(cv_kappa)]
        }).to_excel(writer, sheet_name='Overall_Performance', index=False)

        feature_df.sort_values('Combined_Score', ascending=False).to_excel(writer, sheet_name='Wavelength_Importance', index=False)
        format_confusion_matrix(cm_train, selected_classes, "Training_XGB_Model").to_excel(writer, sheet_name='Training_Confusion_Matrix')
        format_confusion_matrix(cm_test, selected_classes, "Test_XGB_Model").to_excel(writer, sheet_name='Test_Confusion_Matrix')
        
        pd.DataFrame(
            classification_report(y_test_orig, y_test_pred_orig, target_names=selected_classes, output_dict=True, zero_division=0)
        ).transpose().to_excel(writer, sheet_name='Classification_Report')
        
        detailed_metrics.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
        general_metrics.to_excel(writer, sheet_name='General_Metrics', index=False)
        pd.DataFrame(confusion_metrics).to_excel(writer, sheet_name='Confusion_Metrics', index=False)
        
        # Convert params values to string if necessary, to avoid Excel writing issues with specific types
        safe_params = {k: str(v) for k, v in params.items()}
        pd.DataFrame(safe_params.items(), columns=['Parameter', 'Value']).to_excel(writer, sheet_name='Model_Parameters', index=False)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()