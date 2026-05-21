"""
Random Forest Classification for Hyperspectral UAV Survey Data
Associated with the manuscript: [Damage Evolution and Recovery Dynamics of Vegetation Exposed to Toxic Chemical Gases: Development of a Real-World Applicable Hyperspectral UAV Survey Protocol]
Journal: IEEE J-STARS

This script loads hyperspectral data, trains a Random Forest model to classify
chemical substances, and outputs detailed performance metrics including confusion
matrices, feature importances, and classification reports.

Usage:
    python rf_classification.py --input data.csv --label classes.txt --output ./results/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperspectral Data Classification using Random Forest")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--label', type=str, required=True, help="Path to the label text file")
    parser.add_argument('--output', type=str, required=True, help="Directory to save the results")
    return parser.parse_args()

def format_confusion_matrix(cm, classes, title):
    total_samples = np.sum(cm, axis=1)
    # Avoid division by zero warnings by replacing 0 with small epsilon if necessary, 
    # though total_samples for actual classes shouldn't be 0 in stratified sampling.
    class_accuracies = np.round(np.diag(cm) / total_samples * 100, 2)
    total_accuracy = np.round(np.sum(np.diag(cm)) / np.sum(cm) * 100, 2)
    
    cm_df = pd.DataFrame(cm, 
                         columns=[f'Predicted_{c}' for c in classes],
                         index=[f'Actual_{c}' for c in classes])
    
    cm_df['Accuracy(%)'] = class_accuracies
    cm_df['Sum_of_sample(Actual)'] = total_samples
    
    total_row = pd.DataFrame([['' for _ in classes] + [total_accuracy, np.sum(cm)]],
                             index=['Total_Accuracy(%)'],
                             columns=cm_df.columns)
    
    final_df = pd.concat([cm_df, total_row])
    final_df.columns.name = title
    return final_df

def main():
    args = parse_arguments()
    input_file = args.input
    label_file = args.label
    output_dir = args.output

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load label file
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            selected_classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded classes: {selected_classes}")
    except Exception as e:
        raise SystemExit(f"Program terminated: Error reading label file - {str(e)}")

    # Load and preprocess data
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except Exception as e:
        raise SystemExit(f"Program terminated: Error reading input data - {str(e)}")
        
    df_selected = df[df['Label'].isin(selected_classes)]

    print("\nOriginal class distribution:")
    original_dist = df_selected['Label'].value_counts()
    print(original_dist)

    X = df_selected.drop('Label', axis=1)
    y = df_selected['Label']
    wavelengths = X.columns.astype(float)

    # Stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=y
    )

    print("\nExpected vs Actual test set sizes (30% of original):")
    test_dist = y_test.value_counts()
    for class_name in selected_classes:
        expected_size = int(original_dist.get(class_name, 0) * 0.3)
        actual_size = test_dist.get(class_name, 0)
        print(f"  {class_name}: Expected {expected_size}, Actual {actual_size}")

    # Model definition and training
    rf = RandomForestClassifier(
        n_estimators=280,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        bootstrap=True,
        oob_score=True
    )

    print("\nTraining Random Forest model...")
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    # Performance Evaluation
    final_train_score = rf.score(X_train, y_train)
    final_test_score = rf.score(X_test, y_test)
    cv_scores = cross_val_score(rf, X, y, cv=5)

    print("\nModel Performance Summary:")
    print(f"  Training Accuracy: {final_train_score:.4f}")
    print(f"  Testing Accuracy: {final_test_score:.4f}")
    print(f"  Gap (Train-Test): {final_train_score - final_test_score:.4f}")
    print(f"  CV Mean Accuracy: {cv_scores.mean():.4f}")

    # Results formatting
    results_df = pd.DataFrame({
        'Class': selected_classes,
        'Training_Accuracy': [final_train_score] * len(selected_classes),
        'Testing_Accuracy': [final_test_score] * len(selected_classes),
        'CV_Mean': [cv_scores.mean()] * len(selected_classes),
        'CV_Std': [cv_scores.std()] * len(selected_classes)
    })

    cm_train = confusion_matrix(y_train, y_train_pred, labels=selected_classes)
    cm_test = confusion_matrix(y_test, y_pred, labels=selected_classes)
    
    train_cm_formatted = format_confusion_matrix(cm_train, selected_classes, "Training_RF_Model")
    test_cm_formatted = format_confusion_matrix(cm_test, selected_classes, "Test_RF_Model")

    feature_importances = pd.DataFrame({
        'Wavelength': wavelengths,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Additional Metrics Compilation
    kappa_train = cohen_kappa_score(y_train, y_train_pred)
    kappa_test = cohen_kappa_score(y_test, y_pred)
    
    train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(
        y_train, y_train_pred, labels=selected_classes, zero_division=0)
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(
        y_test, y_pred, labels=selected_classes, zero_division=0)

    try:
        y_train_bin = label_binarize(y_train, classes=selected_classes)
        y_test_bin = label_binarize(y_test, classes=selected_classes)
        roc_auc_train = roc_auc_score(y_train_bin, rf.predict_proba(X_train), multi_class='ovr')
        roc_auc_test = roc_auc_score(y_test_bin, rf.predict_proba(X_test), multi_class='ovr')
    except Exception:
        roc_auc_train = roc_auc_test = "N/A"

    detailed_metrics = pd.DataFrame({
        'Class': selected_classes,
        'Train_Precision': train_precision, 'Train_Recall': train_recall, 'Train_F1': train_f1,
        'Test_Precision': test_precision, 'Test_Recall': test_recall, 'Test_F1': test_f1
    })

    general_metrics = pd.DataFrame({
        'Metric': ['Cohen_Kappa', 'ROC_AUC', 'OOB_Score'],
        'Training': [kappa_train, roc_auc_train, rf.oob_score_],
        'Test': [kappa_test, roc_auc_test, rf.oob_score_]
    })

    # Save to Excel
    results_file = os.path.join(output_dir, 'RF_detailed_results.xlsx')
    print(f"\nSaving detailed results to: {results_file}")
    
    with pd.ExcelWriter(results_file) as writer:
        results_df.to_excel(writer, sheet_name='Overall_Performance', index=False)
        feature_importances.reset_index(drop=True).to_excel(writer, sheet_name='Wavelength_Importance', index=True)
        train_cm_formatted.to_excel(writer, sheet_name='Training_Confusion_Matrix')
        test_cm_formatted.to_excel(writer, sheet_name='Test_Confusion_Matrix')
        
        classification_dict = classification_report(y_test, y_pred, target_names=selected_classes, output_dict=True, zero_division=0)
        pd.DataFrame(classification_dict).transpose().to_excel(writer, sheet_name='Classification_Report')
        
        pd.DataFrame({
            'Parameter': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'random_state'],
            'Value': [rf.n_estimators, rf.max_depth, rf.min_samples_split, rf.min_samples_leaf, rf.max_features, rf.random_state]
        }).to_excel(writer, sheet_name='Model_Parameters', index=False)

        detailed_metrics.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
        general_metrics.to_excel(writer, sheet_name='General_Metrics', index=False)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()