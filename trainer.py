"""
RFE Trainer - Tests 3 RFE strategies for feature selection
"""
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR, CLASSES, LABEL_MAP, TRAIN_SIZE, TEST_SIZE,
    RANDOM_STATE, RFE_STRATEGIES, MODELS_TO_TRAIN, HYPERPARAMETER_GRIDS,
    CV_FOLDS, CV_SCORING
)

warnings.filterwarnings('ignore')


class RFETrainer:
    """Trainer for RFE experiments"""

    def __init__(self, strategy_key: str):
        self.strategy_key = strategy_key
        self.strategy = RFE_STRATEGIES[strategy_key]
        self.output_dir = self.strategy['output_dir']
        self.reports_dir = self.strategy['reports_dir']

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # Storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.selector = None
        self.rfe_selector = None
        self.selected_feature_indices = None

    def load_data(self):
        """Load processed data from single detector ml v2"""
        print("\n" + "="*70)
        print(f"LOADING DATA - {self.strategy['name']}")
        print("="*70)

        features_path = PROCESSED_DATA_DIR / 'features.npy'
        labels_path = PROCESSED_DATA_DIR / 'labels.npy'

        if not features_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Processed data not found in {PROCESSED_DATA_DIR}")

        X = np.load(features_path)
        y = np.load(labels_path)

        print(f"[OK] Loaded data: X shape {X.shape}, y shape {y.shape}")

        # Print class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {CLASSES[label]:25s}: {count:6d} samples")

        return X, y

    def stratified_split(self, X: np.ndarray, y: np.ndarray):
        """Split data with stratified sampling (maintains class balance)"""
        print("\n" + "="*70)
        print("STRATIFIED DATA SPLIT")
        print("="*70)

        # First split: 90% train, 10% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Second split: 90% train, 10% val (from train_val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1111,  # 10% of total
            random_state=RANDOM_STATE, stratify=y_train_val
        )

        print(f"Train set: {X_train.shape[0]} samples ({TRAIN_SIZE*0.9:.1%})")
        print(f"Val set:   {X_val.shape[0]} samples ({TRAIN_SIZE*0.1:.1%})")
        print(f"Test set:  {X_test.shape[0]} samples ({TEST_SIZE:.1%})")

        # Verify class balance
        print("\nTrain set class distribution:")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_train, counts_train):
            print(f"  {CLASSES[label]:25s}: {count:6d} ({count/len(y_train)*100:.2f}%)")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self):
        """Standardize features"""
        print("\n" + "="*70)
        print("FEATURE SCALING")
        print("="*70)

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"[OK] Scaled features using StandardScaler")
        print(f"     Train: {self.X_train.shape}")
        print(f"     Val:   {self.X_val.shape}")
        print(f"     Test:  {self.X_test.shape}")

    def apply_feature_selection(self):
        """Apply feature selection based on strategy"""
        print("\n" + "="*70)
        print(f"FEATURE SELECTION - {self.strategy['name']}")
        print("="*70)
        print(f"Strategy: {self.strategy['description']}")
        print(f"Initial features: {self.X_train.shape[1]}")

        # Store initial feature count
        self.initial_features = self.X_train.shape[1]

        X_train_selected = self.X_train
        X_val_selected = self.X_val
        X_test_selected = self.X_test

        # Stage 1: SelectKBest (if enabled)
        if self.strategy.get('use_selectkbest', False):
            k = self.strategy['selectkbest_k']
            print(f"\n[Stage 1] SelectKBest: selecting top {k} features...")

            self.selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = self.selector.fit_transform(X_train_selected, self.y_train)
            X_val_selected = self.selector.transform(X_val_selected)
            X_test_selected = self.selector.transform(X_test_selected)

            print(f"[OK] After SelectKBest: {X_train_selected.shape[1]} features")

        # Stage 2: RFE or RFECV
        if self.strategy.get('use_rfecv', False):
            # Use RFECV (cross-validated RFE)
            min_features = self.strategy['min_features_to_select']
            cv_folds = self.strategy['cv_folds']

            print(f"\n[Stage 2] RFECV: cross-validated feature selection...")
            print(f"  Minimum features: {min_features}")
            print(f"  CV folds: {cv_folds}")
            print(f"  This may take several minutes...")

            # Use a fast estimator for RFECV (LightGBM)
            estimator = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                verbose=-1
            )

            self.rfe_selector = RFECV(
                estimator=estimator,
                step=1,
                cv=cv_folds,
                scoring=CV_SCORING,
                min_features_to_select=min_features,
                n_jobs=-1
            )

            X_train_selected = self.rfe_selector.fit_transform(X_train_selected, self.y_train)
            X_val_selected = self.rfe_selector.transform(X_val_selected)
            X_test_selected = self.rfe_selector.transform(X_test_selected)

            n_selected = self.rfe_selector.n_features_
            print(f"[OK] RFECV selected {n_selected} optimal features")
            print(f"     Grid scores shape: {self.rfe_selector.grid_scores_.shape}")

        else:
            # Use standard RFE
            n_features = self.strategy['n_features_to_select']

            print(f"\n[Stage 2] RFE: selecting {n_features} features...")
            print(f"  This may take several minutes...")

            # Use a fast estimator for RFE (LightGBM)
            estimator = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                verbose=-1
            )

            self.rfe_selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=1
            )

            X_train_selected = self.rfe_selector.fit_transform(X_train_selected, self.y_train)
            X_val_selected = self.rfe_selector.transform(X_val_selected)
            X_test_selected = self.rfe_selector.transform(X_test_selected)

            print(f"[OK] RFE selected {n_features} features")

        # Store selected features
        self.selected_feature_indices = self.rfe_selector.get_support(indices=True)

        print(f"\n[FINAL] Selected features: {X_train_selected.shape[1]}")
        print(f"Feature reduction: {self.initial_features} -> {X_train_selected.shape[1]}")
        print(f"Reduction rate: {(1 - X_train_selected.shape[1]/self.initial_features)*100:.1f}%")

        # Update datasets
        self.X_train = X_train_selected
        self.X_val = X_val_selected
        self.X_test = X_test_selected

        # Save feature selection info
        self._save_feature_selection_info()

    def _save_feature_selection_info(self):
        """Save feature selection information"""
        info = {
            'strategy': self.strategy['name'],
            'description': self.strategy['description'],
            'initial_features': self.initial_features,
            'final_features': self.X_train.shape[1],
            'selected_indices': self.selected_feature_indices.tolist() if self.selected_feature_indices is not None else None
        }

        if self.strategy.get('use_rfecv', False):
            info['rfecv_grid_scores'] = self.rfe_selector.grid_scores_.tolist()
            info['rfecv_optimal_features'] = self.rfe_selector.n_features_

        with open(self.reports_dir / 'feature_selection_info.json', 'w') as f:
            json.dump(info, f, indent=2)

    def train_model(self, model_name: str):
        """Train a single model with hyperparameter tuning"""
        print("\n" + "="*70)
        print(f"TRAINING: {model_name.upper()}")
        print("="*70)

        start_time = time.time()

        # Get model and parameter grid
        model = self._get_base_model(model_name)
        param_grid = HYPERPARAMETER_GRIDS[model_name]

        n_combinations = len(list(self._grid_combinations(param_grid)))
        print(f"Hyperparameter grid: {n_combinations} combinations")
        print(f"Cross-validation: {CV_FOLDS}-fold")
        print(f"Total fits: {n_combinations * CV_FOLDS}")

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=CV_FOLDS,
            scoring=CV_SCORING,
            n_jobs=-1,
            verbose=1  # Show progress
        )

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting GridSearchCV...")
        grid_search.fit(self.X_train, self.y_train)

        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GridSearchCV completed in {elapsed/60:.1f} minutes")

        print(f"[OK] Best hyperparameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"     {param}: {value}")
        print(f"[OK] Best CV score: {grid_search.best_score_:.4f}")

        # Evaluate on validation set
        val_pred = grid_search.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        print(f"[OK] Validation accuracy: {val_acc:.4f}")

        # Evaluate on test set
        test_pred = grid_search.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, test_pred)
        print(f"[OK] Test accuracy: {test_acc:.4f}")

        # Save model
        model_path = self.output_dir / f"{model_name}.pkl"
        joblib.dump(grid_search.best_estimator_, model_path)
        print(f"[OK] Model saved to {model_path}")

        # Generate reports
        self._generate_reports(model_name, grid_search, test_pred)

        return grid_search.best_estimator_, test_acc

    def _get_base_model(self, model_name: str):
        """Get base model instance"""
        if model_name == 'xgboost':
            return xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _grid_combinations(self, param_grid):
        """Generate all combinations from parameter grid"""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _generate_reports(self, model_name: str, grid_search, test_pred):
        """Generate comprehensive reports"""
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)

        # Classification report
        report = classification_report(
            self.y_test, test_pred,
            target_names=CLASSES,
            output_dict=True
        )

        # Save JSON report
        results = {
            'model': model_name,
            'strategy': self.strategy['name'],
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'val_accuracy': float(accuracy_score(self.y_val, grid_search.predict(self.X_val))),
            'test_accuracy': float(accuracy_score(self.y_test, test_pred)),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        with open(self.reports_dir / f"{model_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Plot confusion matrix
        self._plot_confusion_matrix(cm, model_name)

        print(f"[OK] Reports saved to {self.reports_dir}")

    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))

        # Normalize to percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(
            cm_pct,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cbar_kws={'label': 'Percentage (%)'}
        )

        plt.title(f'{model_name.upper()} - Confusion Matrix\n{self.strategy["name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.savefig(self.reports_dir / f"{model_name}_confusion_matrix.png", dpi=150)
        plt.close()

    def train_all_models(self):
        """Train all models"""
        print("\n" + "="*70)
        print(f"TRAINING ALL MODELS - {self.strategy['name']}")
        print("="*70)
        print(f"Total models to train: {len(MODELS_TO_TRAIN)}")
        print(f"Models: {', '.join(MODELS_TO_TRAIN)}")

        results = {}
        for idx, model_name in enumerate(MODELS_TO_TRAIN, 1):
            print(f"\n[PROGRESS] Model {idx}/{len(MODELS_TO_TRAIN)}: {model_name}")
            model, test_acc = self.train_model(model_name)
            results[model_name] = test_acc
            print(f"[PROGRESS] Completed {idx}/{len(MODELS_TO_TRAIN)} models ({idx/len(MODELS_TO_TRAIN)*100:.1f}%)")

        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*70)
        print(f"Strategy: {self.strategy['name']}")
        print(f"Models trained: {len(results)}")
        print("\nTest Accuracies:")
        for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name:20s}: {acc:.4f} ({acc*100:.2f}%)")

        # Save summary
        with open(self.reports_dir / 'summary.json', 'w') as f:
            json.dump({
                'strategy': self.strategy['name'],
                'test_accuracies': results
            }, f, indent=2)

        return results


def run_strategy(strategy_key: str):
    """Run a single RFE strategy"""
    trainer = RFETrainer(strategy_key)

    # Load data
    X, y = trainer.load_data()

    # Split data
    trainer.stratified_split(X, y)

    # Scale features
    trainer.scale_features()

    # Apply feature selection
    trainer.apply_feature_selection()

    # Train all models
    results = trainer.train_all_models()

    return results


def run_all_strategies():
    """Run all RFE strategies"""
    all_results = {}

    total_strategies = len(RFE_STRATEGIES)
    print(f"\n[MAIN PROGRESS] Total strategies to run: {total_strategies}")
    print(f"[MAIN PROGRESS] Strategies: {', '.join(RFE_STRATEGIES.keys())}\n")

    for idx, strategy_key in enumerate(RFE_STRATEGIES.keys(), 1):
        print("\n" + "="*100)
        print(f"[MAIN PROGRESS] STRATEGY {idx}/{total_strategies}: {RFE_STRATEGIES[strategy_key]['name']}")
        print(f"[MAIN PROGRESS] Overall Progress: {(idx-1)/total_strategies*100:.1f}% complete")
        print("="*100)

        strategy_start = time.time()
        results = run_strategy(strategy_key)
        strategy_elapsed = time.time() - strategy_start

        all_results[strategy_key] = results

        print(f"\n[MAIN PROGRESS] Strategy {idx}/{total_strategies} completed in {strategy_elapsed/60:.1f} minutes")
        print(f"[MAIN PROGRESS] Overall Progress: {idx/total_strategies*100:.1f}% complete")
        print("\n")

    # Final comparison
    print("\n" + "="*100)
    print("FINAL COMPARISON - ALL STRATEGIES")
    print("="*100)

    for strategy_key, results in all_results.items():
        strategy_name = RFE_STRATEGIES[strategy_key]['name']
        print(f"\n{strategy_name}:")
        for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name:20s}: {acc*100:.2f}%")


if __name__ == "__main__":
    run_all_strategies()
