"""
RFE Experiments - Configuration
Tests 3 RFE strategies: RFE-only, Hybrid (SelectKBest+RFE), RFE-CV
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data_v2_len_div_10"
REPORTS_BASE_DIR = BASE_DIR / "reports_v2_len_div_10"

# Model output directories
TRAINED_MODELS_RFE_ONLY = BASE_DIR / "trained_models_rfe_only"
TRAINED_MODELS_HYBRID = BASE_DIR / "trained_models_hybrid"
TRAINED_MODELS_RFE_CV = BASE_DIR / "trained_models_rfe_cv"

REPORTS_RFE_ONLY = BASE_DIR / "reports_rfe_only"
REPORTS_HYBRID = BASE_DIR / "reports_hybrid"
REPORTS_RFE_CV = BASE_DIR / "reports_rfe_cv"

# Data split
TRAIN_SIZE = 0.9
TEST_SIZE = 0.1
RANDOM_STATE = 42

# Classes (9-class classification) - FIXED!
CLASSES = [
    'collective_anomaly',
    'contextual_anomaly',
    'deterministic_trend',
    'mean_shift',
    'point_anomaly',
    'Stochastic Trend',
    'trend_shift',
    'variance_shift',
    'Volatility'
]

LABEL_MAP = {class_name: idx for idx, class_name in enumerate(CLASSES)}

# RFE Strategies
RFE_STRATEGIES = {
    'rfe_only': {
        'name': 'RFE Only',
        'description': 'Pure RFE feature selection (no SelectKBest)',
        'use_selectkbest': False,
        'use_rfecv': False,
        'n_features_to_select': 25,
        'output_dir': TRAINED_MODELS_RFE_ONLY,
        'reports_dir': REPORTS_RFE_ONLY
    },
    'hybrid': {
        'name': 'Hybrid (SelectKBest + RFE)',
        'description': 'Two-stage: SelectKBest 60->40, then RFE 40->25',
        'use_selectkbest': True,
        'selectkbest_k': 40,
        'use_rfecv': False,
        'n_features_to_select': 25,
        'output_dir': TRAINED_MODELS_HYBRID,
        'reports_dir': REPORTS_HYBRID
    },
    'rfe_cv': {
        'name': 'RFE-CV',
        'description': 'Cross-validated RFE (automatic feature selection)',
        'use_selectkbest': False,
        'use_rfecv': True,
        'min_features_to_select': 20,
        'cv_folds': 3,
        'output_dir': TRAINED_MODELS_RFE_CV,
        'reports_dir': REPORTS_RFE_CV
    }
}

# Models to train (best performers from v2)
MODELS_TO_TRAIN = ['xgboost', 'lightgbm', 'random_forest']

# Hyperparameter grids for each model
HYPERPARAMETER_GRIDS = {
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# Cross-validation settings
CV_FOLDS = 3
CV_SCORING = 'accuracy'
