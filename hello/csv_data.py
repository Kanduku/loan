import pandas as pd
import numpy as np
import os
import joblib

from django.http import JsonResponse
from django.conf import settings

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from joblib import parallel_backend


def csv_data_json(request):
    try:
        # === Paths & Setup ===
        base_dir = settings.BASE_DIR
        model_dir = base_dir / 'model_cache'
        os.makedirs(model_dir, exist_ok=True)

        model_path = model_dir / 'best_model_compressed.joblib'
        labelencoder_path = model_dir / 'label_encoder.joblib'
        split_path = model_dir / 'split_data.joblib'

        # === Fast-Path: Cached Load ===
        if model_path.exists() and labelencoder_path.exists() and split_path.exists():
            best_model = joblib.load(model_path)
            le = joblib.load(labelencoder_path)
            X_test, y_test = joblib.load(split_path)

        else:
            # === Load CSV only once ===
            df = pd.read_csv(base_dir / 'static' / 'loan_approval_dataset.csv')
            df.columns = df.columns.str.strip()

            X = df.drop(columns='loan_status')
            y = df['loan_status']

            # === Identify Columns ===
            num_cols = X.select_dtypes(include=np.number).columns.difference(['loan_id']).tolist()
            cat_cols = ['education', 'self_employed']

            # === Encode Target ===
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            joblib.dump(le, labelencoder_path, compress=3)

            # === Train/Test Split ===
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
            )
            joblib.dump((X_test, y_test), split_path, compress=3)

            # === Preprocessing ===
            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('power', PowerTransformer(method='yeo-johnson')),
                    ('pca', PCA(n_components=0.95, svd_solver='full'))  # Keep 95% variance
                ]), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ], n_jobs=-1)

            # === Model Pipeline ===
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
            ])

            # === Light GridSearch ===
            param_grid = {
                'classifier__max_depth': [5],
                'classifier__min_samples_split': [2],
                'classifier__min_samples_leaf': [1],
                'classifier__max_leaf_nodes': [50]
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            with parallel_backend('loky'):
                grid = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
                grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            joblib.dump(best_model, model_path, compress=3)

        # === Evaluate ===
        y_pred = best_model.predict(X_test)

        result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'auc_roc': roc_auc_score(y_test, y_pred),
            'sample_predictions': le.inverse_transform(y_pred[:10]).tolist(),
            'feature_importance': best_model.named_steps['classifier'].feature_importances_.tolist(),
            'best_params': getattr(best_model, 'best_params_', 'Loaded from cache')
        }

        return JsonResponse(result, safe=False)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
