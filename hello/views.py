import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def csv_data_json(request):
    # Load CSV
    file_path = settings.BASE_DIR / 'static' / 'loan_approval_dataset.csv'
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    # --- Feature & Target Split ---
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    # --- Column Identification ---
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    if 'loan_id' in numerical_cols:
        numerical_cols.remove('loan_id')
    categorical_cols = ['education', 'self_employed']

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Target Encoding ---
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # --- Preprocessors ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('pca', PCA())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # --- Model Pipeline ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('decisiontreeclassifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ])

    # --- GridSearch Setup ---
    param_grid = {
        'decisiontreeclassifier__max_depth': [3, 5],
        'decisiontreeclassifier__min_samples_split': [2, 10],
        'decisiontreeclassifier__min_samples_leaf': [1, 5],
        'decisiontreeclassifier__max_leaf_nodes': [10, 50],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_encoded)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    original_y_pred = le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    class_report = classification_report(y_test_encoded, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test_encoded, y_pred).tolist()
    auc_roc = roc_auc_score(y_test_encoded, y_pred)

    # Optional: Feature importances
    importances = best_model.named_steps['decisiontreeclassifier'].feature_importances_.tolist()

    result = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'auc_roc': auc_roc,
        'best_params': grid_search.best_params_,
        'sample_predictions': original_y_pred[:10].tolist(),
        'feature_importance': importances
    }

    return JsonResponse(result, safe=False)






@csrf_exempt
def predict_loan_status(request):
    if request.method == 'POST':
        try:
            # 1. Load data
            file_path = settings.BASE_DIR / 'static' / 'loan_approval_dataset.csv'
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()

            X = data.drop('loan_status', axis=1)
            y = data['loan_status']

            # 2. Train the model
            categorical_cols = ['education', 'self_employed']
            numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
            if 'loan_id' in numerical_cols:
                numerical_cols.remove('loan_id')

            preprocessor = ColumnTransformer(transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('power', PowerTransformer(method='yeo-johnson'))
                ]), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ])

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            pipeline.fit(X, y_encoded)

            # 3. Get input from request
            import json
            body = json.loads(request.body)

            input_df = pd.DataFrame([body])

            # 4. Predict
            prediction = pipeline.predict(input_df)
            result = le.inverse_transform(prediction)[0]

            return JsonResponse({'prediction': result})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid method'}, status=405)