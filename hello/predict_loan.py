import pandas as pd
import numpy as np
import json
import os
import joblib

from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@csrf_exempt
def predict_loan_status(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)

    try:
        # === Set up file paths ===
        model_dir = settings.BASE_DIR / 'model_cache'
        os.makedirs(model_dir, exist_ok=True)

        model_path = model_dir / 'loan_model.joblib'
        encoder_path = model_dir / 'label_encoder.joblib'
        metadata_path = model_dir / 'column_metadata.joblib'

        # === TRAIN & CACHE if necessary ===
        if not model_path.exists():
            file_path = settings.BASE_DIR / 'static' / 'loan_approval_dataset.csv'
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()

            X = data.drop(columns='loan_status')
            y = data['loan_status']

            categorical_cols = ['education', 'self_employed']
            numerical_cols = X.select_dtypes(include=np.number).columns.difference(['loan_id']).tolist()

            # Preprocessing setup
            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('power', PowerTransformer(method='yeo-johnson')),
                ]), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ], n_jobs=-1)

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ])

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Fit the pipeline
            pipeline.fit(X, y_encoded)

            # Save everything
            joblib.dump(pipeline, model_path, compress=3)
            joblib.dump(le, encoder_path, compress=3)
            joblib.dump({'numerical_cols': numerical_cols, 'categorical_cols': categorical_cols}, metadata_path, compress=3)

        # === LOAD cached models ===
        pipeline = joblib.load(model_path)
        le = joblib.load(encoder_path)
        metadata = joblib.load(metadata_path)

        # === Parse input ===
        body = json.loads(request.body)
        expected_columns = metadata['numerical_cols'] + metadata['categorical_cols']

        input_df = pd.DataFrame([body])
        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            return JsonResponse({'error': f'Missing input fields: {missing_cols}'}, status=400)

        # === Predict ===
        prediction = pipeline.predict(input_df)
        result = le.inverse_transform(prediction)[0]

        return JsonResponse({'prediction': result})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
