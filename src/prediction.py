import mlflow
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

def load_latest_model():
    # загружаем последнюю обученную модель из MLflow
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        latest_run = None
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs and (latest_run is None or runs[0].info.start_time > latest_run.info.start_time):
                latest_run = runs[0]
        
        if latest_run is None:
            raise ValueError("No trained models found in MLflow")
        
        model_uri = f"runs:/{latest_run.info.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        data_info_path = mlflow.artifacts.download_artifacts(
            run_id=latest_run.info.run_id, 
            artifact_path="data_info.json"
        )
        
        print(f"Загружена модель: {latest_run.info.run_id}")
        return model, data_info_path
        
    except Exception as e:
        raise Exception(f"Ошибка загрузки модели: {str(e)}")

def get_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    # получение предсказания
    try:
        # загружаем модель и информацию о данных
        model, data_info_path = load_latest_model()
        
        # создаем DataFrame с правильным порядком признаков
        feature_names = list(features.keys())
        feature_values = [features[feature] for feature in feature_names]
        
        input_data = pd.DataFrame([feature_values], columns=feature_names)
        
        # получаем предсказание
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": {
                "class_0": float(prediction_proba[0]),
                "class_1": float(prediction_proba[1])
            },
            "model_type": type(model).__name__,
            "features_used": feature_names,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "prediction": None,
            "error": str(e),
            "status": "error"
        }
