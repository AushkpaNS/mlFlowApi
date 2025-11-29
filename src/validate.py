import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import warnings
import yaml
import os
from data_validation import load_config

warnings.filterwarnings('ignore')

def run_evidently_analysis():
    print("Начинаем анализ данных с помощью EvidentlyAI...")
    
    # загружаем конфигурацию
    config = load_config()
    target_col = config['data']['target_column']
    drift_threshold = config['validation']['drift_threshold']
    
    # загружаем те же данные, что использовались при обучении
    print("Загружаем данные из обучения...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # проверяем, что данные загружены корректно
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train columns: {train_data.columns.tolist()}")
    print(f"Test columns: {test_data.columns.tolist()}")
    
    # Data drift анализ (сравниваем train и test)
    print("Стартуем data drift анализ между train и test...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=train_data,
        current_data=test_data
    )
    data_drift_report.save_html('reports/data_drift_report.html')
    print("Data drift отчет сохранён в reports/data_drift_report.html")
    
    # Target drift анализ
    print("Стартуем target drift анализ...")
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=train_data,
        current_data=test_data
    )
    target_drift_report.save_html('reports/target_drift_report.html')
    print("Target drift отчет сохранён в reports/target_drift_report.html")
    
    # анализ результатов
    print("\n" + "="*50)
    print("EVIDENTLYAI ANALYSIS RESULTS")
    print("="*50)
    
    data_drift_metrics = data_drift_report.as_dict()
    target_drift_metrics = target_drift_report.as_dict()
    
    # извлекаем метрики
    n_drifted_features = data_drift_metrics['metrics'][0]['result']['number_of_drifted_columns']
    share_drifted_features = data_drift_metrics['metrics'][0]['result']['share_of_drifted_columns']
    dataset_drift = data_drift_metrics['metrics'][0]['result']['dataset_drift']
    
    target_drift = target_drift_metrics['metrics'][0]['result']['drift_detected']
    target_drift_score = target_drift_metrics['metrics'][0]['result']['drift_score']
    
    # выводим результат 
    print(f"Dataset drift detected: {dataset_drift}")
    print(f"Number of drifted features: {n_drifted_features}")
    print(f"Share of drifted features: {share_drifted_features:.2%}")
    print(f"Target drift detected: {target_drift}")
    print(f"Target drift score: {target_drift_score:.4f}")
    print(f"Drift threshold: {drift_threshold}")
    
    print("\nINTERPRETATION:")
    print("Мы сравниваем распределения признаков и целевой переменной")
    print("между тренировочной (reference) и тестовой (current) выборками.")
    
    print("\nRECOMMENDATIONS:")
    if dataset_drift or target_drift:
        print("Drift detected. Considerations:")
        print("- Проверить корректность разделения данных")
        print("- Убедиться, что данные репрезентативны")
        print("- Рассмотреть стратифицированное разделение")
        if share_drifted_features > drift_threshold:
            print("- High feature drift detected. Model performance may degrade.")
    else:
        print("No significant drift detected. Data splitting is consistent.")
        print("Model should perform reliably on test data.")
    
    print("EvidentlyAI анализ завершён!")
    return dataset_drift, target_drift

if __name__ == "__main__":
    run_evidently_analysis()
    