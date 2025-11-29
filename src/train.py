import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation
import os
import yaml
from data_validation import create_sample_data, load_config

# создаем директории
os.makedirs('reports', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def load_or_create_data():
    config = load_config()
    target_col = config['data']['target_column']
    
    # проверяем, есть ли уже разделенные данные
    train_path = 'data/processed/train_data.csv'
    test_path = 'data/processed/test_data.csv'
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("загружаем существующие разделенные данные...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Разделяем на features и target
        X_train = train_data.drop(target_col, axis=1)
        y_train = train_data[target_col]
        X_test = test_data.drop(target_col, axis=1)
        y_test = test_data[target_col]
        
        print(f"загружено: Train {X_train.shape}, Test {X_test.shape}")
        
    else:
        print("Создаем новые данные и разделяем...")
        data = create_sample_data()
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Делим данные на тренировочные и тестовые
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        # Сохраняем разделенные данные для будущих запусков
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        print("Новые данные сохранены")
    
    return X_train, X_test, y_train, y_test

def run_training():
    # загружаем конфигурацию
    config = load_config()
    target_col = config['data']['target_column']
    
    # загружаем или создаем данные
    X_train, X_test, y_train, y_test = load_or_create_data()
    
    print("Стартуем mlflow...")
    with mlflow.start_run():
        mlflow.log_params(config['model']['parameters'])
        mlflow.log_param("model_name", config['model']['name'])
        mlflow.log_param("test_size", config['data']['test_size'])
        mlflow.log_param("data_source", "existing" if os.path.exists('data/processed/train_data.csv') else "new")
        
        # тренируем модель
        print("Тренируем модель...")
        model = RandomForestClassifier(**config['model']['parameters'])
        model.fit(X_train, y_train)
        
        # считаем метрики
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # валидация Deepchecks
        print("Валидация Deepchecks...")
        ds_train = Dataset(X_train, label=y_train, cat_features=[])
        ds_test = Dataset(X_test, label=y_test, cat_features=[])
        suite = model_evaluation()
        
        result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)
        
        # сохраняем отчет
        result.save_as_html('reports/deepchecks_report.html', as_widget=False)
        mlflow.log_artifact('reports/deepchecks_report.html')
        
        # сохраняем модель
        mlflow.sklearn.log_model(model, "model")
        
        # сохраняем информацию о данных для валидации
        data_info = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X_train.columns),
            'target_column': target_col,
            'data_loaded_from_existing': os.path.exists('data/processed/train_data.csv')
        }
        mlflow.log_dict(data_info, "data_info.json")
        
        print("Тренировка модели завершена успешно!")
    
    print("MLflow завершился. Данные запуска сохранены в папке mlruns.")

if __name__ == "__main__":
    run_training()
