# ML Pipeline с CI/CD

ML Pipeline с автоматизацией и мониторингом.

## Структура проекта
mlflow/  
│  
├── .gitignore  
├── .gitlab-ci.yaml  
├── README.md  
├── requirement.txt  
├── main.py  
├── test_main.py  
│  
├── github/                     
│   └── ml-pipeline.yaml  
│  
├── configs/                     
│   └── model_config.yaml  
├── data/ # данные для обучения модели  
│   ├── sample_data.csv              
│   └── processed/  
│       ├── test_data.csv  
│       └── train_data.csv  
├── mlruns/    
├── reports/  
├── src/         
│   ├── train.py  
│   ├── validate.py  
│   ├── data_validation.py  
│   └── prediction.py  


## Быстрый старт
```bash
# установка зависимостей
pip install -r requirements.txt

# запуск полного пайплайна
python src/train.py
python src/validate.py

# просмотр отчетов
open reports/deepchecks_report.html
open reports/data_drift_report.html

# получение предсказания:
curl -X POST "http://...:8080/prediction" -H "Content-Type: application/json" -d "{\"features\": {\"feature_0\": 0.5, \"feature_1\": -0.2, \"feature_2\": 1.1, \"feature_3\": -0.8, \"feature_4\": 0.3, \"feature_5\": -1.2, \"feature_6\": 0.7, \"feature_7\": -0.5, \"feature_8\": 1.5, \"feature_9\": -0.1}}"
{"prediction":1,"probabilities":{"class_0":0.3071326936189818,"class_1":0.6928673063810181},"model_type":"RandomForestClassifier","features_used":["feature_0","feature_1","feature_2","feature_3","feature_4","feature_5","feature_6","feature_7","feature_8","feature_9"],"status":"success"}
