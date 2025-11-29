# ML Pipeline с CI/CD

ML Pipeline с автоматизацией и мониторингом.

## Структура проекта
- `src/train.py` - обучение модели с MLflow и Deepchecks
- `src/validate.py` - анализ дрейфа данных с EvidentlyAI
- `data/sample_data.csv` - данные для модели
- `data/rocessed/train_data.csv` тестовые данные
- `.github/` - пайплайн GitHub Actions
- `.gitlab-ci.yml` - пайплайн GitLab CI
- `configs/model_config.yaml` - централизованная конфигурация

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
