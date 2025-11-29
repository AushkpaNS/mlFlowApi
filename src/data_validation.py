import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os
import yaml

def load_config():
    """Load configuration from YAML file"""
    with open('configs/model_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_sample_data():
    """Create sample data if it doesn't exist"""
    config = load_config()
    
    if not os.path.exists('data/sample_data.csv'):
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2,
            random_state=config['data']['random_state']
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df[config['data']['target_column']] = y
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_data.csv', index=False)
        print("Sample data created!")
    
    return pd.read_csv('data/sample_data.csv')
