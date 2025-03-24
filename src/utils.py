import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from PIL import Image
import json

def save_object(file_path, obj):
    """
    Save Python object to file using dill
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models with hyperparameter tuning and return performance metrics
    """
    try:
        report = {}
        for model_name, model in models.items():
            para = param.get(model_name, {})
            
            # Hyperparameter tuning if parameters are provided
            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
            
            # Train model
            best_model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'best_params': gs.best_params_ if para else {}
            }
            
            report[model_name] = metrics
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load Python object from file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def save_image_metrics(image_path, metrics, output_dir="artifacts/metrics"):
    """
    Save image enhancement metrics to JSON file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    except Exception as e:
        raise CustomException(e, sys)

def extract_image_features(image):
    """
    Extract features from underwater image for quality prediction
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Basic image features
        features = {
            'width': image.width,
            'height': image.height,
            'aspect_ratio': image.width / image.height,
            'r_mean': np.mean(img_array[:, :, 0]),
            'g_mean': np.mean(img_array[:, :, 1]),
            'b_mean': np.mean(img_array[:, :, 2]),
            'contrast': np.std(img_array),
            'entropy': image_entropy(image)
        }
        return features
    except Exception as e:
        raise CustomException(e, sys)

def image_entropy(image):
    """
    Calculate image entropy (measure of information content)
    """
    histogram = image.histogram()
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram if h != 0]
    return -sum([p * np.log2(p) for p in samples_probability])