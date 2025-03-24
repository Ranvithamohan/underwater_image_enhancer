import sys
import pandas as pd
import numpy as np
from PIL import Image
from src.exception import CustomException
from src.utils import load_object
from src.components.underwater_enhancer import extract_image_features  

class PredictPipeline:
    def __init__(self):
        pass

    def predict_quality(self, image_features: dict):
        """
        Predicts image quality score from extracted features
        Args:
            image_features: Dictionary containing image features
                          (width, height, r_mean, g_mean, b_mean, contrast, etc.)
        Returns:
            Predicted quality score
        """
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Convert features to DataFrame
            features_df = pd.DataFrame([image_features])
            
            # Preprocess and predict
            data_scaled = preprocessor.transform(features_df)
            preds = model.predict(data_scaled)
            
            return preds[0]  # Return single prediction
            
        except Exception as e:
            raise CustomException(e, sys)

class ImageData:
    def __init__(self,
                 image_path: str = None,
                 width: int = None,
                 height: int = None,
                 r_mean: float = None,
                 g_mean: float = None,
                 b_mean: float = None,
                 contrast: float = None):
        """
        Initialize with either:
        - image_path: Path to image file (auto-extracts features)
        OR
        - Manual feature values (width, height, color means, etc.)
        """
        if image_path:
            self.image = Image.open(image_path)
            self.features = extract_image_features(self.image)
        else:
            self.features = {
                'width': width,
                'height': height,
                'r_mean': r_mean,
                'g_mean': g_mean,
                'b_mean': b_mean,
                'contrast': contrast
            }
    
    def get_features_as_dataframe(self):
        """
        Returns features as pandas DataFrame for prediction
        """
        try:
            return pd.DataFrame([self.features])
        except Exception as e:
            raise CustomException(e, sys)

# Example usage:
# pipeline = PredictPipeline()
# data = ImageData(image_path="underwater.jpg")
# features_df = data.get_features_as_dataframe()
# quality_score = pipeline.predict_quality(features_df)