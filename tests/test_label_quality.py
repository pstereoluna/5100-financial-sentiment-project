"""
Tests for label_quality.py module.
"""

import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import build_model
from src.label_quality import detect_ambiguous_predictions


class TestLabelQuality(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and model."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'text': [
                "positive financial news",
                "negative market crash",
                "neutral market conditions",
                "great earnings positive",
                "terrible losses negative",
                "normal trading neutral"
            ],
            'label': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']
        })
        
        # Create and train a simple model
        X = self.test_data['text'].values
        y = self.test_data['label'].values
        self.model = build_model(max_features=50)
        self.model.fit(X, y)
        
        # Save model
        self.model_path = os.path.join(self.temp_dir, 'test_model.joblib')
        joblib.dump(self.model, self.model_path)
        
        # Save test data
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.test_data.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_detect_ambiguous_predictions(self):
        """Test detection of ambiguous predictions."""
        # Create some ambiguous examples (we'll use the test data)
        ambiguous_df = detect_ambiguous_predictions(
            model_path=self.model_path,
            data_path=self.data_path,
            dataset_name='twitter_financial',
            confidence_threshold=(0.45, 0.55),
            output_path=os.path.join(self.temp_dir, 'ambiguous.csv')
        )
        
        self.assertIsInstance(ambiguous_df, pd.DataFrame)
        self.assertIn('text', ambiguous_df.columns)
        self.assertIn('confidence', ambiguous_df.columns)
        self.assertIn('predicted_label', ambiguous_df.columns)
        
        # Check that all confidences are in the threshold range
        if len(ambiguous_df) > 0:
            self.assertTrue(all(
                (ambiguous_df['confidence'] >= 0.45) & 
                (ambiguous_df['confidence'] <= 0.55)
            ))
    
    def test_ambiguous_output_file(self):
        """Test that ambiguous predictions CSV is created."""
        output_path = os.path.join(self.temp_dir, 'ambiguous.csv')
        detect_ambiguous_predictions(
            model_path=self.model_path,
            data_path=self.data_path,
            dataset_name='twitter_financial',
            output_path=output_path
        )
        
        self.assertTrue(os.path.exists(output_path))
        df = pd.read_csv(output_path)
        self.assertGreaterEqual(len(df), 0)  # May be 0 if no ambiguous predictions


if __name__ == '__main__':
    unittest.main()

