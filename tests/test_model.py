"""
Tests for model.py module.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import build_model, get_top_features, get_all_top_features


class TestModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.X_train = [
            "positive financial news about stocks",
            "negative market crash happening",
            "neutral market conditions today",
            "great earnings report positive outlook",
            "terrible losses negative sentiment",
            "normal trading day neutral"
        ]
        self.y_train = ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']
        
        self.X_test = [
            "positive market trends",
            "negative outlook"
        ]
    
    def test_build_model(self):
        """Test model building."""
        model = build_model(max_features=100, ngram_range=(1, 2))
        self.assertIsNotNone(model)
        self.assertEqual(len(model.named_steps), 2)
        self.assertIn('tfidf', model.named_steps)
        self.assertIn('classifier', model.named_steps)
    
    def test_model_fit(self):
        """Test that model can be fitted."""
        model = build_model(max_features=100)
        model.fit(self.X_train, self.y_train)
        self.assertTrue(hasattr(model.named_steps['classifier'], 'coef_'))
    
    def test_model_predict(self):
        """Test model prediction."""
        model = build_model(max_features=100)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(p in ['positive', 'neutral', 'negative'] for p in predictions))
    
    def test_model_predict_proba(self):
        """Test model probability prediction."""
        model = build_model(max_features=100)
        model.fit(self.X_train, self.y_train)
        probabilities = model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], 3)  # 3 classes
        # Probabilities should sum to 1
        for prob in probabilities:
            self.assertAlmostEqual(np.sum(prob), 1.0, places=5)
    
    def test_get_top_features(self):
        """Test getting top features for a class."""
        model = build_model(max_features=100)
        model.fit(self.X_train, self.y_train)
        
        # Should be able to get top features for each class
        for class_name in ['positive', 'neutral', 'negative']:
            top_features = get_top_features(model, class_name, top_n=10)
            self.assertIsInstance(top_features, list)
            self.assertLessEqual(len(top_features), 10)
            if len(top_features) > 0:
                self.assertIsInstance(top_features[0], tuple)
                self.assertEqual(len(top_features[0]), 2)  # (feature, weight)
    
    def test_get_all_top_features(self):
        """Test getting top features for all classes."""
        model = build_model(max_features=100)
        model.fit(self.X_train, self.y_train)
        
        all_features = get_all_top_features(model, top_n=10)
        self.assertIsInstance(all_features, dict)
        self.assertIn('positive', all_features)
        self.assertIn('neutral', all_features)
        self.assertIn('negative', all_features)
    
    def test_get_top_features_unfitted(self):
        """Test that getting top features fails on unfitted model."""
        model = build_model(max_features=100)
        with self.assertRaises(ValueError):
            get_top_features(model, 'positive', top_n=10)
    
    def test_get_top_features_invalid_class(self):
        """Test that invalid class name raises error."""
        model = build_model(max_features=100)
        model.fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            get_top_features(model, 'invalid_class', top_n=10)


if __name__ == '__main__':
    unittest.main()

