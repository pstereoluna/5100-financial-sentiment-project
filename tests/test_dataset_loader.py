"""
Tests for dataset_loader.py module.
"""

import unittest
import sys
import os
import tempfile
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_loader import load_twitter_financial, load_dataset


class TestDatasetLoader(unittest.TestCase):
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV with known labels
        self.test_data = pd.DataFrame({
            'text': [
                'Stock downgraded, price target cut',  # Bearish content
                'Company beats earnings, stock upgraded',  # Bullish content
                'Market trading sideways today',  # Neutral content
            ],
            'label': [0, 1, 2]  # 0=Bearish, 1=Bullish, 2=Neutral
        })
        self.test_path = os.path.join(self.temp_dir, 'test.csv')
        self.test_data.to_csv(self.test_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_label_mapping_correctness(self):
        """
        Verify label mapping matches HuggingFace dataset definition:
        - 0 = Bearish = negative
        - 1 = Bullish = positive
        - 2 = Neutral = neutral
        """
        df = load_twitter_financial(self.test_path)
        
        # Check that labels are correctly mapped
        self.assertEqual(df.loc[0, 'label'], 'negative')  # 0 -> negative
        self.assertEqual(df.loc[1, 'label'], 'positive')  # 1 -> positive
        self.assertEqual(df.loc[2, 'label'], 'neutral')   # 2 -> neutral
    
    def test_label_distribution_after_mapping(self):
        """Verify all three label types are present after mapping."""
        df = load_twitter_financial(self.test_path)
        
        labels = set(df['label'].unique())
        expected = {'positive', 'neutral', 'negative'}
        self.assertEqual(labels, expected)
    
    def test_load_dataset_function(self):
        """Test the unified load_dataset function."""
        df = load_dataset('twitter_financial', self.test_path)
        
        self.assertIn('text', df.columns)
        self.assertIn('label', df.columns)
        self.assertEqual(len(df), 3)
    
    def test_invalid_dataset_name(self):
        """Test that invalid dataset name raises error."""
        with self.assertRaises(ValueError):
            load_dataset('invalid_dataset', self.test_path)
    
    def test_string_label_mapping(self):
        """Test string label mappings."""
        test_data_str = pd.DataFrame({
            'text': ['Text 1', 'Text 2', 'Text 3'],
            'label': ['bearish', 'bullish', 'neutral']
        })
        test_path_str = os.path.join(self.temp_dir, 'test_str.csv')
        test_data_str.to_csv(test_path_str, index=False)
        
        df = load_twitter_financial(test_path_str)
        
        self.assertEqual(df.loc[0, 'label'], 'negative')  # bearish -> negative
        self.assertEqual(df.loc[1, 'label'], 'positive')  # bullish -> positive
        self.assertEqual(df.loc[2, 'label'], 'neutral')   # neutral -> neutral
    
    def test_numeric_string_mapping(self):
        """Test numeric string mappings ('0', '1', '2')."""
        test_data_num_str = pd.DataFrame({
            'text': ['Text 1', 'Text 2', 'Text 3'],
            'label': ['0', '1', '2']  # String form of numbers
        })
        test_path_num_str = os.path.join(self.temp_dir, 'test_num_str.csv')
        test_data_num_str.to_csv(test_path_num_str, index=False)
        
        df = load_twitter_financial(test_path_num_str)
        
        self.assertEqual(df.loc[0, 'label'], 'negative')  # '0' -> negative
        self.assertEqual(df.loc[1, 'label'], 'positive')  # '1' -> positive
        self.assertEqual(df.loc[2, 'label'], 'neutral')   # '2' -> neutral


if __name__ == '__main__':
    unittest.main()

