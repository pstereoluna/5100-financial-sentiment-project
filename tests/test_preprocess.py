"""
Tests for preprocess.py module.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import clean_text, preprocess_batch


class TestPreprocess(unittest.TestCase):
    
    def test_url_removal(self):
        """Test that URLs are removed."""
        text = "Check out https://example.com for more info"
        cleaned = clean_text(text)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("http", cleaned)
    
    def test_cashtag_removal(self):
        """Test that cashtags are removed."""
        text = "I love $TSLA and $AAPL stocks"
        cleaned = clean_text(text)
        self.assertNotIn("$TSLA", cleaned)
        self.assertNotIn("$AAPL", cleaned)
        self.assertNotIn("tsla", cleaned)
        self.assertNotIn("aapl", cleaned)
    
    def test_hashtag_removal(self):
        """Test that hashtags are removed."""
        text = "This is #finance and #stocks"
        cleaned = clean_text(text)
        self.assertNotIn("#finance", cleaned)
        self.assertNotIn("#stocks", cleaned)
    
    def test_mention_removal(self):
        """Test that @mentions are removed."""
        text = "Hey @elonmusk what do you think?"
        cleaned = clean_text(text)
        self.assertNotIn("@elonmusk", cleaned)
    
    def test_lowercase(self):
        """Test that text is lowercased."""
        text = "THIS IS UPPERCASE TEXT"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "this is uppercase text")
    
    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        text = "Hello, world! How are you?"
        cleaned = clean_text(text)
        self.assertNotIn(",", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)
    
    def test_tokenization(self):
        """Test that text is tokenized."""
        text = "This is a test sentence"
        cleaned = clean_text(text)
        # Should be space-separated tokens
        tokens = cleaned.split()
        self.assertGreater(len(tokens), 0)
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        texts = [
            "Check $TSLA https://example.com",
            "This is #finance",
            "Hello @world"
        ]
        cleaned = preprocess_batch(texts)
        self.assertEqual(len(cleaned), len(texts))
        self.assertIsInstance(cleaned[0], str)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        text = ""
        cleaned = clean_text(text)
        self.assertIsInstance(cleaned, str)
    
    def test_complex_example(self):
        """Test a complex real-world example."""
        text = "RT @user: $TSLA is going up! Check https://t.co/abc #stocks #finance"
        cleaned = clean_text(text)
        # Should not contain any of the removed elements
        self.assertNotIn("$TSLA", cleaned)
        self.assertNotIn("@user", cleaned)
        self.assertNotIn("https://t.co/abc", cleaned)
        self.assertNotIn("#stocks", cleaned)
        self.assertNotIn("#finance", cleaned)


if __name__ == '__main__':
    unittest.main()

