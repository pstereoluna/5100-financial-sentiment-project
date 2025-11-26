"""
Helper script to download and format datasets from various sources.

This script helps you quickly get started with common financial sentiment datasets.
"""

import pandas as pd
import os
import sys
import argparse


def download_huggingface_phrasebank(output_path: str = 'data/financial_phrasebank.csv'):
    """
    Download Financial PhraseBank from Hugging Face.
    
    Requires: pip install datasets
    
    Args:
        output_path: Where to save the CSV file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install it with: pip install datasets")
        return False
    
    print("Downloading Financial PhraseBank from Hugging Face...")
    
    try:
        # Try different versions/configurations
        configs_to_try = [
            "sentences_50agree",
            "sentences_66agree", 
            "sentences_75agree",
            "sentences_allagree"
        ]
        
        dataset = None
        for config in configs_to_try:
            try:
                dataset = load_dataset("financial_phrasebank", config)
                print(f"Loaded configuration: {config}")
                break
            except:
                continue
        
        if dataset is None:
            # Try without config
            dataset = load_dataset("financial_phrasebank")
        
        # Convert to pandas
        df = dataset['train'].to_pandas()
        
        # Standardize column names
        # PhraseBank typically has: sentence, label (or similar)
        text_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'sent' in col_lower or 'text' in col_lower or 'phrase' in col_lower:
                text_col = col
            if 'label' in col_lower or 'sentiment' in col_lower:
                label_col = col
        
        if text_col is None or label_col is None:
            print(f"Warning: Could not auto-detect columns. Available: {df.columns.tolist()}")
            print("Please manually format the dataset.")
            return False
        
        # Create standardized dataframe
        result_df = pd.DataFrame({
            'text': df[text_col],
            'label': df[label_col]
        })
        
        # Map labels to standard format
        label_mapping = {
            'positive': 'positive',
            'neutral': 'neutral',
            'negative': 'negative',
            'pos': 'positive',
            'neu': 'neutral',
            'neg': 'negative',
        }
        
        result_df['label'] = result_df['label'].map(label_mapping).fillna(result_df['label'])
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Total samples: {len(result_df)}")
        print(f"  Label distribution:\n{result_df['label'].value_counts()}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def download_huggingface_financial_sentiment(output_path: str = 'data/financial_sentiment.csv'):
    """
    Download Financial-Sentiment-Analysis dataset from Hugging Face.
    
    Args:
        output_path: Where to save the CSV file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install it with: pip install datasets")
        return False
    
    print("Downloading Financial-Sentiment-Analysis from Hugging Face...")
    
    try:
        dataset = load_dataset("Financial-Sentiment-Analysis/financial_sentiment_analysis")
        df = dataset['train'].to_pandas()
        
        # Standardize
        result_df = pd.DataFrame({
            'text': df['sentence'],
            'label': df['sentiment']
        })
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"  Total samples: {len(result_df)}")
        print(f"  Label distribution:\n{result_df['label'].value_counts()}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def format_custom_dataset(input_path: str, output_path: str, 
                         text_col: str = None, label_col: str = None):
    """
    Format a custom dataset to match expected format.
    
    Args:
        input_path: Path to input CSV/TSV
        output_path: Path to save formatted CSV
        text_col: Name of text column (auto-detect if None)
        label_col: Name of label column (auto-detect if None)
    """
    print(f"Formatting dataset from {input_path}...")
    
    # Try CSV first, then TSV
    try:
        df = pd.read_csv(input_path)
    except:
        df = pd.read_csv(input_path, sep='\t')
    
    # Auto-detect columns if not provided
    if text_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['text', 'sent', 'phrase', 'tweet', 'message', 'sentence']):
                text_col = col
                break
    
    if label_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['label', 'sentiment', 'class']):
                label_col = col
                break
    
    if text_col is None or label_col is None:
        print(f"Error: Could not detect columns.")
        print(f"Available columns: {df.columns.tolist()}")
        print("Please specify --text_col and --label_col")
        return False
    
    # Create standardized dataframe
    result_df = pd.DataFrame({
        'text': df[text_col],
        'label': df[label_col]
    })
    
    # Map labels
    label_mapping = {
        'positive': 'positive',
        'neutral': 'neutral',
        'negative': 'negative',
        'pos': 'positive',
        'neu': 'neutral',
        'neg': 'negative',
        'bullish': 'positive',
        'bearish': 'negative',
    }
    
    result_df['label'] = result_df['label'].map(label_mapping).fillna(result_df['label'])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"✓ Formatted dataset saved to {output_path}")
    print(f"  Total samples: {len(result_df)}")
    print(f"  Label distribution:\n{result_df['label'].value_counts()}")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and format financial sentiment datasets')
    parser.add_argument('--dataset', type=str, 
                       choices=['phrasebank', 'financial_sentiment', 'custom'],
                       help='Dataset to download')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: data/{dataset}.csv)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input path (for custom dataset)')
    parser.add_argument('--text_col', type=str, default=None,
                       help='Text column name (for custom dataset)')
    parser.add_argument('--label_col', type=str, default=None,
                       help='Label column name (for custom dataset)')
    
    args = parser.parse_args()
    
    if args.dataset == 'phrasebank':
        output = args.output or 'data/financial_phrasebank.csv'
        download_huggingface_phrasebank(output)
    
    elif args.dataset == 'financial_sentiment':
        output = args.output or 'data/financial_sentiment.csv'
        download_huggingface_financial_sentiment(output)
    
    elif args.dataset == 'custom':
        if args.input is None:
            print("Error: --input required for custom dataset")
            sys.exit(1)
        output = args.output or 'data/custom_dataset.csv'
        format_custom_dataset(args.input, output, args.text_col, args.label_col)
    
    else:
        print("Please specify --dataset (phrasebank, financial_sentiment, or custom)")

