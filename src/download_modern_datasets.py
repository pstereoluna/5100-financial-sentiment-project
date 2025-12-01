"""
Helper script to download modern post-2020 financial social media sentiment datasets.

Downloads:
- Twitter Financial News Sentiment (Zeroshot, 2023) from HuggingFace
- Financial Tweets Sentiment (TimKoornstra, 2023) from HuggingFace
- TweetFinSent (JP Morgan, 2022) from GitHub or HuggingFace

All files are saved to the data/ directory.
"""

import os
import sys
import argparse
import requests
from pathlib import Path

# Ensure data directory exists
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)


def download_huggingface_dataset(dataset_name: str, output_path: str, config: str = None):
    """
    Download a dataset from HuggingFace using the datasets library.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_path: Where to save the CSV file
        config: Optional dataset configuration name
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install it with: pip install datasets")
        return False
    
    print(f"Downloading {dataset_name} from HuggingFace...")
    
    try:
        if config:
            dataset = load_dataset(dataset_name, config)
        else:
            dataset = load_dataset(dataset_name)
        
        # Convert to pandas DataFrame
        if 'train' in dataset:
            df = dataset['train'].to_pandas()
        elif 'test' in dataset:
            df = dataset['test'].to_pandas()
        else:
            # Use the first split
            split_name = list(dataset.keys())[0]
            df = dataset[split_name].to_pandas()
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} samples to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        return False


def download_github_raw(url: str, output_path: str):
    """
    Download a file from GitHub raw content.
    
    Args:
        url: GitHub raw URL
        output_path: Where to save the file
    """
    print(f"Downloading from GitHub: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading from GitHub: {e}")
        return False


def download_twitter_financial():
    """Download Twitter Financial News Sentiment dataset."""
    dataset_name = "Zeroshot/twitter-financial-news-sentiment"
    # Check if train file exists first
    train_path = DATA_DIR / "twitter_financial_train.csv"
    output_path = DATA_DIR / "twitter_financial.csv"
    
    # If train file exists, use it
    if train_path.exists():
        print(f"✓ Found existing file: {train_path}")
        print("  Using twitter_financial_train.csv (no download needed)")
        return True
    
    if output_path.exists():
        print(f"⚠ File already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping...")
            return True
    
    return download_huggingface_dataset(dataset_name, str(output_path))


def download_financial_tweets_2023():
    """Download Financial Tweets Sentiment dataset."""
    dataset_name = "TimKoornstra/financial-tweets-sentiment"
    output_path = DATA_DIR / "financial_tweets_2023.csv"
    
    if output_path.exists():
        print(f"⚠ File already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping...")
            return True
    
    return download_huggingface_dataset(dataset_name, str(output_path))


def download_tweetfinsent():
    """Download TweetFinSent dataset from GitHub or HuggingFace."""
    output_path = DATA_DIR / "tweetfinsent.csv"
    
    if output_path.exists():
        print(f"⚠ File already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping...")
            return True
    
    # Try HuggingFace first
    dataset_name = "tweetfinsent"  # Update if actual HF name differs
    try:
        if download_huggingface_dataset(dataset_name, str(output_path)):
            return True
    except:
        pass
    
    # Fallback to GitHub (update URL if needed)
    github_urls = [
        "https://raw.githubusercontent.com/jpmorganchase/tweetfinsent/main/data/train.csv",
        "https://raw.githubusercontent.com/jpmorganchase/tweetfinsent/main/tweetfinsent.csv",
    ]
    
    for url in github_urls:
        try:
            if download_github_raw(url, str(output_path)):
                return True
        except:
            continue
    
    print("✗ Could not download TweetFinSent from HuggingFace or GitHub")
    print("  Please download manually from: https://github.com/jpmorganchase/tweetfinsent")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Download modern post-2020 financial social media sentiment datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['twitter_financial', 'financial_tweets_2023', 'tweetfinsent', 'all'],
        default='all',
        help='Dataset to download (default: all)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory to save datasets (default: data/)'
    )
    
    args = parser.parse_args()
    
    # Update data directory if specified
    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    DATA_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Modern Financial Social Media Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print()
    
    success_count = 0
    total_count = 0
    
    if args.dataset == 'all' or args.dataset == 'twitter_financial':
        total_count += 1
        if download_twitter_financial():
            success_count += 1
        print()
    
    if args.dataset == 'all' or args.dataset == 'financial_tweets_2023':
        total_count += 1
        if download_financial_tweets_2023():
            success_count += 1
        print()
    
    if args.dataset == 'all' or args.dataset == 'tweetfinsent':
        total_count += 1
        if download_tweetfinsent():
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} datasets downloaded")
    print("=" * 60)
    
    if success_count < total_count:
        print("\nNote: Some datasets may need manual download.")
        print("See data/DATASET_RECOMMENDATIONS.md for links.")


if __name__ == '__main__':
    main()

