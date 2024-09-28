import logging
import pandas as pd
import re
import string
import torch
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from transformers import RobertaTokenizer
import nltk

nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, train_file, test_file=None, tokenizer_model='roberta-base', max_length=128):
        """
        Initialize the DataProcessor with tokenizer, lemmatizer, and file paths.

        Args:
            train_file (str): Path to the training dataset.
            test_file (str): Path to the test dataset (optional).
            tokenizer_model (str): Pre-trained tokenizer model to use (default: 'roberta-base').
            max_length (int): Maximum token length for tokenization (default: 128).
        """
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length
        self.lemmatizer = WordNetLemmatizer()
        logging.info(f"Initialized DataProcessor with train_file: {train_file}, test_file: {test_file}, tokenizer_model: {tokenizer_model}")

    def load_data(self, file_path):
        """
        Load the dataset from a CSV file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape {df.shape}")
        return df

    def clean_tweet(self, tweet):
        """
        Clean the tweet text by removing URLs, mentions, hashtags, and special characters.

        Args:
            tweet (str): The raw tweet text.

        Returns:
            str: The cleaned tweet text.
        """
        logging.debug(f"Cleaning tweet: {tweet}")
        # Convert to lowercase
        tweet = tweet.lower()
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        # Remove mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\w+', '', tweet)
        # Remove special characters and punctuation
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        # Lemmatization
        tweet = ' '.join([self.lemmatizer.lemmatize(word) for word in tweet.split()])
        logging.debug(f"Cleaned tweet: {tweet}")
        return tweet

    def preprocess_data(self, df, test_data=False):
        """
        Preprocess the dataset by cleaning tweets and tokenizing.

        Args:
            df (pd.DataFrame): DataFrame containing the tweets to be processed.
            test_data (bool): Flag to indicate whether the data is test data (default: False).

        Returns:
            List[Dict]: Tokenized dataset.
        """
        logging.info("Starting data preprocessing.")
        df['cleaned_tweet'] = df['tweet'].apply(self.clean_tweet)

        tokens = df['cleaned_tweet'].apply(
            lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        )

        tokenized_data = []
        for index, row in df.iterrows():
            input_ids = tokens[index]['input_ids'].squeeze(0)  # Remove the batch dimension
            attention_mask = tokens[index]['attention_mask'].squeeze(0)

            if test_data:
                tokenized_data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
            else:
                label = row['label']
                tokenized_data.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(label, dtype=torch.long)
                })

        logging.info(f"Data preprocessing complete. Processed {len(tokenized_data)} samples.")
        return tokenized_data

    def prepare_train_val_data(self):
        """
        Load, clean, and split the training dataset into training and validation sets.

        Returns:
            Tuple[List, List]: Training and validation tokenized datasets.
        """
        logging.info("Preparing training and validation data.")
        train_df = self.load_data(self.train_file)
        tokenized_data = self.preprocess_data(train_df)

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(tokenized_data, test_size=0.2, random_state=42)
        logging.info(f"Training and validation split complete: {len(train_data)} training samples, {len(val_data)} validation samples.")
        return train_data, val_data

    def prepare_test_data(self):
        """
        Load, clean, and tokenize the test dataset.

        Returns:
            List[Dict]: Tokenized test dataset.
        """
        if self.test_file:
            logging.info("Preparing test data.")
            test_df = self.load_data(self.test_file)
            tokenized_data = self.preprocess_data(test_df, test_data=True)
            logging.info(f"Test data preparation complete: {len(tokenized_data)} samples.")
            return tokenized_data
        else:
            logging.error("Test file not provided. Unable to prepare test data.")
            raise ValueError("Test file not provided")

# Usage
# processor = DataProcessor(train_file='data/train.csv', test_file='data/test.csv')
# train_data, val_data = processor.prepare_train_val_data()
# test_data = processor.prepare_test_data()
