import re
import html
import numpy as np
import pandas as pd
from multiprocessing import Pool

# NLTK imports for NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded and available
try:
    stopwords.words('english')
    word_tokenize("test")
    WordNetLemmatizer().lemmatize("testing")
except LookupError:
    nltk.download('punkt')         # Tokenizer models
    nltk.download('punkt_tab')     # Additional tokenizer support
    nltk.download('stopwords')     # Stopwords lists
    nltk.download('wordnet')       # WordNet lexicon for lemmatization
    nltk.download('omw-1.4')       # WordNet lemmas data

# Convert sentiment labels to binary: 0 for negative, 1 for all other classes
def convert_sentiment_label(df):
    """
    Convert the 'target' column in the DataFrame to binary sentiment labels:
    - 0 for negative sentiment (label 0)
    - 1 for any other sentiment label
    Returns a new DataFrame with updated labels.
    """
    converted_target_data = df.copy()
    converted_target_data['target'] = converted_target_data['target'].apply(lambda x: 0 if x == 0 else 1)
    return converted_target_data

# Downsample the dataset to balance classes by limiting samples per class
def downsample_data(df, n_samples=50000, random_state=42):
    """
    Perform downsampling to balance dataset classes.
    Selects `n_samples` examples from both negative (0) and positive (1) classes.
    Returns a balanced DataFrame with equal number of samples per class.
    """
    negative_samples = df[df['target'] == 0].sample(n=n_samples, random_state=random_state)
    positive_samples = df[df['target'] == 1].sample(n=n_samples, random_state=random_state)
    downsampled_data = pd.concat([negative_samples, positive_samples])
    return downsampled_data

# PREPROCESSING FUNCTION FOR CLASSIC AND ADVANCED MODELS
def preprocess_tweet(tweet):
    """
    Preprocess a tweet for classic ML or word embedding models (e.g., Word2Vec, GloVe).
    Steps:
    - Normalize text (lowercase)
    - Replace URLs and mentions with tokens
    - Decode HTML entities
    - Remove unwanted special characters and non-printable chars
    - Tokenize text
    - Lemmatize tokens
    - Remove stopwords except negation words
    Returns cleaned, normalized text as a single string.
    """

    # Ensure input is a string
    if not isinstance(tweet, str):
        return ""

    # Convert to lowercase
    tweet = tweet.lower()

    # Replace URLs with placeholder token
    tweet = re.sub(r'https?://\S+|www\.\S+', '<URL>', tweet)

    # Replace user mentions with placeholder token
    tweet = re.sub(r'@\w+', '<MENTION>', tweet)

    # Separate hashtags from words to facilitate tokenization
    tweet = re.sub(r'#(\w+)', r'# \1', tweet)

    # Decode HTML entities (e.g., &amp; to &)
    tweet = html.unescape(tweet)

    # Remove unwanted punctuation and special characters except some useful ones
    tweet = re.sub(r'[^\w\s<>@#!?]', '', tweet)

    # Remove non-printable and control characters (e.g., tabs, newlines, emojis)
    tweet = re.sub(r'[^\x20-\x7E]', '', tweet)

    # Tokenize tweet into words
    tokens = word_tokenize(tweet)

    # Lemmatize tokens to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords but keep negation-related words to preserve sentiment meaning
    stop_words = set(stopwords.words('english'))
    important_words = {'no', 'not', 'nor', 'neither', 'never', 'nobody', 'none', 'nothing', 'nowhere'}
    stop_words = stop_words - important_words
    tokens = [token for token in tokens if token not in stop_words]

    # Rebuild cleaned tweet text
    return ' '.join(tokens)

# Parallel processing of a DataFrame using multiple CPU cores
def process_in_parallel(df, func, n_jobs=4):
    """
    Split a DataFrame into `n_jobs` parts and apply `func` to each part in parallel,
    using multiprocessing Pool for speed-up.
    Returns the concatenated processed DataFrame.
    """
    df_split = np.array_split(df, n_jobs)
    with Pool(n_jobs) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

# Apply preprocessing function on a DataFrame partition
def apply_preprocessing(df_part):
    """
    Apply tweet preprocessing to the 'text' column of a DataFrame partition.
    Stores results in a new column 'processed_text'.
    Returns the updated DataFrame partition.
    """
    df_part['processed_text'] = df_part['text'].apply(preprocess_tweet)
    return df_part
