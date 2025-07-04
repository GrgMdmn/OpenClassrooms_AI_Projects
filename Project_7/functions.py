import re
import html
import numpy as np
import pandas as pd
from multiprocessing import Pool

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Assure que les ressources nécessaires sont disponibles
try:
    stopwords.words('english')
    word_tokenize("test")
    WordNetLemmatizer().lemmatize("testing")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Pour les lemmes WordNet en anglais

# Convertir les labels : 0 pour négatif, 1 pour tout le reste
def convert_sentiment_label(df):
    converted_target_data = df.copy()
    converted_target_data['target'] = converted_target_data['target'].apply(lambda x: 0 if x == 0 else 1)
    return converted_target_data

# Réduction de la taille du dataset pour équilibrer
def downsample_data(df, n_samples=50000, random_state=42):
    negative_samples = df[df['target'] == 0].sample(n=n_samples, random_state=random_state)
    positive_samples = df[df['target'] == 1].sample(n=n_samples, random_state=random_state)
    downsampled_data = pd.concat([negative_samples, positive_samples])
    return downsampled_data

# Fonction principale de prétraitement
def preprocess_tweet(tweet):
    if not isinstance(tweet, str):
        return ""

    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+', '<URL>', tweet)
    tweet = re.sub(r'@\w+', '<MENTION>', tweet)
    tweet = re.sub(r'#(\w+)', r'# \1', tweet)
    tweet = html.unescape(tweet)
    tweet = re.sub(r'[^\w\s<>@#!?]', '', tweet)

    tokens = word_tokenize(tweet)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stop_words = set(stopwords.words('english'))
    important_words = {'no', 'not', 'nor', 'neither', 'never', 'nobody', 'none', 'nothing', 'nowhere'}
    stop_words = stop_words - important_words
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

# Traitement en parallèle sur un DataFrame
def process_in_parallel(df, func, n_jobs=4):
    df_split = np.array_split(df, n_jobs)
    with Pool(n_jobs) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

# Application du prétraitement sur une partition
def apply_preprocessing(df_part):
    df_part['processed_text'] = df_part['text'].apply(preprocess_tweet)
    return df_part
