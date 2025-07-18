import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure that the necessary resources are available
try:
    stopwords.words('english')
    word_tokenize("test")
    WordNetLemmatizer().lemmatize("testing")
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# PREPROCESSING FUNCTION FOR CLASSIC AND ADVANCED MODELS
def preprocess_tweet(tweet):
    """
    Preprocesses a tweet for classic ML models or word embedding models (e.g., Word2Vec, GloVe).
    The text is cleaned, normalized, lemmatized, and stripped of unnecessary noise.
    """

    # Ensure input is a string
    if not isinstance(tweet, str):
        return ""

    # Decode HTML entities (e.g., &amp; → &)
    tweet = html.unescape(tweet)
    # print(tweet + '\n')
    
    # Convert text to lowercase
    tweet = tweet.lower()
    # print(tweet + '\n')

    # Replace URLs with a generic token
    tweet = re.sub(r'https?://\S+|www\.\S+', '<URL>', tweet)
    # print(tweet + '\n')

    # Replace mentions (@username) with a generic token
    tweet = re.sub(r'@\w+', '<MENTION>', tweet)
    # print(tweet + '\n')

    # Separate hashtags from the word to preserve semantics during tokenization
    tweet = re.sub(r'#(\w+)', r'# \1', tweet)
    # print(tweet + '\n')

    # Remove unwanted punctuation and special characters (except useful ones like < > @ # ! ?)
    tweet = re.sub(r'[^\w\s<>@#!?]', '', tweet)
    # print(tweet + '\n')

    # Remove non-printable/control characters (e.g., \n, \t, emojis, corrupted unicode)
    tweet = re.sub(r'[^\x20-\x7E]', '', tweet)
    # print(tweet + '\n')

    # Tokenize the tweet into individual words
    tokens = word_tokenize(tweet)
    # print(tokens)
    # print('\n')

    # Lemmatize each token to its base form (e.g., "running" → "run")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # print(tokens)
    # print('\n')

    # Remove stopwords, but keep negation-related words (important for sentiment)
    stop_words = set(stopwords.words('english'))
    important_words = {'no', 'not', 'nor', 'neither', 'never', 'nobody', 'none', 'nothing', 'nowhere'}
    stop_words = stop_words - important_words
    tokens = [token for token in tokens if token not in stop_words]
    # print(tokens)
    # print('\n')


    # Reconstruct the cleaned tweet
    return ' '.join(tokens)