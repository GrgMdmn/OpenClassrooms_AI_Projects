import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Assurer que les ressources nécessaires sont disponibles
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

def preprocess_tweet(tweet):
    """
    Fonction de prétraitement des tweets pour l'API de sentiment analysis
    """
    if not isinstance(tweet, str):
        return ""
    tweet = html.unescape(tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+', '<URL>', tweet)
    tweet = re.sub(r'@\w+', '<MENTION>', tweet)
    tweet = re.sub(r'#(\w+)', r'# \1', tweet)
    tweet = re.sub(r'[^\w\s<>@#!?]', '', tweet)
    
    tokens = word_tokenize(tweet)
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    stop_words = set(stopwords.words('english'))
    important_words = {'no', 'not', 'nor', 'neither', 'never', 'nobody', 'none', 'nothing', 'nowhere'}
    stop_words = stop_words - important_words
    
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)