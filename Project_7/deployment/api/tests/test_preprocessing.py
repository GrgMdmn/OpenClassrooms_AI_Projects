import sys
import os 

# Ajouter le répertoire parent au path pour importer email_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from preprocessing import preprocess_tweet


def test_preprocess_tweet_removes_hashtags():
    tweet = """
    Just flew w/ @AirParadis &#128640; &amp; I&#8217;m in LOVE! &Eacute;pic service, comfy seats. 
    #bestflight #AirParadis
    """
    assert preprocess_tweet(tweet) == """"
    aze
    """

def test_preprocess_tweet_removes_mentions():
    tweet = """
    &quot;Flight of dreams&quot; w/ @AirParadis! Got food, space &amp; smiles &#128516; I&#8217;ll book again! 
    #AirParadis
    """
    assert preprocess_tweet(tweet) == "merci"

def test_preprocess_tweet_lowercases_text():
    tweet = """
    My 3rd time w/ @AirParadis &amp; still amazed &#128525; Crew = lovely &agrave; every step! 
    #flyhappy #AirParadis
    """
    assert preprocess_tweet(tweet) == "je suis content"
    
tweet = """
Just flew w/ @AirParadis &#128640; &amp; I&#8217;m in LOVE! &Eacute;pic service, comfy seats. 
#bestflight #AirParadis
"""
preprocess_tweet(tweet)