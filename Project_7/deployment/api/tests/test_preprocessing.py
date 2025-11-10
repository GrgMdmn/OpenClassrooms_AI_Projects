import os
import sys

# Ajouter le répertoire parent au path pour importer email_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from preprocessing import preprocess_tweet


def test_preprocess_tweet_removes_hashtags():
    tweet = """
    Just flew w/ @AirParadis &#128640; &amp; I&#8217;m in LOVE! Epic service, comfy seats, good prices at https://airparadis.com
    #bestflight #AirParadis
    """
    assert preprocess_tweet(tweet) == 'flew w < MENTION > im love ! epic service comfy seat good price < URL > # bestflight # airparadis'

def test_preprocess_tweet_removes_mentions():
    tweet = """
    &quot;Flight of dreams&quot; w/ @AirParadis! Got food, space &amp; smiles &#128516; I&#8217;ll book again! 
    #AirParadis
    """
    assert preprocess_tweet(tweet) == 'flight dream w < MENTION > ! got food space smile ill book ! # airparadis'

def test_preprocess_tweet_lowercases_text():
    tweet = """
    My 3rd time w/ @AirParadis &amp; still amazed &#128525; Crew = lovely &agrave; every step! 
    #flyhappy #AirParadis
    """
    assert preprocess_tweet(tweet) == '3rd time w < MENTION > still amazed crew lovely every step ! # flyhappy # airparadis'
