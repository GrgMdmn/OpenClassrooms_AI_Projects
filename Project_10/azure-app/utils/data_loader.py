import pickle
import pandas as pd
import os
import logging
from typing import Dict, List, Optional
import numpy as np

# Chemins des fichiers
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings_azure_50d.pkl')
CLICKS_PATH = os.path.join(DATA_DIR, 'clicks_sample.csv')
ARTICLES_METADATA_PATH = os.path.join(DATA_DIR, 'articles_metadata.csv')

# Cache global
_embeddings_cache: Optional[np.ndarray] = None
_user_data_cache: Optional[Dict[int, List[int]]] = None
_articles_metadata_cache: Optional[pd.DataFrame] = None

def _load_pickle_file(file_path: str) -> np.ndarray:
    """Charge un fichier pickle."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def _load_csv_file(file_path: str) -> pd.DataFrame:
    """Charge un fichier CSV."""
    return pd.read_csv(file_path)

def _process_clicks_data(clicks_df: pd.DataFrame) -> Dict[int, List[int]]:
    """Convertit le DataFrame en dictionnaire user->articles."""
    user_data = {}
    for _, row in clicks_df.iterrows():
        user_id = int(row['user_id'])
        article_id = int(row['click_article_id'])
        
        if user_id not in user_data:
            user_data[user_id] = []
        user_data[user_id].append(article_id)
    
    return user_data

def load_embeddings() -> np.ndarray:
    """Charge les embeddings avec cache."""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        try:
            _embeddings_cache = _load_pickle_file(EMBEDDINGS_PATH)
            logging.info(f"Embeddings loaded: {_embeddings_cache.shape}")
        except Exception as e:
            logging.error(f"Error loading embeddings: {str(e)}")
            raise
    
    return _embeddings_cache

def load_user_data() -> Dict[int, List[int]]:
    """Charge les données utilisateur avec cache."""
    global _user_data_cache
    
    if _user_data_cache is None:
        try:
            clicks_df = _load_csv_file(CLICKS_PATH)
            _user_data_cache = _process_clicks_data(clicks_df)
            logging.info(f"User data loaded: {len(_user_data_cache)} users")
        except Exception as e:
            logging.error(f"Error loading user data: {str(e)}")
            raise
    
    return _user_data_cache

def load_articles_metadata() -> pd.DataFrame:
    """Charge les métadonnées des articles avec cache."""
    global _articles_metadata_cache
    
    if _articles_metadata_cache is None:
        try:
            _articles_metadata_cache = _load_csv_file(ARTICLES_METADATA_PATH)
            logging.info(f"Articles metadata loaded: {len(_articles_metadata_cache)} articles")
        except Exception as e:
            logging.error(f"Error loading articles metadata: {str(e)}")
            raise
    
    return _articles_metadata_cache

def get_user_articles(user_id: int) -> Optional[List[int]]:
    """Récupère les articles d'un utilisateur."""
    user_data = load_user_data()
    return user_data.get(user_id)

def get_article_category(article_id: int) -> Optional[int]:
    """Récupère la catégorie d'un article."""
    try:
        metadata_df = load_articles_metadata()
        article_row = metadata_df[metadata_df['article_id'] == article_id]
        if not article_row.empty:
            return int(article_row.iloc[0]['category_id'])
        return None
    except Exception as e:
        logging.warning(f"Error getting category for article {article_id}: {str(e)}")
        return None

def get_articles_categories(article_ids: List[int]) -> Dict[int, Optional[int]]:
    """Récupère les catégories de plusieurs articles."""
    try:
        metadata_df = load_articles_metadata()
        categories = {}
        for article_id in article_ids:
            article_row = metadata_df[metadata_df['article_id'] == article_id]
            if not article_row.empty:
                categories[article_id] = int(article_row.iloc[0]['category_id'])
            else:
                categories[article_id] = None
        return categories
    except Exception as e:
        logging.warning(f"Error getting categories for articles: {str(e)}")
        return {aid: None for aid in article_ids}

def check_data_files() -> bool:
    """Vérifie que tous les fichiers de données existent."""
    files_to_check = [EMBEDDINGS_PATH, CLICKS_PATH, ARTICLES_METADATA_PATH]
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            logging.error(f"Missing data file: {file_path}")
            return False
    
    return True

def get_available_users() -> List[int]:
    """Retourne la liste de tous les user_ids disponibles."""
    user_data = load_user_data()
    return list(user_data.keys())