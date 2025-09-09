import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple
import logging

from .data_loader import load_embeddings, get_user_articles

def _calculate_mean_embedding(embeddings: np.ndarray, article_ids: List[int]) -> np.ndarray:
    """Calcule la moyenne des embeddings pour une liste d'articles."""
    valid_ids = [aid for aid in article_ids if aid < len(embeddings)]
    if not valid_ids:
        raise ValueError("No valid article IDs found")
    
    articles_embeddings = embeddings[valid_ids]
    return np.mean(articles_embeddings, axis=0).reshape(1, -1)

def _compute_similarities(reference_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Calcule la similarité cosinus entre l'embedding de référence et tous les embeddings."""
    return cosine_similarity(reference_embedding, embeddings)[0]

def _filter_candidates(similarities: np.ndarray, clicked_articles: List[int]) -> List[Tuple[int, float]]:
    """Filtre les candidats en excluant les articles déjà vus."""
    candidates = []
    for article_id, similarity in enumerate(similarities):
        if article_id not in clicked_articles:
            candidates.append((article_id, similarity))
    return candidates

def _get_top_recommendations(candidates: List[Tuple[int, float]], n_recommendations: int) -> List[int]:
    """Trie les candidats et retourne le top N."""
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [article_id for article_id, _ in candidates[:n_recommendations]]

def _validate_user_articles(clicked_articles: List[int], embeddings: np.ndarray) -> List[int]:
    """Valide et filtre les articles dans les limites des embeddings."""
    valid_articles = [art for art in clicked_articles if 0 <= art < len(embeddings)]
    if not valid_articles:
        raise ValueError("No valid articles found for user")
    return valid_articles

def get_recommendations(user_id: int, n_recommendations: int = 5) -> Optional[List[int]]:
    """
    Génère des recommandations pour un utilisateur donné.
    
    Args:
        user_id: ID de l'utilisateur
        n_recommendations: Nombre de recommandations à retourner
        
    Returns:
        Liste des article_ids recommandés ou None si erreur
    """
    try:
        # Charger les données
        embeddings = load_embeddings()
        clicked_articles = get_user_articles(user_id)
        
        if not clicked_articles:
            logging.warning(f"No articles found for user {user_id}")
            return None
        
        # Valider les articles
        valid_articles = _validate_user_articles(clicked_articles, embeddings)
        
        # Calculer l'embedding de référence (stratégie mean)
        reference_embedding = _calculate_mean_embedding(embeddings, valid_articles)
        
        # Calculer similarités
        similarities = _compute_similarities(reference_embedding, embeddings)
        
        # Filtrer et trier
        candidates = _filter_candidates(similarities, clicked_articles)
        recommendations = _get_top_recommendations(candidates, n_recommendations)
        
        logging.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
        
    except Exception as e:
        logging.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        return None