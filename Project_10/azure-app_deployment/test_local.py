import logging
import sys
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ajouter le dossier utils au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.content_based import get_recommendations
from utils.data_loader import get_available_users, check_data_files, load_embeddings

def test_data_loading():
    """Test du chargement des données."""
    print("🔄 Test du chargement des données...")
    
    if not check_data_files():
        print("❌ Fichiers de données manquants")
        return False
    
    try:
        # Test chargement embeddings
        embeddings = load_embeddings()
        print(f"✅ Embeddings chargés: {embeddings.shape}")
        
        # Test chargement utilisateurs
        users = get_available_users()
        print(f"✅ Utilisateurs chargés: {len(users)} utilisateurs")
        print(f"   Premiers utilisateurs: {users[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False

def test_recommendations():
    """Test des recommandations."""
    print("\n🔄 Test des recommandations...")
    
    users = get_available_users()
    test_users = users[:3] if len(users) >= 3 else users
    
    for user_id in test_users:
        print(f"\n👤 Test pour utilisateur {user_id}:")
        
        try:
            recommendations = get_recommendations(user_id, n_recommendations=5)
            
            if recommendations:
                print(f"✅ {len(recommendations)} recommandations générées")
                print(f"   Articles recommandés: {recommendations}")
            else:
                print("⚠️  Aucune recommandation générée")
                
        except Exception as e:
            print(f"❌ Erreur pour utilisateur {user_id}: {e}")

def test_edge_cases():
    """Test des cas limites."""
    print("\n🔄 Test des cas limites...")
    
    # Test utilisateur inexistant
    print("Test utilisateur inexistant (ID: 99999):")
    result = get_recommendations(99999)
    if result is None:
        print("✅ Utilisateur inexistant géré correctement")
    else:
        print(f"⚠️  Résultat inattendu: {result}")

def main():
    """Fonction principale de test."""
    print("🚀 Démarrage des tests locaux\n")
    
    # Test 1: Chargement des données
    if not test_data_loading():
        print("\n❌ Tests interrompus - problème de chargement")
        return
    
    # Test 2: Recommandations
    test_recommendations()
    
    # Test 3: Cas limites
    test_edge_cases()
    
    print("\n✨ Tests terminés")

if __name__ == "__main__":
    main()