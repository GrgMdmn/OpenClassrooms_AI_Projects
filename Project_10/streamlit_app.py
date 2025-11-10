import streamlit as st
import requests
import json
import time

# Configuration de la page
st.set_page_config(
    page_title="Système de Recommandation",
    page_icon="🎯",
    layout="wide"
)

# Configuration de l'API
API_BASE_URL = "http://localhost:7071/api"

def call_recommendation_api(user_id, base_url=API_BASE_URL):
    """
    Appelle l'API Azure Function pour obtenir des recommandations.
    """
    try:
        url = f"{base_url}/recommend"
        params = {"user_id": user_id}
        
        start_time = time.time()
        response = requests.get(url, params=params, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return response.json(), response_time, None
        else:
            error_msg = f"Erreur {response.status_code}: {response.text}"
            return None, response_time, error_msg
            
    except requests.exceptions.Timeout:
        return None, None, "Timeout: L'API a mis trop de temps à répondre"
    except requests.exceptions.ConnectionError:
        return None, None, "Erreur de connexion: Vérifiez que l'Azure Function est démarrée"
    except Exception as e:
        return None, None, f"Erreur inattendue: {str(e)}"

def main():
    # Titre principal
    st.title("🎯 Système de Recommandation de Contenu")
    st.markdown("---")
    
    # Configuration dans la sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # URL de l'API
        api_url = st.text_input(
            "URL de l'API", 
            value=API_BASE_URL,
            help="URL de base de votre Azure Function"
        )
        
        # Test de connexion
        if st.button("🔍 Tester la connexion"):
            with st.spinner("Test en cours..."):
                try:
                    test_response = requests.get(f"{api_url}/recommend?user_id=0", timeout=5)
                    if test_response.status_code in [200, 400, 404]:  # API répond
                        st.success("✅ API accessible")
                    else:
                        st.error(f"❌ API répond avec le code {test_response.status_code}")
                except Exception as e:
                    st.error(f"❌ Impossible de joindre l'API: {str(e)}")
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Assurez-vous que `func start` est lancé")
        st.markdown("2. Entrez un ID utilisateur")
        st.markdown("3. Cliquez sur 'Obtenir des recommandations'")
    
    # Interface principale
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("👤 Sélection utilisateur")
        
        # Input pour l'ID utilisateur
        user_id = st.number_input(
            "ID de l'utilisateur",
            min_value=0,
            max_value=10000,
            value=0,
            step=1,
            help="Entrez l'ID de l'utilisateur pour lequel obtenir des recommandations"
        )
        
        # Bouton pour lancer la recommandation
        if st.button("🚀 Obtenir des recommandations", type="primary"):
            with st.spinner(f"Génération des recommandations pour l'utilisateur {user_id}..."):
                result, response_time, error = call_recommendation_api(user_id, api_url)
                
                # Stocker les résultats dans la session
                st.session_state['last_result'] = result
                st.session_state['last_response_time'] = response_time
                st.session_state['last_error'] = error
                st.session_state['last_user_id'] = user_id
    
    with col2:
        st.header("📊 Résultats")
        
        # Affichage des résultats
        if 'last_result' in st.session_state:
            if st.session_state['last_error']:
                st.error(f"❌ {st.session_state['last_error']}")
            elif st.session_state['last_result']:
                result = st.session_state['last_result']
                
                # Métriques
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("👤 Utilisateur", st.session_state['last_user_id'])
                with col_metrics2:
                    if st.session_state['last_response_time']:
                        st.metric("⏱️ Temps de réponse", f"{st.session_state['last_response_time']:.2f}s")
                
                # Recommandations
                if 'recommendations' in result:
                    st.subheader("🎯 Articles recommandés")
                    recommendations = result['recommendations']
                    
                    # Vérifier si on a le format enrichi (avec scores et catégories)
                    if recommendations and isinstance(recommendations[0], dict) and 'similarity_score_percent' in recommendations[0]:
                        # Format enrichi - affichage en tableau
                        import pandas as pd
                        
                        df_data = []
                        for i, rec in enumerate(recommendations, 1):
                            df_data.append({
                                'Rang': i,
                                'Article ID': rec['article_id'],
                                'Similarité (%)': f"{rec['similarity_score_percent']:.1f}%",
                                'Catégorie': rec.get('category_id', 'N/A')
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        # Format simple - affichage basique pour compatibilité
                        for i, article_id in enumerate(recommendations, 1):
                            if isinstance(article_id, dict):
                                article_id = article_id.get('article_id', article_id)
                            st.write(f"**{i}.** Article ID: `{article_id}`")
                    
                    # JSON brut (collapsible)
                    with st.expander("📝 Réponse JSON complète"):
                        st.json(result)
                else:
                    st.warning("⚠️ Aucune recommandation trouvée dans la réponse")
        else:
            st.info("👆 Sélectionnez un utilisateur et cliquez sur 'Obtenir des recommandations' pour commencer")
    
    # Section de test avancé
    st.markdown("---")
    st.header("🧪 Tests avancés")
    
    col_test1, col_test2 = st.columns(2)
    
    with col_test1:
        st.subheader("📈 Test de performance")
        if st.button("Tester avec plusieurs utilisateurs"):
            test_users = [0, 1, 2, 5, 10]
            progress_bar = st.progress(0)
            results = []
            
            for i, uid in enumerate(test_users):
                result, resp_time, error = call_recommendation_api(uid, api_url)
                results.append({
                    'user_id': uid,
                    'success': error is None,
                    'response_time': resp_time,
                    'error': error
                })
                progress_bar.progress((i + 1) / len(test_users))
            
            # Affichage des résultats
            success_count = sum(1 for r in results if r['success'])
            avg_time = sum(r['response_time'] for r in results if r['response_time']) / len([r for r in results if r['response_time']])
            
            st.success(f"✅ {success_count}/{len(test_users)} appels réussis")
            st.info(f"⏱️ Temps moyen: {avg_time:.2f}s")
    
    with col_test2:
        st.subheader("🔍 Test d'erreur")
        if st.button("Tester utilisateur inexistant"):
            with st.spinner("Test en cours..."):
                result, resp_time, error = call_recommendation_api(99999, api_url)
                if error:
                    st.warning(f"Erreur attendue: {error}")
                elif result:
                    st.info("Réponse reçue (comportement inattendu)")
                    st.json(result)

if __name__ == "__main__":
    main()