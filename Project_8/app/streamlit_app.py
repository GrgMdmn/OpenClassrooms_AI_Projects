import os
import sys
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import pickle

# Configuration de l'API
# prod docker
# API_BASE_URL = os.getenv("MULTISEG_API_BASE_URL", "http://127.0.0.1:8080/api")
# test local
API_BASE_URL = os.getenv("MULTISEG_API_BASE_URL", "http://127.0.0.1:8000")  # ← Port 8000 au lieu de 8080

# Import des fonctions utils pour l'affichage local
sys.path.append('/app')
# APRÈS (pour test local)
sys.path.append('.')
from utils.utils import colorize_mask

# Configuration de la page
st.set_page_config(
    page_title="Multi-Class Segmentation",
    page_icon="🚗",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
    color: #2c3e50;
    font-weight: 500;
}
.prediction-stats {
    background: #e8f4fd;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🚗 Segmentation Multi-Classes</h1>
    <p>Système de segmentation sémantique pour véhicules autonomes</p>
</div>
""", unsafe_allow_html=True)

# Initialisation des états
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

def reset_state():
    """Reset des états de session"""
    st.session_state.prediction_result = None
    st.session_state.selected_image = None
    st.session_state.uploaded_file = None

def get_model_info():
    """Récupère les informations du modèle"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des infos modèle: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def get_sample_images():
    """Récupère la liste des images d'exemple"""
    try:
        response = requests.get(f"{API_BASE_URL}/sample-images")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des images: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def predict_sample_image(filename):
    """Lance une prédiction sur une image d'exemple"""
    try:
        # CORRECTION: Envoyer en JSON au lieu de form data
        data = {"filename": filename}
        response = requests.post(
            f"{API_BASE_URL}/predict-sample", 
            json=data,  # ← Changé de 'data=' à 'json='
            headers={"Content-Type": "application/json"}  # ← Ajout du header explicite
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None
    
def predict_uploaded_image(uploaded_file):
    """Lance une prédiction sur une image uploadée"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/predict-upload", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None

def display_segmentation_plots(fig):
    """Affiche chaque subplot séparément dans des colonnes"""
    if fig is None:
        st.warning("Aucune figure de segmentation disponible")
        return
    
    axes = fig.get_axes()
    num_plots = len(axes)
    
    st.markdown("### 🎨 Résultat de la Segmentation")
    
    # Créer les colonnes selon le nombre de subplots
    if num_plots <= 4:
        cols = st.columns(num_plots)
    else:
        cols = st.columns(4)  # Maximum 4 colonnes
    
    for i, ax in enumerate(axes):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Récupérer le titre existant
            title = ax.get_title()
            # st.markdown(f"**{title}:**")
            
            # Créer une figure individuelle avec ce subplot
            individual_fig, individual_ax = plt.subplots(figsize=(6, 6))
            
            # Copier toutes les images de l'axe original
            for image in ax.get_images():
                individual_ax.imshow(
                    image.get_array(),
                    cmap=image.get_cmap() if image.get_cmap() else None,
                    alpha=image.get_alpha() if image.get_alpha() else None,
                    vmin=image.get_clim()[0] if image.get_clim() else None,
                    vmax=image.get_clim()[1] if image.get_clim() else None
                )
            
            # Copier les propriétés de l'axe
            individual_ax.set_title(title)
            individual_ax.axis('off')
            
            # Ajuster la mise en page
            individual_fig.tight_layout()
            
            # Afficher dans Streamlit
            st.pyplot(individual_fig)
            
            # Libérer la mémoire
            plt.close(individual_fig)
    
    # Fermer la figure originale pour libérer la mémoire
    plt.close(fig)

        
def display_prediction_stats_with_chart(stats, model_info):
    """Affiche les statistiques avec un graphique en barres"""
    st.markdown("### 📊 Statistiques de Segmentation")
    
    if not stats:
        st.warning("Aucune statistique disponible")
        return
    
    # Ratio 1:2 pour donner plus de place au graphique
    col1, col2 = st.columns([1, 2])
    
    # Colonne 1: Cartes de statistiques compactes avec carrés colorés
    with col1:
        st.markdown("**Détails par classe:**")
        
        # Récupérer les couleurs des classes pour les carrés
        class_colors_dict = {}
        if model_info and 'class_colors' in model_info and 'class_names' in model_info:
            model_class_names = model_info['class_names']
            model_class_colors = model_info['class_colors']
            
            for class_name in stats.keys():
                try:
                    idx = model_class_names.index(class_name)
                    color = model_class_colors[idx]
                    # Convertir en hex pour CSS
                    color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    class_colors_dict[class_name] = color_hex
                except (ValueError, IndexError):
                    # Couleur par défaut si pas trouvée
                    class_colors_dict[class_name] = "#808080"
        
        for class_name, data in stats.items():
            # Récupérer la couleur pour cette classe
            color_hex = class_colors_dict.get(class_name, "#808080")
            
            # ✨ NOUVEAU: Carré coloré + nom de classe
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 0.5rem 0.8rem;
                border-radius: 6px;
                border-left: 3px solid #667eea;
                margin: 0.3rem 0;
                color: #2c3e50;
                font-weight: 500;
                font-size: 0.85rem;
                line-height: 1.3;
            ">
                <div style="
                    font-weight: 600; 
                    margin-bottom: 0.2rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                ">
                    <div style="
                        width: 12px;
                        height: 12px;
                        background-color: {color_hex};
                        border-radius: 2px;
                        border: 1px solid rgba(0,0,0,0.2);
                        flex-shrink: 0;
                    "></div>
                    <span>{class_name}</span>
                </div>
                <div style="font-size: 0.8rem; color: #555;">
                    🔢 {data['pixels']:,} pixels | 📊 {data['percentage']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Colonne 2: Graphique en barres (reste identique)
    with col2:
        st.markdown("**Répartition visuelle:**")
        
        # Préparer les données pour le graphique
        class_names = list(stats.keys())
        percentages = [data['percentage'] for data in stats.values()]
        
        # Récupérer les couleurs des classes depuis model_info
        colors = []
        if model_info and 'class_colors' in model_info and 'class_names' in model_info:
            model_class_names = model_info['class_names']
            model_class_colors = model_info['class_colors']
            
            for class_name in class_names:
                # Trouver l'index de cette classe dans le modèle
                try:
                    idx = model_class_names.index(class_name)
                    color = model_class_colors[idx]
                    # Convertir RGB en format matplotlib
                    colors.append([c/255.0 for c in color])
                except (ValueError, IndexError):
                    # Couleur par défaut si pas trouvée
                    colors.append([0.5, 0.5, 0.5])
        else:
            # Couleurs par défaut si pas d'info du modèle
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        # Créer un graphique plus grand et mieux adapté
        fig, ax = plt.subplots(figsize=(10, 6))  # Plus large
        
        bars = ax.bar(class_names, percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Personnalisation du graphique améliorée
        ax.set_xlabel('Classes de Segmentation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pourcentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Répartition des Classes Détectées', fontsize=13, fontweight='bold', pad=20)
        
        # Meilleure gestion des labels selon le nombre de classes
        if len(class_names) <= 4:
            plt.xticks(rotation=0, ha='center', fontsize=10)
        elif len(class_names) <= 6:
            plt.xticks(rotation=30, ha='right', fontsize=9)
        else:
            plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.yticks(fontsize=10)
        
        # Valeurs sur les barres avec meilleur positionnement
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            # Adapter la position du texte selon la hauteur de la barre
            if height < max(percentages) * 0.1:
                # Si la barre est très petite, mettre le texte au-dessus
                va = 'bottom'
                y_pos = height + max(percentages) * 0.02
            else:
                # Sinon, centrer dans la barre
                va = 'center'
                y_pos = height / 2
            
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{percentage:.1f}%',
                   ha='center', va=va, fontsize=9, fontweight='bold',
                   color='white' if va == 'center' else 'black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7) if va == 'center' else None)
        
        # Améliorer la mise en page et le style
        plt.tight_layout()
        
        # Grille plus subtile
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        
        # Limites et style des axes
        ax.set_ylim(0, max(percentages) * 1.15)  # Plus d'espace au-dessus
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666')
        ax.spines['bottom'].set_color('#666')
        
        # Couleur de fond
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        
        # Afficher dans Streamlit
        st.pyplot(fig)
        plt.close(fig)       


def display_class_legend(model_info):
    """Affiche la légende des classes de segmentation"""
    if model_info and 'class_colors' in model_info:
        st.markdown("**Légende des classes:**")
        
        class_names = model_info.get('class_names', [])
        class_colors = model_info.get('class_colors', [])
        
        # Créer une légende colorée
        legend_cols = st.columns(len(class_names))
        for i, (name, color) in enumerate(zip(class_names, class_colors)):
            with legend_cols[i]:
                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                st.markdown(f"""
                <div style="
                    background-color: {color_hex}; 
                    padding: 10px; 
                    border-radius: 5px; 
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
                    margin: 2px;
                ">
                    {name}
                </div>
                """, unsafe_allow_html=True)

# Sidebar avec informations du modèle
with st.sidebar:
    st.markdown("## 🔧 Informations du Modèle")
    
    model_info = get_model_info()
    if model_info:
        st.success("✅ Modèle chargé")
        st.write(f"**Nom:** {model_info.get('model_name', 'N/A')}")
        st.write(f"**Encoder:** {model_info.get('encoder', 'N/A')}")
        st.write(f"**Classes:** {model_info.get('num_classes', 'N/A')}")
        st.write(f"**Taille d'entrée:** {model_info.get('input_size', 'N/A')}")
        st.write(f"**Run ID:** {model_info.get('run_id', 'N/A')[:8]}...")
    else:
        st.error("❌ Modèle non disponible")

    st.markdown("---")
    st.markdown("## 🎯 Classes Segmentées")
    if model_info and 'class_names' in model_info:
        for i, class_name in enumerate(model_info['class_names']):
            st.write(f"{i}. {class_name}")


# Initialisation des états - MODIFIÉE pour séparer les résultats par onglet
if "sample_prediction_result" not in st.session_state:
    st.session_state.sample_prediction_result = None
if "sample_selected_image" not in st.session_state:
    st.session_state.sample_selected_image = None

if "upload_prediction_result" not in st.session_state:
    st.session_state.upload_prediction_result = None
if "upload_uploaded_file" not in st.session_state:
    st.session_state.upload_uploaded_file = None

# ANCIEN système (gardé pour compatibilité si besoin)
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

def reset_sample_state():
    """Reset des états de session pour l'onglet Sample"""
    st.session_state.sample_prediction_result = None
    st.session_state.sample_selected_image = None

def reset_upload_state():
    """Reset des états de session pour l'onglet Upload"""
    st.session_state.upload_prediction_result = None
    st.session_state.upload_uploaded_file = None

def reset_state():
    """Reset complet (garde l'ancienne fonction pour compatibilité)"""
    reset_sample_state()
    reset_upload_state()

# Interface principale - MODIFIÉE pour résultats indépendants
tab1, tab2 = st.tabs(["📸 Images d'Exemple", "📤 Upload Personnel"])

# Tab 1: Images d'exemple
with tab1:
    st.markdown("## 📸 Sélectionner une Image d'Exemple")
    st.markdown("Ces images proviennent du dataset Cityscapes avec masques de vérité disponibles.")
    
    # Récupérer les images d'exemple
    sample_data = get_sample_images()
    if sample_data and sample_data['images']:
        # Sélecteur d'image
        image_options = {img['display_name']: img['filename'] for img in sample_data['images']}
        selected_display_name = st.selectbox(
            "Choisissez une image:",
            options=list(image_options.keys()),
            help=f"{sample_data['total_count']} images disponibles"
        )
        
        selected_filename = image_options[selected_display_name]
        
        # Afficher les infos de l'image sélectionnée
        selected_img_info = next(img for img in sample_data['images'] if img['filename'] == selected_filename)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📁 **Fichier:** {selected_filename}")
        with col2:
            ground_truth_status = "✅ Disponible" if selected_img_info['has_ground_truth'] else "❌ Non disponible"
            st.info(f"🎯 **Vérité terrain:** {ground_truth_status}")
        
        # Prévisualisation de l'image sélectionnée
        if selected_filename:
            try:
                # Récupérer l'image via l'API
                image_url = f"{API_BASE_URL}/sample-image/{selected_filename}"
                response = requests.get(image_url, timeout=5)
                
                if response.status_code == 200:
                    # Charger l'image depuis les bytes
                    sample_image = Image.open(io.BytesIO(response.content))
                    st.image(
                        sample_image, 
                        caption=f"Aperçu: {selected_display_name}", 
                        use_container_width=True
                    )
                elif response.status_code == 404:
                    st.warning("📷 Image non trouvée sur le serveur")
                else:
                    st.info("📷 Prévisualisation temporairement indisponible")
                    
            except requests.exceptions.Timeout:
                st.warning("⏱️ Timeout lors du chargement de l'aperçu")
            except Exception as e:
                st.info("📷 Aperçu non disponible - L'image sera chargée lors de la prédiction")
        
        # Bouton de prédiction
        if st.button("🔮 Lancer la Segmentation", key="predict_sample", type="primary"):
            if selected_filename:
                with st.spinner("Segmentation en cours..."):
                    result = predict_sample_image(selected_filename)
                    if result and result['success']:
                        # ✨ NOUVEAU: Stocker dans les variables Sample
                        st.session_state.sample_prediction_result = result
                        st.session_state.sample_selected_image = selected_filename
                        st.success("✅ Segmentation terminée!")
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors de la segmentation")
    else:
        st.warning("Aucune image d'exemple disponible")

    # ✨ NOUVEAU: Affichage des résultats Sample (indépendants)
    if st.session_state.sample_prediction_result and st.session_state.sample_selected_image:
        st.markdown("---")
        result = st.session_state.sample_prediction_result
        
        st.markdown(f"## 🎯 Résultats pour: {st.session_state.sample_selected_image}")
        
        # Afficher la segmentation
        if result.get('figure_data'):
            try:
                # Désérialiser la figure
                fig_bytes = base64.b64decode(result['figure_data'])
                fig = pickle.loads(fig_bytes)
                display_segmentation_plots(fig)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage de la segmentation: {e}")
                # Fallback sur la légende des classes
                display_class_legend(model_info)
        else:
            # Fallback si pas de figure_data
            display_class_legend(model_info)
        
        # Afficher les statistiques
        if result.get('prediction_stats'):
            display_prediction_stats_with_chart(result['prediction_stats'], model_info)
        
        # Informations additionnelles
        if result.get('ground_truth_available'):
            st.success("🎯 Masque de vérité terrain disponible pour comparaison")
        else:
            st.info("ℹ️ Pas de vérité terrain disponible")
        
        # Bouton de reset
        if st.button("🔄 Nouvelle Analyse", key="reset_sample", type="secondary"):
            reset_sample_state()
            st.rerun()

# Tab 2: Upload personnel
with tab2:
    st.markdown("## 📤 Uploader Votre Image")
    st.markdown("Uploadez une image de rue pour analyse. Formats acceptés: JPG, PNG")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une image:",
        type=['jpg', 'jpeg', 'png'],
        help="Taille maximale recommandée: 5MB"
    )
    
    if uploaded_file is not None:
        # Afficher l'image uploadée
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Image uploadée: {uploaded_file.name}", use_container_width=True)
        
        # Informations sur le fichier
        file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        st.info(f"📁 **Fichier:** {uploaded_file.name} | 📊 **Taille:** {file_size_mb:.2f} MB")
        
        # Bouton de prédiction
        if st.button("🔮 Lancer la Segmentation", key="predict_upload", type="primary"):
            with st.spinner("Segmentation en cours..."):
                result = predict_uploaded_image(uploaded_file)
                if result and result['success']:
                    # ✨ NOUVEAU: Stocker dans les variables Upload
                    st.session_state.upload_prediction_result = result
                    st.session_state.upload_uploaded_file = uploaded_file.name
                    st.success("✅ Segmentation terminée!")
                    st.rerun()
                else:
                    st.error("❌ Erreur lors de la segmentation")

    # ✨ NOUVEAU: Affichage des résultats Upload (indépendants)
    if st.session_state.upload_prediction_result and st.session_state.upload_uploaded_file:
        st.markdown("---")
        result = st.session_state.upload_prediction_result
        
        st.markdown(f"## 🎯 Résultats pour: {st.session_state.upload_uploaded_file}")
        
        # Afficher la segmentation
        if result.get('figure_data'):
            try:
                # Désérialiser la figure
                fig_bytes = base64.b64decode(result['figure_data'])
                fig = pickle.loads(fig_bytes)
                display_segmentation_plots(fig)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage de la segmentation: {e}")
                # Fallback sur la légende des classes
                display_class_legend(model_info)
        else:
            # Fallback si pas de figure_data
            display_class_legend(model_info)
        
        # Afficher les statistiques
        if result.get('prediction_stats'):
            display_prediction_stats_with_chart(result['prediction_stats'], model_info)
        
        # Informations additionnelles
        if result.get('ground_truth_available'):
            st.success("🎯 Masque de vérité terrain disponible pour comparaison")
        else:
            st.info("ℹ️ Pas de vérité terrain disponible (image personnelle)")
        
        # Bouton de reset
        if st.button("🔄 Nouvelle Analyse", key="reset_upload", type="secondary"):
            reset_upload_state()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚗 <strong>Multi-Class Segmentation API</strong> - 
    Système de vision par ordinateur pour véhicules autonomes<br>
    <small>Développé avec FastAPI, Streamlit et TensorFlow</small>
</div>
""", unsafe_allow_html=True)