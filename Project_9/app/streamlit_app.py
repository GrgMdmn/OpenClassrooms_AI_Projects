import os
import sys
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import pickle

# Configuration de l'API
API_BASE_URL = os.getenv("MULTISEG_API_BASE_URL", "http://127.0.0.1:8080/api")

# Import des fonctions utils pour l'affichage local
sys.path.append('/app')
sys.path.append('.')
from utils.utils import colorize_mask

# Configuration de la page
st.set_page_config(
    page_title="Multi-Class Segmentation Comparison",
    page_icon="🔄",
    layout="wide"
)

# CSS personnalisé mis à jour
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
.comparison-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    color: #2c3e50;
    font-weight: 500;
}
.speed-winner {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left-color: #28a745;
}
.speed-slower {
    background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
    border-left-color: #dc3545;
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
.color-box {
    display: inline-block;
    width: 14px;
    height: 14px;
    margin-right: 8px;
    border-radius: 2px;
    border: 1px solid rgba(0,0,0,0.2);
    vertical-align: middle;
}
.comparison-table {
    margin: 1rem 0;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.comparison-table th {
    background-color: #667eea !important;
    color: white !important;
    font-weight: 600 !important;
    text-align: center !important;
}
.comparison-table td {
    text-align: center !important;
    padding: 0.5rem !important;
    vertical-align: middle !important;
}
.diff-positive {
    background-color: #d4edda !important;
    color: #155724 !important;
    font-weight: 600;
}
.diff-negative {
    background-color: #f8d7da !important;
    color: #721c24 !important;
    font-weight: 600;
}
.diff-neutral {
    background-color: #fff3cd !important;
    color: #856404 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🔄 Comparaison de Segmentation Multi-Classes</h1>
    <p>Système de comparaison entre deux modèles de segmentation sémantique</p>
</div>
""", unsafe_allow_html=True)

def get_models_info():
    """Récupère les informations des deux modèles"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des infos modèles: {response.status_code}")
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

def compare_sample_image(filename):
    """Lance une comparaison sur une image d'exemple"""
    try:
        data = {"filename": filename}
        response = requests.post(
            f"{API_BASE_URL}/compare-sample", 
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la comparaison: {e}")
        return None
    
def compare_uploaded_image(uploaded_file):
    """Lance une comparaison sur une image uploadée"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/compare-upload", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la comparaison: {e}")
        return None

def display_comparison_results(result, models_info):
    """Affiche les résultats de comparaison entre les deux modèles"""
    if not result or not result['success']:
        st.error("Aucun résultat de comparaison disponible")
        return
    
    # Informations des modèles
    st.markdown("### 🏆 Modèles Comparés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = result['model1_info']
        # Formater les métriques séparément
        miou1 = f"{model1.get('test_mean_iou'):.4f}" if model1.get('test_mean_iou') is not None else 'N/A'
        acc1 = f"{model1.get('test_accuracy'):.4f}" if model1.get('test_accuracy') is not None else 'N/A'
        
        st.markdown(f"""
        <div class="comparison-card">
            <h4>🥇 Modèle #1</h4>
            <strong>{model1.get('architecture', 'N/A')} - {model1['encoder_name']}</strong><br>
            📐 Taille: {model1['input_size']}<br>
            📊 mIoU: {miou1}<br>
            🎯 Accuracy: {acc1}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model2 = result['model2_info']
        # Formater les métriques séparément
        miou2 = f"{model2.get('test_mean_iou'):.4f}" if model2.get('test_mean_iou') is not None else 'N/A'
        acc2 = f"{model2.get('test_accuracy'):.4f}" if model2.get('test_accuracy') is not None else 'N/A'
        
        st.markdown(f"""
        <div class="comparison-card">
            <h4>🥈 Modèle #2</h4>
            <strong>{model2.get('architecture', 'N/A')} - {model2['encoder_name']}</strong><br>
            📐 Taille: {model2['input_size']}<br>
            📊 mIoU: {miou2}<br>
            🎯 Accuracy: {acc2}
        </div>
        """, unsafe_allow_html=True)
    
    # Comparaison des temps d'inférence
    st.markdown("### ⏱️ Comparaison des Performances")
    
    speed_comp = result['speed_comparison']
    model1_time = result['model1_inference_time'] * 1000
    model2_time = result['model2_inference_time'] * 1000
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        winner_class = "speed-winner" if speed_comp['faster_model'] == model1['encoder_name'] else "speed-slower"
        st.markdown(f"""
        <div class="comparison-card {winner_class}">
            <h5>🥇 {model1.get('architecture', 'N/A')} - {model1['encoder_name']}</h5>
            <strong>{model1_time:.1f} ms</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="comparison-card">
            <h5>⚡ Comparaison</h5>
            <strong>{speed_comp['faster_model']}</strong><br>
            est {speed_comp['speedup_ratio']}x plus rapide<br>
            <small>Différence: {speed_comp['time_difference_ms']:.1f} ms</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        winner_class = "speed-winner" if speed_comp['faster_model'] == model2['encoder_name'] else "speed-slower"
        st.markdown(f"""
        <div class="comparison-card {winner_class}">
            <h5>🥈 {model2.get('architecture', 'N/A')} - {model2['encoder_name']}</h5>
            <strong>{model2_time:.1f} ms</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Affichage des segmentations
    if result.get('figure_data'):
        st.markdown("### 🎨 Résultats de Segmentation")
        try:
            fig_bytes = base64.b64decode(result['figure_data'])
            fig = pickle.loads(fig_bytes)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Erreur lors de l'affichage: {e}")
    
    # Comparaison des statistiques
    st.markdown("### 📊 Répartition des classes par modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if result.get('model1_stats'):
            display_model_stats_chart(result['model1_stats'], models_info, f"{model1.get('architecture', 'N/A')} - {model1['encoder_name']}")
    
    with col2:
        if result.get('model2_stats'):
            display_model_stats_chart(result['model2_stats'], models_info, f"{model2.get('architecture', 'N/A')} - {model2['encoder_name']}")
    
    # Analyse comparative des classes
    if result.get('model1_stats') and result.get('model2_stats'):
        display_class_comparison(result['model1_stats'], result['model2_stats'], model1, model2, models_info)

def display_model_stats_chart(stats, models_info, title):
    """Affiche les statistiques d'un modèle sous forme de graphique"""
    if not stats:
        st.warning("Aucune statistique disponible")
        return
    
    # Utiliser l'ordre des classes du modèle pour la cohérence
    if models_info and 'class_names' in models_info:
        # Ordonner selon l'ordre du modèle
        ordered_class_names = [name for name in models_info['class_names'] if name in stats]
    else:
        # Fallback : ordre alphabétique
        ordered_class_names = sorted(stats.keys())
    
    percentages = [stats[name]['percentage'] for name in ordered_class_names]
    
    # Récupérer les couleurs des classes
    colors = []
    if models_info and 'class_colors' in models_info and 'class_names' in models_info:
        model_class_names = models_info['class_names']
        model_class_colors = models_info['class_colors']
        
        for class_name in ordered_class_names:
            try:
                idx = model_class_names.index(class_name)
                color = model_class_colors[idx]
                colors.append([c/255.0 for c in color])
            except (ValueError, IndexError):
                colors.append([0.5, 0.5, 0.5])
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(ordered_class_names)))
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(ordered_class_names, percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Classes de Segmentation', fontsize=10, fontweight='bold')
    ax.set_ylabel('Pourcentage (%)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    
    # Rotation des labels selon le nombre de classes
    if len(ordered_class_names) <= 4:
        plt.xticks(rotation=0, ha='center', fontsize=9)
    elif len(ordered_class_names) <= 6:
        plt.xticks(rotation=30, ha='right', fontsize=8)
    else:
        plt.xticks(rotation=45, ha='right', fontsize=8)
    
    plt.yticks(fontsize=9)
    
    # Valeurs sur les barres
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(percentages) * 0.01,
               f'{percentage:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style des axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Couleur de fond
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    st.pyplot(fig)
    plt.close(fig)

def display_class_comparison(stats1, stats2, model1, model2, models_info):
    """Affiche une comparaison détaillée des classes entre les deux modèles"""
    st.markdown("### 🔍 Analyse Comparative par Classe")
    
    # Utiliser l'ordre des classes du modèle pour la cohérence
    if models_info and 'class_names' in models_info:
        # Combiner toutes les classes présentes en conservant l'ordre du modèle
        all_classes_in_stats = set(stats1.keys()) | set(stats2.keys())
        ordered_all_classes = [name for name in models_info['class_names'] if name in all_classes_in_stats]
    else:
        # Fallback : ordre alphabétique
        all_classes = set(stats1.keys()) | set(stats2.keys())
        ordered_all_classes = sorted(all_classes)
    
    comparison_data = []
    for class_name in ordered_all_classes:
        pct1 = stats1.get(class_name, {}).get('percentage', 0)
        pct2 = stats2.get(class_name, {}).get('percentage', 0)
        diff = pct1 - pct2
        
        comparison_data.append({
            'class': class_name,
            'model1_pct': pct1,
            'model2_pct': pct2,
            'difference': diff
        })
    
    # Créer un graphique comparatif des barres
    st.markdown("**📊 Comparaison des classes entre les modèles:**")
    
    # Préparation des données
    class_names = [data['class'] for data in comparison_data]
    model1_values = [data['model1_pct'] for data in comparison_data]
    model2_values = [data['model2_pct'] for data in comparison_data]
    
    # Récupérer les couleurs des classes
    colors = []
    if models_info and 'class_colors' in models_info and 'class_names' in models_info:
        model_class_names = models_info['class_names']
        model_class_colors = models_info['class_colors']
        
        for class_name in class_names:
            try:
                idx = model_class_names.index(class_name)
                color = model_class_colors[idx]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                colors.append(hex_color)
            except (ValueError, IndexError):
                colors.append("#999999")
    else:
        colors = ['#999999'] * len(class_names)
    
    # Créer le graphique comparatif
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(class_names))  # Positions des barres
    width = 0.35  # Largeur des barres
    
    bars1 = ax.bar(x - width/2, model1_values, width, label=f"{model1.get('architecture', 'N/A')} - {model1['encoder_name']}", 
                  color=colors, alpha=0.8, edgecolor='white')
    
    # Barres hachurées pour le modèle 2
    bars2 = ax.bar(x + width/2, model2_values, width, label=f"{model2.get('architecture', 'N/A')} - {model2['encoder_name']}", 
                  color=colors, alpha=0.6, edgecolor='white', hatch='///')
    
    ax.set_xlabel('Classes', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pourcentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Comparaison des classes entre les deux modèles', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    
    # Rotation des labels selon le nombre de classes
    if len(class_names) <= 4:
        ax.set_xticklabels(class_names, fontsize=10)
    elif len(class_names) <= 6:
        ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    else:
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Style des axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Valeurs sur les barres
    for bar1, bar2, val1, val2 in zip(bars1, bars2, model1_values, model2_values):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
               f'{val1:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
               f'{val2:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Tableau détaillé avec un beau format
    st.markdown("**🔎 Détails des différences par classe:**")
    
    # Préparer les données pour le tableau
    table_data = []
    for i, data in enumerate(comparison_data):
        class_color = colors[i]
        diff_value = data['difference']
        
        # Classification de la différence
        if abs(diff_value) < 1:
            diff_category = "Similaire"
            diff_emoji = "🟢"
        elif abs(diff_value) < 5:
            diff_category = "Différence modérée"
            diff_emoji = "🟡"
        else:
            diff_category = "Différence importante"
            diff_emoji = "🔴"
        
        table_data.append({
            "Classe": data['class'],
            f"🥇 {model1.get('architecture', 'N/A')[:15]} (%)": f"{data['model1_pct']:.1f}%",
            f"🥈 {model2.get('architecture', 'N/A')[:15]} (%)": f"{data['model2_pct']:.1f}%",
            "Différence": f"{diff_value:+.1f}%",
            "Catégorie": f"{diff_emoji} {diff_category}"
        })
    
    # Créer le DataFrame et l'afficher
    df = pd.DataFrame(table_data)
    
    # Utiliser st.dataframe avec style personnalisé
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Classe": st.column_config.TextColumn("🎯 Classe", width="medium"),
            f"🥇 {model1.get('architecture', 'N/A')[:15]} (%)": st.column_config.TextColumn(width="medium"),
            f"🥈 {model2.get('architecture', 'N/A')[:15]} (%)": st.column_config.TextColumn(width="medium"),
            "Différence": st.column_config.TextColumn("📊 Différence", width="small"),
            "Catégorie": st.column_config.TextColumn("📈 Évaluation", width="medium")
        }
    )

# Sidebar avec informations des modèles
with st.sidebar:
    st.markdown("## 🔧 Informations des Modèles")
    
    models_info = get_models_info()
    if models_info:
        st.success("✅ Modèles chargés")
        
        # Modèle 1
        best_model = models_info.get('best_model', {})
        st.markdown("**🥇 Meilleur Modèle:**")
        st.write(f"**Architecture:** {best_model.get('architecture', 'N/A')}")
        st.write(f"**Encoder:** {best_model.get('encoder_name', 'N/A')}")
        st.write(f"**Résolution:** {best_model.get('input_size', 'N/A')}")
        
        # Modèle 2
        second_model = models_info.get('second_best_model', {})
        st.markdown("**🥈 Deuxième Modèle:**")
        st.write(f"**Architecture:** {second_model.get('architecture', 'N/A')}")
        st.write(f"**Encoder:** {second_model.get('encoder_name', 'N/A')}")
        st.write(f"**Résolution:** {second_model.get('input_size', 'N/A')}")
    else:
        st.error("❌ Modèles non disponibles")

    st.markdown("---")
    st.markdown("## 🎯 Classes Segmentées (IoU)")
    
    # Affichage avec carrés colorés et IoU
    if models_info and 'class_names' in models_info and 'class_colors' in models_info:
        class_names = models_info['class_names']
        class_colors = models_info['class_colors']
        
        # Récupérer les IoU des modèles
        best_ious = best_model.get('class_ious', {}) if models_info else {}
        second_ious = second_model.get('class_ious', {}) if models_info else {}
        
        for i, (class_name, color) in enumerate(zip(class_names, class_colors)):
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            
            # Construire la chaîne IoU
            iou_str = ""
            if best_ious.get(class_name) is not None and second_ious.get(class_name) is not None:
                best_iou = best_ious[class_name]
                second_iou = second_ious[class_name]
                iou_str = f" (🥇{best_iou:.2f} - 🥈{second_iou:.2f})"
            elif best_ious.get(class_name) is not None:
                best_iou = best_ious[class_name]
                iou_str = f" (🥇{best_iou:.2f})"
            elif second_ious.get(class_name) is not None:
                second_iou = second_ious[class_name]
                iou_str = f" (🥈{second_iou:.2f})"
            
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin: 0.3rem 0;
                font-size: 0.85rem;
            ">
                <div style="
                    width: 14px;
                    height: 14px;
                    background-color: {color_hex};
                    border-radius: 2px;
                    border: 1px solid rgba(0,0,0,0.2);
                    flex-shrink: 0;
                "></div>
                <span style="font-weight: 500;">{i}. {class_name}{iou_str}</span>
            </div>
            """, unsafe_allow_html=True)

# États de session
if "sample_comparison_result" not in st.session_state:
    st.session_state.sample_comparison_result = None
if "upload_comparison_result" not in st.session_state:
    st.session_state.upload_comparison_result = None

def reset_sample_comparison():
    st.session_state.sample_comparison_result = None

def reset_upload_comparison():
    st.session_state.upload_comparison_result = None

# Interface principale
tab1, tab2 = st.tabs(["📸 Images d'Exemple", "📤 Upload Personnel"])

# Tab 1: Images d'exemple
with tab1:
    st.markdown("## 📸 Comparaison sur Images d'Exemple")
    st.markdown("Comparez les performances des deux modèles sur des images Cityscapes.")
    
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
                image_url = f"{API_BASE_URL}/sample-image/{selected_filename}"
                response = requests.get(image_url, timeout=5)
                
                if response.status_code == 200:
                    sample_image = Image.open(io.BytesIO(response.content))
                    st.image(
                        sample_image, 
                        caption=f"Aperçu: {selected_display_name}", 
                        use_container_width=True
                    )
            except:
                st.info("📷 Aperçu non disponible - L'image sera chargée lors de la comparaison")
        
        # Bouton de comparaison
        if st.button("🔄 Lancer la Comparaison", key="compare_sample", type="primary"):
            if selected_filename:
                with st.spinner("Comparaison en cours..."):
                    result = compare_sample_image(selected_filename)
                    if result and result['success']:
                        st.session_state.sample_comparison_result = result
                        st.success("✅ Comparaison terminée!")
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors de la comparaison")
    else:
        st.warning("Aucune image d'exemple disponible")

    # Affichage des résultats de comparaison
    if st.session_state.sample_comparison_result:
        st.markdown("---")
        display_comparison_results(st.session_state.sample_comparison_result, models_info)
        
        # Bouton de reset
        if st.button("🔄 Nouvelle Comparaison", key="reset_sample", type="secondary"):
            reset_sample_comparison()
            st.rerun()

# Tab 2: Upload personnel
with tab2:
    st.markdown("## 📤 Comparaison sur Image Personnelle")
    st.markdown("Uploadez une image pour comparer les performances des deux modèles.")
    
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
        
        # Bouton de comparaison
        if st.button("🔄 Lancer la Comparaison", key="compare_upload", type="primary"):
            with st.spinner("Comparaison en cours..."):
                result = compare_uploaded_image(uploaded_file)
                if result and result['success']:
                    st.session_state.upload_comparison_result = result
                    st.success("✅ Comparaison terminée!")
                    st.rerun()
                else:
                    st.error("❌ Erreur lors de la comparaison")

    # Affichage des résultats de comparaison
    if st.session_state.upload_comparison_result:
        st.markdown("---")
        display_comparison_results(st.session_state.upload_comparison_result, models_info)
        
        # Bouton de reset
        if st.button("🔄 Nouvelle Comparaison", key="reset_upload", type="secondary"):
            reset_upload_comparison()
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🔄 <strong>Multi-Class Segmentation Comparison API</strong> - 
    Système de comparaison de modèles de vision par ordinateur<br>
    <small>Développé avec FastAPI, Streamlit et PyTorch</small>
</div>
""", unsafe_allow_html=True)