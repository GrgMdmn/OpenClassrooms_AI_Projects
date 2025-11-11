import os
import sys
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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

# Configuration de la page avec titre approprié (WCAG 2.4.2)
st.set_page_config(
    page_title="Street Vision - Segmentation Comparison | Dashboard Accessible",
    page_icon="🚗",
    layout="wide"
)


# Ajouts à placer au début de streamlit_app.py après les imports

# Palettes de couleurs conformes WCAG AA (ratio contraste ≥ 3:1)
WCAG_ACCESSIBLE_COLORS = {
    'primary_blue': '#0066CC',      # [0, 102, 204]
    'primary_red': '#CC0000',       # [204, 0, 0]
    'primary_green': '#006600',     # [0, 102, 0]
    'primary_orange': '#CC6600',    # [204, 102, 0]
    'secondary_purple': '#663399',  # [102, 51, 153]
    'secondary_teal': '#006666',    # [0, 102, 102]
    'neutral_dark': '#333333',      # [51, 51, 51]
    'neutral_medium': '#666666',    # [102, 102, 102]
    'neutral_light': '#CCCCCC',     # Pour éléments désactivés
    'background_alt': '#F5F5F5'     # Fond alternatif accessible
}

# Convertir hex vers RGB
def hex_to_rgb(hex_color):
    """Convertit une couleur hex en tuple RGB"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Couleurs RGB pour les masques (dans le même ordre que les classes)
WCAG_RGB_COLORS = [
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['neutral_medium']),    # Flat: gris moyen [102, 102, 102]
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['primary_red']),       # Human: rouge [204, 0, 0]  
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['secondary_purple']),  # Vehicle: violet [102, 51, 153]
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['primary_orange']),    # Construction: orange [204, 102, 0]
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['secondary_teal']),    # Object: teal [0, 102, 102]
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['primary_green']),     # Nature: vert [0, 102, 0]
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['primary_blue']),      # Sky: bleu [0, 102, 204]
    hex_to_rgb(WCAG_ACCESSIBLE_COLORS['neutral_dark'])       # Void: gris foncé [51, 51, 51]
]


# Motifs pour différencier sans couleur uniquement
ACCESSIBILITY_PATTERNS = {
    'solid': None,
    'diagonal_lines': '///',
    'vertical_lines': '|||',
    'horizontal_lines': '---',
    'dots': '...',
    'crosses': 'xxx'
}

def create_accessible_pie_chart_conditional(sizes, labels, title, use_patterns=False):
    """Version corrigée qui règle le problème de lisibilité avec les motifs"""
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    
    accessible_colors = [
        WCAG_ACCESSIBLE_COLORS['primary_blue'], WCAG_ACCESSIBLE_COLORS['primary_red'],
        WCAG_ACCESSIBLE_COLORS['primary_green'], WCAG_ACCESSIBLE_COLORS['primary_orange'],
        WCAG_ACCESSIBLE_COLORS['secondary_purple'], WCAG_ACCESSIBLE_COLORS['secondary_teal']
    ]
    colors = accessible_colors[:len(sizes)]
    
    # SOLUTION : texte avec fond contrasté
    uniform_textprops = {
        'fontsize': 10, 'fontweight': 'bold', 'color': 'white',
        'bbox': dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8)
    }
    
    try:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90, shadow=True,
            textprops=uniform_textprops, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            pctdistance=0.85  # Éloigner les pourcentages
        )
        
        # Améliorer la lisibilité des pourcentages
        for autotext in autotexts:
            autotext.set_bbox(dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.9))
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Motifs APRÈS configuration du texte
        if use_patterns:
            patterns = ['///', '|||', '---', '...', 'xxx', '+++']
            for wedge, pattern in zip(wedges, patterns[:len(wedges)]):
                if pattern:
                    wedge.set_hatch(pattern)
        
        ax.set_title(title, fontsize=10, fontweight='bold', pad=15, color='#333333')
        ax.axis('equal')
        chart_displayed = True
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Erreur d'affichage du graphique\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red', weight='bold')
        chart_displayed = False
    
    return fig, ax, {'colors': colors, 'patterns': patterns if use_patterns else None}, chart_displayed

# Fonctions Plotly pour graphiques interactifs
def create_interactive_pie_chart(sizes, labels, title, models_info=None):
    """Crée un pie chart interactif avec les couleurs des classes définies dans models_info"""
    
    # Utiliser les couleurs des classes si disponibles
    if models_info and 'class_colors' in models_info and 'class_names' in models_info:
        colors = []
        class_names = models_info['class_names']
        class_colors = models_info['class_colors']
        
        for label in labels:
            if label in class_names:
                idx = class_names.index(label)
                color_rgb = class_colors[idx]
                color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
                colors.append(color_hex)
            else:
                # Couleur par défaut si classe non trouvée
                colors.append('#666666')
    else:
        # Fallback vers les couleurs WCAG
        accessible_colors_list = list(WCAG_ACCESSIBLE_COLORS.values())
        colors = accessible_colors_list[:len(sizes)]
    
    df = pd.DataFrame({'labels': labels, 'values': sizes, 'colors': colors})
    
    fig = go.Figure(data=[go.Pie(
        labels=df['labels'],
        values=df['values'],
        marker=dict(
            colors=df['colors'], 
            line=dict(color='#FFFFFF', width=3),
            pattern=dict(
                shape=['', '/', '\\', '|', '-', '+', 'x', '.'][:len(sizes)],
                size=8,
                solidity=0.3
            )
        ),
        textinfo='label+percent',
        textposition='auto',
        textfont=dict(
            size=13,
            color='white', 
            family="Arial Black"
        ),
        hovertemplate='<b>%{label}</b><br>Valeur: %{value}<br>Pourcentage: %{percent}<br><extra></extra>',
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#333333",
            font=dict(size=14, color="#333333")
        )
    )])
    
    fig.update_layout(
        title={
            'text': title, 
            'x': 0.5, 
            'xanchor': 'center', 
            'font': {'size': 16, 'color': '#333333', 'family': 'Arial Black'}
        },
        showlegend=True,
        legend=dict(
            orientation="v", 
            yanchor="middle", 
            y=0.5, 
            xanchor="left", 
            x=1.02, 
            font=dict(size=12, color='#333333'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#333333',
        ),
        margin=dict(l=20, r=120, t=60, b=20),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333')
    )
    
    return fig


def create_interactive_comparison_chart(model1_stats, model2_stats, model1_name, model2_name):
    """Crée un graphique en barres interactif WCAG-compliant pour comparer les deux modèles"""
    classes = list(set(model1_stats.keys()) | set(model2_stats.keys()))
    
    data = []
    for class_name in classes:
        val1 = model1_stats.get(class_name, {}).get('percentage', 0)
        val2 = model2_stats.get(class_name, {}).get('percentage', 0)
        
        data.append({'Classe': class_name, 'Modèle': model1_name, 'Pourcentage': val1, 'Pixels': model1_stats.get(class_name, {}).get('pixels', 0)})
        data.append({'Classe': class_name, 'Modèle': model2_name, 'Pourcentage': val2, 'Pixels': model2_stats.get(class_name, {}).get('pixels', 0)})
    
    df = pd.DataFrame(data)
    
    # Filtrer les données pour supprimer les lignes vides potentielles
    df = df[df['Modèle'].notna() & (df['Modèle'] != '')]
    
    fig = px.bar(
        df, x='Classe', y='Pourcentage', color='Modèle', barmode='group',
        title='Comparaison interactive des classes détectées',
        hover_data=['Pixels'],
        color_discrete_map={model1_name: '#0066CC', model2_name: '#CC0000'}
        # PAS de pattern_shape du tout
    )
    
    fig.update_layout(
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#333333', 'family': 'Arial Black'}},
        xaxis=dict(
            title='Classes de segmentation', 
            titlefont=dict(size=14, color='#666666'),  # Plus clair comme vos pie charts
            tickfont=dict(size=12, color='#666666'),   # Plus clair
            tickangle=45,
            showgrid=True,
            gridcolor='#E8E8E8',     # Grille beaucoup plus claire
            gridwidth=1,
            linecolor='#AAAAAA',     # Axe gris moyen
            linewidth=1,
            showline=True
        ),
        yaxis=dict(
            title='Pourcentage de l\'image (%)', 
            titlefont=dict(size=14, color='#666666'),  # Plus clair
            tickfont=dict(size=12, color='#666666'),   # Plus clair
            showgrid=True,
            gridcolor='#E8E8E8',     # Grille beaucoup plus claire
            gridwidth=1,
            linecolor='#AAAAAA',     # Axe gris moyen
            linewidth=1,
            showline=True
        ),
        legend=dict(
            title='Modèles', 
            font=dict(size=12, color='#333333'),       # Comme vos pie charts
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#333333',
        ),
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',        # Transparent
        paper_bgcolor='rgba(0,0,0,0)',       # Transparent
        font=dict(color='#333333')           # Sans spécifier size qui peut poser problème
    )
    
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='outside', 
        textfont=dict(size=11, color='#666666'),      # Plus clair pour lisibilité
        marker=dict(line=dict(color='#FFFFFF', width=1))  # Bordures blanches comme vos pie charts
    )
    
    return fig


def create_dataset_interactive_overview():
    """Vue d'ensemble interactive du dataset avec les MÊMES couleurs que 'Visualisations accessibles'"""
    
    # RÉORGANISER les données pour que l'ordre soit logique et que les couleurs correspondent
    dataset_data = {
        'Split': ['Train (notre projet)', 'Validation (notre projet)', 'Test final', 'Non utilisé'],
        'Images': [2380, 595, 500, 21525],
        'Utilisation': ['Entraînement', 'Validation', 'Test', 'Non utilisé']
    }
    
    df = pd.DataFrame(dataset_data)
    
    # Couleurs dans l'ordre exact des données
    colors = [
        WCAG_ACCESSIBLE_COLORS['primary_blue'],      # Train (notre projet) - bleu
        WCAG_ACCESSIBLE_COLORS['primary_red'],       # Validation (notre projet) - rouge  
        WCAG_ACCESSIBLE_COLORS['primary_orange'],    # Test final - orange
        WCAG_ACCESSIBLE_COLORS['neutral_dark']       # Non utilisé - gris foncé
    ]
    
    # FORCER l'ordre avec category_orders
    fig = px.pie(
        df, 
        values='Images', 
        names='Split',
        title='Répartition interactive du dataset Cityscapes',
        color_discrete_sequence=colors,
        hover_data=['Utilisation'],
        category_orders={'Split': ['Train (notre projet)', 'Validation (notre projet)', 'Test final', 'Non utilisé']}
    )
    
    # FORCER la correspondance couleur-catégorie avec color_discrete_map
    color_map = {
        'Train (notre projet)': WCAG_ACCESSIBLE_COLORS['primary_blue'],
        'Validation (notre projet)': WCAG_ACCESSIBLE_COLORS['primary_red'],
        'Test final': WCAG_ACCESSIBLE_COLORS['primary_orange'],
        'Non utilisé': WCAG_ACCESSIBLE_COLORS['neutral_dark']
    }
    
    fig.update_traces(
        marker=dict(
            colors=[color_map[split] for split in df['Split']],  # Forcer les couleurs exactes
            line=dict(color='#FFFFFF', width=3),
            pattern=dict(
                shape=['|', '/', '\\', ''],
                size=8,
                solidity=0.3
            )
        ),
        textposition='auto', 
        textinfo='percent+label',
        textfont=dict(size=12, color='white', family="Arial Black"),
        hovertemplate='<b>%{label}</b><br>Images: %{value}<br>Usage: %{customdata[0]}<extra></extra>'
    )
    
    fig.update_layout(
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#333333'}}, 
        height=500,
        legend=dict(
            font=dict(size=12, color='#333333'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#333333',
        ),
        font=dict(color='#333333')
    )
    
    return fig


def create_accessible_bar_chart_conditional(data, labels, title, colors=None, use_patterns=False):
    """
    Crée un bar chart accessible - descriptions uniquement si erreur d'affichage
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if colors is None:
        colors = [WCAG_ACCESSIBLE_COLORS['primary_blue']] * len(data)
    
    # Motifs pour différenciation
    patterns = None
    if use_patterns:
        pattern_list = list(ACCESSIBILITY_PATTERNS.values())
        patterns = pattern_list[:len(data)]
    
    try:
        bars = ax.bar(
            labels, 
            data, 
            color=colors, 
            alpha=0.8, 
            edgecolor='#333333', 
            linewidth=1
        )
        
        # Ajouter des motifs
        if patterns:
            for bar, pattern in zip(bars, patterns):
                if pattern:
                    bar.set_hatch(pattern)
        
        # Style accessible
        ax.set_ylabel('Valeur', fontweight='bold', color='#333333')
        ax.set_xlabel('Catégories', fontweight='bold', color='#333333')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='#333333')
        
        # Grille pour faciliter la lecture
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='#666666')
        ax.set_axisbelow(True)
        
        # Valeurs sur les barres pour accessibilité
        for bar, value in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(data) * 0.01,
                   f'{value:.1f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
        
        # Style des axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        plt.xticks(rotation=45, ha='right', color='#333333')
        plt.yticks(color='#333333')
        
        chart_displayed = True
        
    except Exception as e:
        # En cas d'erreur, message d'erreur
        ax.text(0.5, 0.5, f"Erreur d'affichage du graphique\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red', weight='bold')
        chart_displayed = False
    
    return fig, ax, chart_displayed

# Fonction pour générer du texte alternatif (WCAG 1.1.1)
def generate_alt_text_for_pie(sizes, labels, title):
    """
    Génère un texte alternatif détaillé pour un pie chart
    """
    total = sum(sizes)
    descriptions = []
    
    for size, label in zip(sizes, labels):
        percentage = (size / total) * 100
        descriptions.append(f"{label}: {size} éléments ({percentage:.1f}%)")
    
    alt_text = f"Graphique en secteurs: {title}. "
    alt_text += f"Total de {total} éléments répartis comme suit: "
    alt_text += "; ".join(descriptions) + "."
    
    return alt_text

# Fonction pour générer du texte alternatif (WCAG 1.1.1)
def generate_alt_text_for_images(image_type, model_name=None, stats=None):
    """Génère un texte alternatif détaillé pour les images de segmentation"""
    if image_type == "original":
        return "Image originale en couleur montrant une scène urbaine typique du dataset Cityscapes"
    elif image_type == "segmentation":
        base_text = f"Masque de segmentation sémantique prédit par le modèle {model_name or 'IA'}"
        if stats:
            # Ajouter les statistiques principales
            main_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)[:3]
            stats_text = ". Classes principales détectées: " + ", ".join([
                f"{name} ({data['percentage']:.1f}%)" for name, data in main_classes
            ])
            return base_text + stats_text
        return base_text
    elif image_type == "comparison":
        return "Visualisation comparative des prédictions de deux modèles de segmentation sémantique"
    else:
        return "Image de segmentation sémantique"

def generate_alt_text_for_bar(data, labels, title, unit=""):
    """
    Génère un texte alternatif détaillé pour un bar chart
    """
    descriptions = []
    
    for value, label in zip(data, labels):
        descriptions.append(f"{label}: {value}{unit}")
    
    alt_text = f"Graphique en barres: {title}. "
    alt_text += "Valeurs: " + "; ".join(descriptions) + "."
    
    return alt_text

def generate_comprehensive_alt_text(image_type, model_name=None, stats=None, comparison_data=None):
    """Génère un texte alternatif complet selon WCAG 1.1.1"""
    if image_type == "preprocessing_original":
        return "Image urbaine originale redimensionnée à la résolution d'entraînement du modèle, montrant une scène typique du dataset Cityscapes avec véhicules, piétons et infrastructure urbaine"
    
    elif image_type == "preprocessing_normalized":
        return "Même image après normalisation des pixels pour l'entraînement du modèle IA, les couleurs peuvent paraître légèrement différentes"
    
    elif image_type == "preprocessing_augmented":
        return "Image avec augmentations de données appliquées (rotation, changement de luminosité, etc.) pour améliorer la robustesse du modèle"
    
    elif image_type == "segmentation_prediction":
        base_text = f"Masque de segmentation sémantique coloré prédit par le modèle {model_name or 'IA'}"
        if stats:
            main_classes = sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True)[:3]
            stats_text = ". Classes principales détectées : " + ", ".join([
                f"{name} ({data['percentage']:.1f}% de l'image)" for name, data in main_classes
            ])
            return base_text + stats_text
        return base_text
    
    elif image_type == "comparison_grid":
        return "Grille comparative montrant l'image originale et les prédictions de segmentation de deux modèles d'IA différents côte à côte pour évaluation"
    
    else:
        return "Image de segmentation sémantique urbaine générée par intelligence artificielle"

def create_data_table(data, columns, title, description=""):
    """
    Crée un tableau de données accessible complémentaire aux graphiques
    """
    import pandas as pd
    
    df = pd.DataFrame(data, columns=columns)
    
    st.markdown(f"**Tableau de données: {title}**")
    if description:
        st.markdown(f"*{description}*")
    
    # Configuration accessible du dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.TextColumn(col, width="medium") 
            for col in columns
        }
    )
    
    return df

# CSS amélioré pour l'accessibilité complète
ENHANCED_WCAG_CSS = """
<style>
/* WCAG 1.4.4 - Support du redimensionnement jusqu'à 200% */
html {
    font-size: 16px; /* Taille de base */
}

/* WCAG 1.4.1 - Ne pas utiliser la couleur comme seul moyen de transmission d'information */
.status-success {
    background: #E8F5E8;
    border-left: 4px solid #006600;
    color: #333333;
    font-weight: 500;
}
.status-success::before {
    content: "✅ ";
    font-weight: bold;
}

.status-error {
    background: #FFF0F0;
    border-left: 4px solid #CC0000;
    color: #333333;
    font-weight: 500;
}
.status-error::before {
    content: "❌ ";
    font-weight: bold;
}

.status-warning {
    background: #FFF8E1;
    border-left: 4px solid #CC6600;
    color: #333333;
    font-weight: 500;
}
.status-warning::before {
    content: "⚠️ ";
    font-weight: bold;
}

/* WCAG 1.4.3 - Contraste minimum respecté */
.main-header {
    background: linear-gradient(90deg, #0066CC 0%, #663399 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

/* Styles pour les cartes de comparaison - MANQUANTS DANS VOTRE VERSION */
.comparison-card {
    background: #F5F5F5;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #0066CC;
    margin: 1rem 0;
    color: #333333;
    font-weight: 500;
}

.speed-winner {
    background: #E8F5E8;
    border-left-color: #006600;
    color: #333333;
}

.speed-slower {
    background: #FFF0F0;
    border-left-color: #CC0000;
    color: #333333;
}

/* Navigation accessible */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 12px 16px;
    font-size: 16px;
    font-weight: 500;
    border-radius: 6px;
    border: 2px solid transparent;
}

.stTabs [data-baseweb="tab"]:focus {
    outline: 2px solid #0066CC;
    outline-offset: 2px;
}

/* Amélioration des boutons pour l'accessibilité */
.stButton button {
    min-height: 44px; /* Taille minimale pour accessibilité tactile */
    font-size: 16px;
    font-weight: 500;
    border-radius: 6px;
    border: 2px solid transparent;
}

.stButton button:focus {
    outline: 2px solid #0066CC;
    outline-offset: 2px;
}

/* Sélecteurs accessibles */
.stSelectbox label {
    font-weight: 600;
    font-size: 16px;
    color: #333333;
}

/* Messages d'état accessibles */
.element-container .stAlert {
    border-radius: 6px;
    border-left-width: 4px;
    font-weight: 500;
}

/* Tableaux accessibles */
.stDataFrame {
    border: 2px solid #666666;
}

.stDataFrame table {
    border-collapse: collapse;
}

.stDataFrame th {
    background-color: #F5F5F5;
    color: #333333;
    font-weight: bold;
    border: 1px solid #666666;
    padding: 12px;
    font-size: 16px;
}

.stDataFrame td {
    border: 1px solid #CCCCCC;
    color: #333333;
    padding: 12px;
    font-size: 16px;
}

/* Images accessibles */
.stImage > div {
    border: 1px solid #CCCCCC;
    border-radius: 4px;
}

/* Support du zoom jusqu'à 200% sans perte de fonctionnalité */
@media (min-resolution: 192dpi) {
    body {
        font-size: 18px;
    }
}
</style>
"""

st.markdown(ENHANCED_WCAG_CSS, unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🚗 Street Vision - Comparaison de Modèles</h1>
    <p>Évaluation comparative SegFormer vs FPN sur le dataset Cityscapes</p>
</div>
""", unsafe_allow_html=True)

# Fonctions utilitaires
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

def get_dataset_info():
    """Récupère les informations sur le dataset"""
    try:
        response = requests.get(f"{API_BASE_URL}/dataset-info")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des infos dataset: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def get_train_images():
    """Récupère la liste des images d'entraînement"""
    try:
        response = requests.get(f"{API_BASE_URL}/train-images")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des images: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def get_class_repartition():
    """Récupère le graphique de répartition des classes"""
    try:
        response = requests.get(f"{API_BASE_URL}/class-repartition", timeout=10)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        st.warning(f"Graphique de répartition non disponible: {e}")
        return None

def preprocess_train_image(filename, apply_augmentation=False):
    """Lance le preprocessing sur une image d'entraînement"""
    try:
        data = {
            "filename": filename,
            "apply_augmentation": apply_augmentation
        }
        response = requests.post(
            f"{API_BASE_URL}/preprocess-train-image",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du preprocessing: {e}")
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

def display_comparison_results(result, models_info, context=""):
    """Affiche les résultats de comparaison avec graphiques interactifs et accessibilité complète"""
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
    
    # Affichage des segmentations avec descriptions alternatives CONDITIONNELLES
    if result.get('figure_data'):
        st.markdown("### 🎨 Résultats de Segmentation")
        
        try:
            fig_bytes = base64.b64decode(result['figure_data'])
            fig = pickle.loads(fig_bytes)
            
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Erreur lors de l'affichage: {e}")
            # Description alternative SEULEMENT en cas d'erreur
            alt_text = generate_comprehensive_alt_text("comparison_grid")
            st.warning(f"**Description alternative:** {alt_text}")
    
    # Section interactive (CE2 - Graphiques interactifs) - SEULEMENT 2 ONGLETS
    st.markdown("### 📊 Analyse Interactive des Résultats")
    
    if result.get('model1_stats') and result.get('model2_stats'):
        # SUPPRIMER l'onglet "Performances"
        tab1, tab2 = st.tabs(["🥧 Répartitions", "📊 Comparaison"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Répartition Modèle 1")
                sizes1 = [data['percentage'] for data in result['model1_stats'].values()]
                labels1 = list(result['model1_stats'].keys())
                
                fig_pie1 = create_interactive_pie_chart(sizes1, labels1, 
                                           f"Distribution - {result['model1_info'].get('architecture', 'N/A')} {result['model1_info']['encoder_name']}", 
                                           models_info)
                # AJOUTER key unique pour éviter l'erreur Streamlit
                st.plotly_chart(fig_pie1, use_container_width=True, key=f"pie_model1_{context}")
                
            with col2:
                st.markdown("#### Répartition Modèle 2")
                sizes2 = [data['percentage'] for data in result['model2_stats'].values()]
                labels2 = list(result['model2_stats'].keys())
                
                fig_pie2 = create_interactive_pie_chart(sizes2, labels2, 
                                           f"Distribution - {result['model2_info'].get('architecture', 'N/A')} {result['model2_info']['encoder_name']}", 
                                           models_info)
                # AJOUTER key unique pour éviter l'erreur Streamlit
                st.plotly_chart(fig_pie2, use_container_width=True, key=f"pie_model2_{context}")
        
        with tab2:
            st.markdown("#### Comparaison Interactive des Classes")
            
            fig_comparison = create_interactive_comparison_chart(
                result['model1_stats'], result['model2_stats'],
                f"{result['model1_info'].get('architecture', 'N/A')} {result['model1_info']['encoder_name']}",
                f"{result['model2_info'].get('architecture', 'N/A')} {result['model2_info']['encoder_name']}"
            )
            # AJOUTER key unique pour éviter l'erreur Streamlit
            st.plotly_chart(fig_comparison, use_container_width=True, key=f"comparison_chart_{context}")
            
            st.info("💡 **Interactivité :** Survolez les barres, cliquez sur la légende, zoomez en sélectionnant une zone.")
            # SUPPRIMER la description systématique
    
    # Garder seulement le tableau détaillé (sans graphique redondant)
    if result.get('model1_stats') and result.get('model2_stats'):
        display_class_comparison(result['model1_stats'], result['model2_stats'], 
                                result['model1_info'], result['model2_info'], models_info)
    
# def display_model_stats_chart(stats, models_info, title):
#     """Affiche les statistiques d'un modèle sous forme de graphique accessible"""
#     if not stats:
#         st.warning("Aucune statistique disponible")
#         return
    
#     # Utiliser l'ordre des classes du modèle pour la cohérence
#     if models_info and 'class_names' in models_info:
#         ordered_class_names = [name for name in models_info['class_names'] if name in stats]
#     else:
#         ordered_class_names = sorted(stats.keys())
    
#     percentages = [stats[name]['percentage'] for name in ordered_class_names]
    
#     # Couleurs accessibles WCAG
#     accessible_colors = [
#         WCAG_ACCESSIBLE_COLORS['neutral_medium'],    # Flat: gris moyen
#         WCAG_ACCESSIBLE_COLORS['primary_red'],       # Human: rouge
#         WCAG_ACCESSIBLE_COLORS['secondary_purple'],  # Vehicle: violet
#         WCAG_ACCESSIBLE_COLORS['primary_orange'],    # Construction: orange
#         WCAG_ACCESSIBLE_COLORS['secondary_teal'],    # Object: teal
#         WCAG_ACCESSIBLE_COLORS['primary_green'],     # Nature: vert
#         WCAG_ACCESSIBLE_COLORS['primary_blue'],      # Sky: bleu
#         WCAG_ACCESSIBLE_COLORS['neutral_dark']       # Void: gris foncé
#     ]
    
#     colors = accessible_colors[:len(ordered_class_names)]
    
#     # Créer le graphique accessible
#     fig, ax, displayed = create_accessible_bar_chart_conditional(
#         percentages, ordered_class_names, title, 
#         colors=colors, use_patterns=True
#     )
    
#     # Description alternative SEULEMENT si erreur d'affichage
#     if not displayed:
#         alt_text = generate_alt_text_for_bar(
#             percentages, ordered_class_names, title, unit="%"
#         )
#         st.warning(f"**Description du graphique:** {alt_text}")
    
#     st.pyplot(fig)
#     plt.close(fig)
    
#     # Tableau de données complémentaire (toujours affiché)
#     table_data = []
#     for name, percentage in zip(ordered_class_names, percentages):
#         pixels = stats[name]['pixels']
#         table_data.append([name, f"{percentage:.1f}%", f"{pixels:,} pixels"])
    
#     create_data_table(
#         table_data,
#         ["Classe", "Pourcentage", "Nombre de pixels"],
#         f"Données - {title}",
#         "Répartition détaillée des classes de segmentation"
#     )

def display_class_comparison(stats1, stats2, model1, model2, models_info):
    """Affiche uniquement le tableau détaillé de comparaison des classes (sans graphique redondant)"""
    st.markdown("### 🔎 Tableau Détaillé des Comparaisons")
    
    # Préparer les données
    if models_info and 'class_names' in models_info:
        all_classes_in_stats = set(stats1.keys()) | set(stats2.keys())
        ordered_all_classes = [name for name in models_info['class_names'] if name in all_classes_in_stats]
    else:
        all_classes = set(stats1.keys()) | set(stats2.keys())
        ordered_all_classes = sorted(all_classes)
    
    class_names = ordered_all_classes
    model1_values = [stats1.get(name, {}).get('percentage', 0) for name in class_names]
    model2_values = [stats2.get(name, {}).get('percentage', 0) for name in class_names]
    
    # SUPPRIMER tout le code de génération du graphique matplotlib
    # GARDER seulement le tableau
    
    table_data = []
    for i, name in enumerate(class_names):
        val1 = model1_values[i]
        val2 = model2_values[i]
        diff = val1 - val2
        
        if abs(diff) < 1:
            category = "🟢 Similaire"
        elif abs(diff) < 5:
            category = "🟡 Différence modérée"
        else:
            category = "🔴 Différence importante"
        
        table_data.append([
            name,
            f"{val1:.1f}%",
            f"{val2:.1f}%", 
            f"{diff:+.1f}%",
            category
        ])
    
    create_data_table(
        table_data,
        [
            "Classe",
            f"🥇 {model1.get('architecture', 'N/A')[:15]} (%)",
            f"🥈 {model2.get('architecture', 'N/A')[:15]} (%)",
            "Différence",
            "Évaluation"
        ],
        "Comparaison détaillée par classe",
        "Analyse des écarts de performance entre les deux modèles pour chaque classe de segmentation"
    )

# def display_class_comparison(stats1, stats2, model1, model2, models_info):
#     """Affiche une comparaison accessible des classes entre les deux modèles"""
#     st.markdown("### 🔍 Analyse Comparative par Classe")
    
#     # Préparer les données
#     if models_info and 'class_names' in models_info:
#         all_classes_in_stats = set(stats1.keys()) | set(stats2.keys())
#         ordered_all_classes = [name for name in models_info['class_names'] if name in all_classes_in_stats]
#     else:
#         all_classes = set(stats1.keys()) | set(stats2.keys())
#         ordered_all_classes = sorted(all_classes)
    
#     class_names = ordered_all_classes
#     model1_values = [stats1.get(name, {}).get('percentage', 0) for name in class_names]
#     model2_values = [stats2.get(name, {}).get('percentage', 0) for name in class_names]
    
#     # Couleurs accessibles avec motifs différents
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     x = np.arange(len(class_names))
#     width = 0.35
    
#     # Couleurs accessibles
#     color1 = WCAG_ACCESSIBLE_COLORS['primary_blue']
#     color2 = WCAG_ACCESSIBLE_COLORS['primary_red']
    
#     bars1 = ax.bar(x - width/2, model1_values, width, 
#                    label=f"{model1.get('architecture', 'N/A')} - {model1['encoder_name']}", 
#                    color=color1, alpha=0.8, edgecolor='#333333')
    
#     bars2 = ax.bar(x + width/2, model2_values, width,
#                    label=f"{model2.get('architecture', 'N/A')} - {model2['encoder_name']}", 
#                    color=color2, alpha=0.8, edgecolor='#333333', hatch='///')
    
#     # Style accessible
#     ax.set_xlabel('Classes', fontsize=11, fontweight='bold', color='#333333')
#     ax.set_ylabel('Pourcentage (%)', fontsize=11, fontweight='bold', color='#333333')
#     ax.set_title('Comparaison des classes entre les deux modèles', 
#                  fontsize=13, fontweight='bold', pad=15, color='#333333')
#     ax.set_xticks(x)
    
#     if len(class_names) <= 4:
#         ax.set_xticklabels(class_names, fontsize=10, color='#333333')
#     elif len(class_names) <= 6:
#         ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9, color='#333333')
#     else:
#         ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9, color='#333333')
    
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(axis='y', alpha=0.3, linestyle='--', color='#666666')
#     ax.set_axisbelow(True)
    
#     # Style des axes
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_color('#666666')
#     ax.spines['bottom'].set_color('#666666')
    
#     # Valeurs sur les barres
#     for bar1, bar2, val1, val2 in zip(bars1, bars2, model1_values, model2_values):
#         height1, height2 = bar1.get_height(), bar2.get_height()
#         ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
#                f'{val1:.1f}%', ha='center', va='bottom', fontsize=8, 
#                fontweight='bold', color='#333333')
#         ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
#                f'{val2:.1f}%', ha='center', va='bottom', fontsize=8, 
#                fontweight='bold', color='#333333')
    
#     plt.tight_layout()
    
#     # Description alternative
#     alt_text = f"Graphique comparatif en barres des classes entre {model1['encoder_name']} et {model2['encoder_name']}. "
#     comparisons = []
#     for name, val1, val2 in zip(class_names, model1_values, model2_values):
#         comparisons.append(f"{name}: {val1:.1f}% vs {val2:.1f}%")
#     alt_text += "Comparaisons: " + "; ".join(comparisons)
    
#     st.markdown(f"**Description du graphique:** {alt_text}")
#     st.pyplot(fig)
#     plt.close(fig)
    
#     # Tableau détaillé accessible
#     st.markdown("**🔎 Tableau détaillé des comparaisons:**")
    
#     table_data = []
#     for i, name in enumerate(class_names):
#         val1 = model1_values[i]
#         val2 = model2_values[i]
#         diff = val1 - val2
        
#         if abs(diff) < 1:
#             category = "🟢 Similaire"
#         elif abs(diff) < 5:
#             category = "🟡 Différence modérée"
#         else:
#             category = "🔴 Différence importante"
        
#         table_data.append([
#             name,
#             f"{val1:.1f}%",
#             f"{val2:.1f}%", 
#             f"{diff:+.1f}%",
#             category
#         ])
    
#     create_data_table(
#         table_data,
#         [
#             "Classe",
#             f"🥇 {model1.get('architecture', 'N/A')[:15]} (%)",
#             f"🥈 {model2.get('architecture', 'N/A')[:15]} (%)",
#             "Différence",
#             "Évaluation"
#         ],
#         "Comparaison détaillée par classe",
#         "Analyse des écarts de performance entre les deux modèles pour chaque classe de segmentation"
#     )


# Onglets principaux
main_tab1, main_tab2 = st.tabs(["📊 Présentation Jeu de données Cityscapes", "🔄 Comparaison de performances"])

# ===== ONGLET 1: PRÉSENTATION DATASET =====
with main_tab1:
    st.markdown("## 📊 Analyse Exploratoire - Dataset Cityscapes")
    
    # Informations générales sur le dataset
    dataset_info = get_dataset_info()
    
    if dataset_info:
        # Section méthodologique MODIFIÉE
        st.markdown("""
        ### 🎯 Présentation générale du dataset Cityscapes

        #### Dataset retenu

        **Cityscapes Dataset : Référence en Segmentation Urbaine**
        """)
        
        # Spécificités techniques
        st.markdown("""
        ### ⚙️ Spécificités techniques
        
        - **Taille Datase**: 25 000 images haute résolution
        - **Résolutions d'acquisition** : Données natives 2048×1024 pixels
        - **Conditions d'acquisition** : Conditions météo bonnes à moyennes, pas de conditions adverses (pluie, neige)
        - **Diversité géographique** : Villes allemandes et suisses (architectures variées)  
        - **Annotations pixel-perfect** : Masques de segmentation précis au pixel près (Train + Val)

        Ce dataset permet d'évaluer la capacité des modèles à comprendre les scènes urbaines complexes, élément fondamental pour les applications de conduite autonome et de surveillance urbaine intelligente.
        """)
    
        # Composition du dataset avec pie charts NOUVEAU
        st.markdown("### 🗂️ Composition et utilisation du dataset")
        
        # Version finale qui combine : tailles uniformes + couleurs WCAG + descriptions conditionnelles
        if dataset_info.get('splits'):
            splits_info = dataset_info['splits']
            project_splits = dataset_info.get('project_splits', {})
            
            st.markdown("#### 📊 Visualisations accessibles")
            
            # Option pour activer les motifs (accessibilité)
            use_patterns = st.checkbox(
                "🎨 Activer les motifs (accessibilité visuelle)",
                help="Active les motifs en plus des couleurs pour faciliter la distinction des données",
                value=True
            )
            
            # Organiser en 1 ligne de 4 colonnes
            col1, col2, col3, col4 = st.columns(4)
            
            # Configuration uniforme pour tous les graphiques (SOLUTION QUI MARCHAIT)
            uniform_figsize = (6, 6)
            uniform_textprops = {'fontsize': 8, 'fontweight': 'bold', 'color': '#333333'}
            uniform_title_props = {'fontsize': 10, 'fontweight': 'bold', 'pad': 15}
            
            # Couleurs WCAG accessibles
            accessible_colors = {
                'blue': WCAG_ACCESSIBLE_COLORS['primary_blue'],
                'red': WCAG_ACCESSIBLE_COLORS['primary_red'], 
                'orange': WCAG_ACCESSIBLE_COLORS['primary_orange'],
                'gray': WCAG_ACCESSIBLE_COLORS['neutral_medium'],
                'light_gray': WCAG_ACCESSIBLE_COLORS['neutral_light']
            }
            
            # Motifs pour accessibilité
            patterns = ['///', '|||', '---', '...', 'xxx', None] if use_patterns else [None] * 6
            
            # Données pour le tableau récapitulatif
            pie_data_for_table = []
            
            with col1:
                # Pie chart Train (avec répartition projet)
                fig_train, ax_train = plt.subplots(figsize=uniform_figsize, tight_layout=True)
                
                train_total = splits_info['train']['images']
                train_our_train = project_splits['train']['images']
                train_our_val = project_splits['validation']['images']
                
                sizes = [train_our_train, train_our_val]
                labels = [f'Notre Entraînement\n{train_our_train} images (80%)', 
                         f'Notre Validation\n{train_our_val} images (20%)']
                colors = [accessible_colors['blue'], accessible_colors['red']]
                explode = (0.05, 0.05)
                
                try:
                    wedges, texts, autotexts = ax_train.pie(
                        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                        startangle=90, explode=explode, shadow=True,
                        textprops=uniform_textprops,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                    )
                    
                    # Ajouter des motifs si demandé
                    if use_patterns:
                        for wedge, pattern in zip(wedges, patterns[:2]):
                            if pattern:
                                wedge.set_hatch(pattern)
                    
                    ax_train.set_title(f'Split Train Original\n{train_total} images total', 
                                     **uniform_title_props, color='#333333')
                    ax_train.axis('equal')
                    
                    chart_displayed = True
                    
                except Exception as e:
                    ax_train.text(0.5, 0.5, f"Erreur d'affichage\n{str(e)}", 
                                 ha='center', va='center', transform=ax_train.transAxes,
                                 fontsize=10, color='red', weight='bold')
                    chart_displayed = False
                
                # Description SEULEMENT si erreur
                if not chart_displayed:
                    alt_text = generate_alt_text_for_pie(sizes, labels, "Split Train")
                    st.warning(f"**Description du graphique:** {alt_text}")
                
                st.pyplot(fig_train)
                plt.close(fig_train)
                
                pie_data_for_table.extend([
                    ["Train - Entraînement", train_our_train, "80%", "Entraînement des modèles"],
                    ["Train - Validation", train_our_val, "20%", "Validation pendant entraînement"]
                ])
            
            with col2:
                # Pie chart Validation (utilisation complète)
                fig_val, ax_val = plt.subplots(figsize=uniform_figsize, tight_layout=True)
                
                val_total = splits_info['validation']['images']
                val_our_test = project_splits['test']['images']
                
                sizes = [val_our_test]
                labels = [f'Notre Test Final\n{val_our_test} images (100%)']
                colors = [accessible_colors['orange']]
                
                try:
                    wedges, texts, autotexts = ax_val.pie(
                        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                        startangle=90, shadow=True, textprops=uniform_textprops,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                    )
                    
                    if use_patterns and patterns[2]:
                        wedges[0].set_hatch(patterns[2])
                    
                    ax_val.set_title(f'Split Validation Original\n{val_total} images total', 
                                   **uniform_title_props, color='#333333')
                    ax_val.axis('equal')
                    
                    chart_displayed = True
                    
                except Exception as e:
                    ax_val.text(0.5, 0.5, f"Erreur d'affichage\n{str(e)}", 
                               ha='center', va='center', transform=ax_val.transAxes,
                               fontsize=10, color='red', weight='bold')
                    chart_displayed = False
                
                if not chart_displayed:
                    alt_text = generate_alt_text_for_pie(sizes, labels, "Split Validation")
                    st.warning(f"**Description du graphique:** {alt_text}")
                
                st.pyplot(fig_val)
                plt.close(fig_val)
                
                pie_data_for_table.append(["Validation - Test", val_our_test, "100%", "Test final des modèles"])
            
            with col3:
                # Pie chart Test (non utilisé)
                fig_test, ax_test = plt.subplots(figsize=uniform_figsize, tight_layout=True)
                
                test_total = splits_info['test']['images']
                
                sizes = [test_total]
                labels = [f'Non utilisé dans notre projet\n{test_total} images\n(annotations privées)']
                colors = [accessible_colors['gray']]
                
                try:
                    wedges, texts, autotexts = ax_test.pie(
                        sizes, labels=labels, colors=colors, autopct='',
                        startangle=90, shadow=True, textprops=uniform_textprops,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                    )
                    
                    if use_patterns and patterns[3]:
                        wedges[0].set_hatch(patterns[3])
                    
                    ax_test.set_title(f'Split Test Original\n{test_total} images total', 
                                    **uniform_title_props, color='#333333')
                    ax_test.axis('equal')
                    
                    chart_displayed = True
                    
                except Exception as e:
                    ax_test.text(0.5, 0.5, f"Erreur d'affichage\n{str(e)}", 
                                ha='center', va='center', transform=ax_test.transAxes,
                                fontsize=10, color='red', weight='bold')
                    chart_displayed = False
                
                if not chart_displayed:
                    alt_text = generate_alt_text_for_pie(sizes, labels, "Split Test")
                    st.warning(f"**Description du graphique:** {alt_text}")
                
                st.pyplot(fig_test)
                plt.close(fig_test)
                
                pie_data_for_table.append(["Test", 0, "0%", "Non utilisé (annotations privées)"])
            
            with col4:
                # Pie chart Coarse (non utilisé)
                fig_coarse, ax_coarse = plt.subplots(figsize=uniform_figsize, tight_layout=True)
                
                coarse_total = splits_info['coarse']['images']
                
                sizes = [coarse_total]
                labels = [f'Non utilisé dans notre projet\n{coarse_total} images\n(annotations grossières)']
                colors = [accessible_colors['light_gray']]
                
                try:
                    wedges, texts, autotexts = ax_coarse.pie(
                        sizes, labels=labels, colors=colors, autopct='',
                        startangle=90, shadow=True, textprops=uniform_textprops,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                    )
                    
                    if use_patterns and patterns[4]:
                        wedges[0].set_hatch(patterns[4])
                    
                    ax_coarse.set_title(f'Split Coarse Original\n{coarse_total} images total', 
                                      **uniform_title_props, color='#333333')
                    ax_coarse.axis('equal')
                    
                    chart_displayed = True
                    
                except Exception as e:
                    ax_coarse.text(0.5, 0.5, f"Erreur d'affichage\n{str(e)}", 
                                  ha='center', va='center', transform=ax_coarse.transAxes,
                                  fontsize=10, color='red', weight='bold')
                    chart_displayed = False
                
                if not chart_displayed:
                    alt_text = generate_alt_text_for_pie(sizes, labels, "Split Coarse")
                    st.warning(f"**Description du graphique:** {alt_text}")
                
                st.pyplot(fig_coarse)
                plt.close(fig_coarse)
                
                pie_data_for_table.append(["Coarse", 0, "0%", "Non utilisé (annotations grossières)"])
            
            
            
            
            st.markdown("### 📊 Exploration Interactive du Dataset")
    
            fig_dataset = create_dataset_interactive_overview()
            
            try:
                st.plotly_chart(fig_dataset, use_container_width=True, key="dataset_overview")
            except Exception as e:
                st.warning("**Description :** Graphique circulaire interactif montrant la répartition de l'utilisation du dataset Cityscapes dans notre projet.")
            st.info("💡 **Interactivité :** Survolez les sections pour voir les détails, cliquez sur la légende pour masquer/afficher des catégories.")
            
            # Tableau de données accessible (toujours affiché)
            st.markdown("#### 📋 Tableau récapitulatif des données")
            
            create_data_table(
                pie_data_for_table,
                ["Split", "Images utilisées", "Pourcentage", "Utilisation"],
                "Répartition du dataset Cityscapes",
                "Détail de l'utilisation de chaque split du dataset original dans notre projet"
            )

        
                       
        # Architecture des classes
        st.markdown("### 🏗️ Architecture de classes hiérarchique")
        st.markdown("""
        Le dataset organise 34 classes détaillées en **8 méta-classes** pour la segmentation :

        | Méta-classe | Description | Importance |
        |-------------|-------------|------------|
        | **Flat** | Route, trottoir, parkings | Navigation de base |
        | **Human** | Piéton, cycliste | **Sécurité critique** |
        | **Vehicle** | Voiture, camion, bus, train | Obstacles dynamiques |
        | **Construction** | Bâtiment, mur, clôture, pont | Structure urbaine |
        | **Object** | Poteau, feu, panneau | Signalétique |
        | **Nature** | Végétation, terrain | Environnement |
        | **Sky** | Ciel | Contexte |
        | **Void** | Zones non-définies | Exclusion |
        """)
        
        # Graphique de répartition des classes
        st.markdown("### 📊 Répartition des classes dans le dataset d'entraînement")
        
        class_repartition_data = get_class_repartition()
        if class_repartition_data:
            try:
                class_repartition_image = Image.open(io.BytesIO(class_repartition_data))
                st.image(
                    class_repartition_image,
                    caption="Distribution des classes dans les images d'entraînement Cityscapes",
                    use_container_width=True
                )
            except Exception as e:
                st.warning("Impossible d'afficher le graphique de répartition des classes")
        else:
            st.info("Graphique de répartition des classes non disponible")
    
    # Section preprocessing des images
    st.markdown("---")
    st.markdown("### 🔧 Exemple de Preprocessing d'Images")
    
    # Récupérer les images d'entraînement disponibles
    train_data = get_train_images()
    if train_data and train_data['images']:
        # Sélecteur d'image
        image_options = {img['display_name']: img['filename'] for img in train_data['images']}
        selected_display_name = st.selectbox(
            "Choisissez une image d'entraînement pour voir les transformations:",
            options=list(image_options.keys()),
            help=f"{train_data['total_count']} images disponibles",
            key="train_image_selector"
        )
        
        selected_filename = image_options[selected_display_name]
        
        # Preview de l'image sélectionnée (avant preprocessing)
        if selected_filename and "preprocessing_result" not in st.session_state:
            try:
                image_url = f"{API_BASE_URL}/train-image/{selected_filename}"
                response = requests.get(image_url, timeout=5)
                
                if response.status_code == 200:
                    preview_image = Image.open(io.BytesIO(response.content))
                    st.image(
                        preview_image, 
                        caption=f"Aperçu de l'image sélectionnée : {selected_display_name}", 
                        use_container_width=True
                    )
            except:
                st.info("Aperçu non disponible")
        
        # Options de preprocessing
        col1, col2 = st.columns(2)
        with col1:
            show_augmentation = st.checkbox(
                "Inclure l'augmentation aléatoire",
                help="Affiche un exemple d'augmentation de données utilisée pendant l'entraînement. Cliquez plusieurs fois sur 'Appliquer le Preprocessing' pour tester différentes augmentations aléatoires !",
                key="augmentation_checkbox"
            )
        
        with col2:
            if st.button("🔧 Appliquer le Preprocessing", type="primary", key="apply_preprocessing"):
                with st.spinner("Application des transformations..."):
                    result = preprocess_train_image(selected_filename, show_augmentation)
                    
                    if result and result['success']:
                        st.session_state.preprocessing_result = result
                        st.success("Preprocessing terminé!")
                        st.rerun()
                    else:
                        st.error("Erreur lors du preprocessing")
        
        # Bouton pour revenir à l'aperçu
        if "preprocessing_result" in st.session_state:
            if st.button("🔄 Choisir une nouvelle image", type="secondary", key="reset_preprocessing"):
                del st.session_state.preprocessing_result
                st.rerun()
        
        # Affichage des résultats de preprocessing
        if "preprocessing_result" in st.session_state:
            result = st.session_state.preprocessing_result
            
            # Affichage des images transformées sur toute la largeur
            st.markdown("#### Résultats des transformations :")
            
            if show_augmentation and result.get('augmented_image'):
                cols = st.columns(3)
                col_titles = ["Image Originale (redimensionnée)", "Après Normalisation", "Avec Augmentation Aléatoire"]
                col_data = [result.get('original_image'), result.get('normalized_image'), result.get('augmented_image')]
            else:
                cols = st.columns(2)
                col_titles = ["Image Originale (redimensionnée)", "Après Normalisation"]
                col_data = [result.get('original_image'), result.get('normalized_image')]
            
            for i, (col, title, data) in enumerate(zip(cols, col_titles, col_data)):
                with col:
                    st.markdown(f"**{title}**")
                    if data:
                        img_data = base64.b64decode(data)
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)           
            

            # Explications des transformations appliquées
            if result.get('transformations_applied'):
                st.markdown("#### 📋 Transformations appliquées :")
                
                # Préparer les transformations pour l'alignement
                transforms = result['transformations_applied']
                
                # Extraire les transformations de base (toujours présentes)
                base_transforms = [t for t in transforms if "Resize" in t or "Normalization" in t]
                augmentation_transforms = [t for t in transforms if "Resize" not in t and "Normalization" not in t]
                
                if show_augmentation and result.get('augmented_image'):
                    # 3 colonnes quand augmentation active
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Image Originale (redimensionnée)**")
                        resize_transform = next((t for t in base_transforms if "Resize" in t), "1. Resize (512x512)")
                        st.write(f"• {resize_transform}")
                    
                    with col2:
                        st.markdown("**Après Normalisation**")
                        norm_transform = next((t for t in base_transforms if "Normalization" in t), "2. Normalization (0-1)")
                        st.write(f"• {norm_transform}")
                    
                    with col3:
                        st.markdown("**Avec Augmentation Aléatoire**")
                        if augmentation_transforms:
                            for i, transform in enumerate(augmentation_transforms, 3):
                                st.write(f"• {i}. {transform}")
                        else:
                            st.write("• 3. Aucune augmentation appliquée (aléatoire)")
                
                else:
                    # 2 colonnes quand pas d'augmentation
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Image Originale (redimensionnée)**")
                        resize_transform = next((t for t in base_transforms if "Resize" in t), "1. Resize (512x512)")
                        st.write(f"• {resize_transform}")
                    
                    with col2:
                        st.markdown("**Après Normalisation**")
                        norm_transform = next((t for t in base_transforms if "Normalization" in t), "2. Normalization (0-1)")
                        st.write(f"• {norm_transform}")
                        
            # Explications détaillées avec description complète des augmentations
            with st.expander("ℹ️ Détails des transformations possibles"):
                st.markdown("""
                **Normalisation standard (toujours appliquée) :**
                - **Redimensionnement** : Toutes les images sont redimensionnées à 512×512 pixels
                - **Normalisation** : Les valeurs de pixels sont normalisées de [0-255] vers [0-1]
                - Cette normalisation uniforme est appliquée à tous les modèles (paramètre encoder non utilisé)
                
                **Augmentations de données possibles (si activées) :**
                
                Le système sélectionne aléatoirement entre 1 et 4 transformations parmi les suivantes :
                
                1. **Rotation aléatoire** : Rotation de l'image entre -15° et +15°
                2. **Flip horizontal** : Miroir horizontal avec 50% de probabilité 
                3. **Changement de luminosité/contraste** : Ajustement de ±20% de la luminosité et du contraste
                4. **Flou gaussien léger** : Application d'un flou gaussien subtil (blur 1-3 pixels)
                5. **Décalage couleur** : Modification de la teinte, saturation et valeur (±10-15%)
                6. **Zoom aléatoire** : Zoom de -10% à +10% (avec probabilité réduite)
                
                **Important :** Les augmentations sont **aléatoires** à chaque application. Cliquez plusieurs fois sur 
                "Appliquer le Preprocessing" pour voir différentes combinaisons ! Le résultat peut inclure une ou 
                plusieurs transformations, ou même aucune selon le hasard.
                
                Ces transformations permettent d'améliorer la robustesse et la généralisation du modèle pendant 
                l'entraînement en exposant le modèle à des variations réalistes des données.
                """)
    else:
        st.warning("Aucune image d'entraînement disponible pour la démonstration")

# ===== ONGLET 2: COMPARAISON DE PERFORMANCES =====
with main_tab2:
    # Variables de session pour les résultats
    if "sample_comparison_result" not in st.session_state:
        st.session_state.sample_comparison_result = None
    if "upload_comparison_result" not in st.session_state:
        st.session_state.upload_comparison_result = None

    def reset_sample_comparison():
        st.session_state.sample_comparison_result = None

    def reset_upload_comparison():
        st.session_state.upload_comparison_result = None

    # Sous-onglets pour la comparaison
    sub_tab1, sub_tab2 = st.tabs(["🏙️ Images Test Cityscapes", "📤 Upload Personnel"])

    # SUB-TAB 1: Images Test Cityscapes
    with sub_tab1:
        st.markdown("## 🏙️ Comparaison sur Images Test Cityscapes")
        st.markdown("Comparez les performances des deux modèles sur des images Cityscapes avec masques de vérité.")
        
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
            models_info = get_models_info()
            display_comparison_results(st.session_state.sample_comparison_result, models_info, "sample")
            
            # Bouton de reset
            if st.button("🔄 Nouvelle Comparaison", key="reset_sample", type="secondary"):
                reset_sample_comparison()
                st.rerun()

    # SUB-TAB 2: Upload Personnel
    with sub_tab2:
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
            models_info = get_models_info()
            display_comparison_results(st.session_state.upload_comparison_result, models_info, "upload")
            
            # Bouton de reset
            if st.button("🔄 Nouvelle Comparaison", key="reset_upload", type="secondary"):
                reset_upload_comparison()
                st.rerun()

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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚗 <strong>Street Vision - Comparaison de Modèles</strong> - 
    Évaluation comparative de modèles de segmentation sémantique<br>
    <small>Développé avec FastAPI, Streamlit et PyTorch</small>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("---")
    st.markdown("## ♿ Conformité WCAG 2.1")
    
    criteria_status = {
        "1.1.1 Contenu non textuel": "✅",
        "1.4.1 Utilisation de la couleur": "✅", 
        "1.4.3 Contraste (minimum)": "✅",
        "1.4.4 Redimensionnement du texte": "✅",
        "2.4.2 Titre de page": "✅"
    }
    
    for criterion, status in criteria_status.items():
        st.write(f"{status} {criterion}")
    
    st.markdown("**Niveau :** AA | **Référence :** [WCAG 2.1](https://www.w3.org/WAI/WCAG21/quickref/)")

