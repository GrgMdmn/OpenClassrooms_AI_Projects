import os
import sys
import traceback
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import mlflow
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Configuration MLflow avec variables d'environnement
def configure_mlflow():
    """Configure MLflow avec les variables d'environnement"""
    print("=== DEBUG VARIABLES D'ENVIRONNEMENT ===")
    
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "").strip()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    
    print(f"MLFLOW_TRACKING_URI: {mlflow_tracking_uri}")
    print(f"MLFLOW_S3_ENDPOINT_URL: {mlflow_s3_endpoint_url}")
    print(f"AWS_ACCESS_KEY_ID: {aws_access_key_id}")
    print(f"AWS_SECRET_ACCESS_KEY: {'***' if aws_secret_access_key else None} ({'set' if aws_secret_access_key else 'not set'})")
    print("============================================")
    
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not found in environment variables")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    
    if not mlflow_s3_endpoint_url or not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("Variables d'environnement MLflow manquantes (S3_ENDPOINT, ACCESS_KEY, SECRET_KEY)")
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    print(f"MLflow S3 Endpoint: {mlflow_s3_endpoint_url}")
    print("✅ Identifiants AWS configurés")
    print("✅ Configuration MLflow terminée")

# Import des fonctions utils
sys.path.append('.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    load_cityscapes_config, 
    load_model,
    run_inference_and_visualize,
    colorize_mask
)

# Configuration
MODEL_NAME = "StreetsSegmentation"
SAMPLE_IMAGES_DIR = "./notebooks/content/data/test_images_sample"
TRAIN_IMAGES_DIR = "./notebooks/content/data/train_images_sample"
CITYSCAPES_CONFIG_PATH = "./cityscapes_config.json"
TEMP_UPLOAD_DIR = "/tmp/uploads"

# Modèles de données mis à jour
class ModelInfo(BaseModel):
    name: str
    run_id: str
    encoder_name: str
    input_size: tuple
    rank: int
    architecture: Optional[str] = None
    test_mean_iou: Optional[float] = None
    test_accuracy: Optional[float] = None
    class_ious: Optional[dict] = None

class ComparisonResult(BaseModel):
    success: bool
    message: str
    model1_info: ModelInfo
    model2_info: ModelInfo
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    model1_stats: Optional[dict] = None
    model2_stats: Optional[dict] = None
    model1_inference_time: float
    model2_inference_time: float
    speed_comparison: dict
    ground_truth_available: bool = False
    figure_data: Optional[str] = None

class ImageInfo(BaseModel):
    filename: str
    display_name: str
    has_ground_truth: bool

class AvailableImagesResponse(BaseModel):
    images: List[ImageInfo]
    total_count: int

class PreprocessingRequest(BaseModel):
    filename: str
    apply_augmentation: bool = False

class PreprocessingResult(BaseModel):
    success: bool
    message: str
    original_image: Optional[str] = None
    normalized_image: Optional[str] = None
    augmented_image: Optional[str] = None
    transformations_applied: Optional[list] = None

class SampleImageRequest(BaseModel):
    filename: str

# Variables globales pour les deux modèles
best_model = None
second_best_model = None
best_model_info = {}
second_best_model_info = {}
mapping_config = None

def get_model_detailed_info(run_id, mapping_config):
    """Récupère les informations détaillées d'un modèle depuis MLflow"""
    client = mlflow.tracking.MlflowClient()
    
    try:
        run = mlflow.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics
        
        # Architecture depuis le paramètre model_architecture
        architecture = params.get("model_architecture", "Unknown").split('_')[0]
        
        # Métriques de performance
        test_mean_iou = metrics.get("test_mean_iou", None)
        test_accuracy = metrics.get("test_accuracy", None)
        
        # IoU par classe depuis les métriques class_X_iou
        class_ious = {}
        if mapping_config and 'group_names' in mapping_config:
            for i, class_name in enumerate(mapping_config['group_names']):
                metric_key = f"class_{i}_iou"
                if metric_key in metrics:
                    class_ious[class_name] = round(metrics[metric_key], 3)
        
        return {
            "architecture": architecture,
            "test_mean_iou": test_mean_iou,
            "test_accuracy": test_accuracy,
            "class_ious": class_ious
        }
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des infos détaillées: {e}")
        return {
            "architecture": "Unknown",
            "test_mean_iou": None,
            "test_accuracy": None,
            "class_ious": {}
        }

def initialize_models():
    """Initialise les deux meilleurs modèles au démarrage"""
    global best_model, second_best_model, best_model_info, second_best_model_info, mapping_config
    
    try:
        print("🔄 Configuration de MLflow...")
        configure_mlflow()
        
        print("🔄 Chargement de la configuration Cityscapes avec couleurs WCAG...")
        mapping_config = load_cityscapes_config(CITYSCAPES_CONFIG_PATH, verbose=False, use_wcag_colors=True)
        print("✅ Configuration Cityscapes chargée avec couleurs WCAG")

        
        print("🔄 Chargement du meilleur modèle...")
        best_model, best_run_id, best_encoder_name, best_img_size = load_model(
            experiment_name="OC Projet 9", 
            metric="test_mean_iou", 
            top_n=1
        )
        
        # Récupérer les infos détaillées du meilleur modèle
        best_detailed_info = get_model_detailed_info(best_run_id, mapping_config)
        
        best_model_info = {
            "name": "Meilleur modèle",
            "run_id": best_run_id,
            "encoder_name": best_encoder_name,
            "input_size": best_img_size,
            "rank": 1,
            "architecture": best_detailed_info["architecture"],
            "test_mean_iou": best_detailed_info["test_mean_iou"],
            "test_accuracy": best_detailed_info["test_accuracy"],
            "class_ious": best_detailed_info["class_ious"]
        }
        
        print("🔄 Chargement du deuxième meilleur modèle...")
        second_best_model, second_run_id, second_encoder_name, second_img_size = load_model(
            experiment_name="OC Projet 9", 
            metric="test_mean_iou", 
            top_n=2
        )
        
        # Récupérer les infos détaillées du deuxième modèle
        second_detailed_info = get_model_detailed_info(second_run_id, mapping_config)
        
        second_best_model_info = {
            "name": "Deuxième modèle",
            "run_id": second_run_id,
            "encoder_name": second_encoder_name,
            "input_size": second_img_size,
            "rank": 2,
            "architecture": second_detailed_info["architecture"],
            "test_mean_iou": second_detailed_info["test_mean_iou"],
            "test_accuracy": second_detailed_info["test_accuracy"],
            "class_ious": second_detailed_info["class_ious"]
        }
        
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        traceback.print_exc()
        raise e
        
def get_accessibility_info():
    """Retourne les informations d'accessibilité pour le frontend"""
    return {
        "wcag_compliant": True,
        "color_palette": {
            "primary_blue": "#0066CC",
            "primary_red": "#CC0000", 
            "primary_green": "#006600",
            "primary_orange": "#CC6600",
            "secondary_purple": "#663399",
            "secondary_teal": "#006666",
            "neutral_dark": "#333333",
            "neutral_medium": "#666666"
        },
        "contrast_ratios": {
            "primary_colors_vs_white": "≥ 4.1:1 (WCAG AA)",
            "text_vs_background": "≥ 4.5:1 (WCAG AA)"
        },
        "features": [
            "Couleurs à contraste élevé",
            "Motifs alternatifs aux couleurs",
            "Descriptions textuelles alternatives en cas d'absence d'images",
            "Tableaux de données complémentaires",
            "Navigation clavier supportée"
        ]
    }


# Initialisation de l'API
app = FastAPI(
    title="Multi-Class Segmentation Comparison API",
    description="API de comparaison de segmentation sémantique avec deux modèles",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    try:
        initialize_models()
        print("🚀 API démarrée avec succès - Modèles chargés!")
    except Exception as e:
        print(f"⚠️ ATTENTION: Impossible de charger les modèles au démarrage: {e}")

@app.get("/")
def root():
    return {
        "message": "Multi-Class Segmentation Comparison API - Online",
        "models_loaded": best_model is not None and second_best_model is not None,
        "best_model": best_model_info if best_model else None,
        "second_best_model": second_best_model_info if second_best_model else None,
        "num_classes": mapping_config['num_classes'] if mapping_config else None
    }

@app.get("/health")
def health_check():
    if best_model is None or second_best_model is None or mapping_config is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }


# Modifier le endpoint /models pour inclure des couleurs accessibles :
@app.get("/models", response_model=dict)
def get_models_info():
    if best_model is None or second_best_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # NOUVELLE RÉPARTITION DES COULEURS selon votre mapping final
    wcag_accessible_colors = [
        [102, 102, 102],  # Flat: neutral_medium (gris moyen pour routes/trottoirs)
        [204, 0, 0],      # Human: primary_red (rouge pour piétons/cyclistes)  
        [102, 51, 153],   # Vehicle: secondary_purple (violet pour véhicules)
        [204, 102, 0],    # Construction: primary_orange (orange pour bâtiments)
        [0, 102, 102],    # Object: secondary_teal (teal pour objets/signalétique)
        [0, 102, 0],      # Nature: primary_green (vert pour végétation)
        [0, 102, 204],    # Sky: primary_blue (bleu pour ciel)
        [51, 51, 51]      # Void: neutral_dark (gris foncé pour zones non-définies)
    ]
    
    # Mettre à jour la configuration avec les nouvelles couleurs
    if mapping_config:
        mapping_config['group_colors'] = np.array(wcag_accessible_colors, dtype=np.uint8)
    
    return {
        "best_model": best_model_info,
        "second_best_model": second_best_model_info,
        "num_classes": mapping_config['num_classes'],
        "class_names": mapping_config['group_names'],
        "class_colors": wcag_accessible_colors,
        "accessibility_info": get_accessibility_info(),
        "preprocessing_resolution": best_model_info.get("input_size", (512, 512))
    }



@app.get("/dataset-info")
def get_dataset_info():
    """Retourne les informations sur le dataset"""
    return {
        "name": "Cityscapes Dataset",
        "description": "Référence internationale pour l'évaluation des algorithmes de segmentation sémantique en environnement urbain",
        "total_images": 25000,
        "resolution": "2048×1024 pixels",
        "cities": 50,
        "countries": ["Allemagne", "Suisse"],
        "splits": {
            "train": {"images": 2975, "usage": "Entraînement avec annotations fines"},
            "validation": {"images": 500, "usage": "Validation avec annotations de référence"},
            "test": {"images": 1525, "usage": "Test (annotations non-publiques)"},
            "coarse": {"images": 20000, "usage": "Entraînement avec annotations grossières"}
        },
        "project_splits": {
            "train": {"images": 2380, "percentage": 80, "source": "Cityscapes train"},
            "validation": {"images": 595, "percentage": 20, "source": "Cityscapes train"},
            "test": {"images": 500, "percentage": 100, "source": "Cityscapes val"}
        },
        "meta_classes": mapping_config['group_names'] if mapping_config else [],
        "num_classes": mapping_config['num_classes'] if mapping_config else 0
    }

@app.get("/train-images", response_model=AvailableImagesResponse)
def get_train_images():
    """Retourne la liste des images d'entraînement disponibles"""
    try:
        original_dir = Path(TRAIN_IMAGES_DIR) / "original"
        mask_dir = Path(TRAIN_IMAGES_DIR) / "mask"
        
        if not original_dir.exists():
            raise HTTPException(status_code=404, detail="Train images directory not found")
        
        images = []
        for img_path in original_dir.glob("*leftImg8bit.png"):
            # Construire le nom du masque correspondant
            base_name = img_path.stem.replace("_leftImg8bit", "")
            mask_name = f"{base_name}_gtFine_labelIds.png"
            mask_path = mask_dir / mask_name
            
            # Créer un nom d'affichage plus convivial
            display_name = base_name.replace("_", " ").title()
            
            images.append(ImageInfo(
                filename=img_path.name,
                display_name=display_name,
                has_ground_truth=mask_path.exists()
            ))
        
        # Trier par nom pour un affichage cohérent
        images.sort(key=lambda x: x.filename)
        
        return AvailableImagesResponse(
            images=images,
            total_count=len(images)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing train images: {str(e)}")

@app.get("/train-image/{filename}")
def get_train_image(filename: str):
    """Sert une image d'entraînement pour prévisualisation"""
    try:
        original_dir = Path(TRAIN_IMAGES_DIR) / "original"
        image_path = original_dir / filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Train image {filename} not found")
        
        # Vérifier que c'est bien un fichier image attendu
        if not filename.endswith('_leftImg8bit.png'):
            raise HTTPException(status_code=400, detail="Invalid image filename format")
        
        return FileResponse(
            image_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving train image: {str(e)}")

@app.get("/class-repartition")
def get_class_repartition():
    """Sert le graphique de répartition des classes"""
    try:
        graph_path = Path(TRAIN_IMAGES_DIR) / "class_repartition.png"
        
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail="Class repartition graph not found")
        
        return FileResponse(
            graph_path,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=class_repartition.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving class repartition graph: {str(e)}")

@app.post("/preprocess-train-image", response_model=PreprocessingResult)
def preprocess_train_image(request: PreprocessingRequest):
    """Prétraite une image d'entraînement et retourne les versions transformées"""
    if mapping_config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    # Vérifier que le meilleur modèle est chargé
    if best_model is None or not best_model_info:
        raise HTTPException(status_code=503, detail="Best model not loaded")
    
    try:
        original_dir = Path(TRAIN_IMAGES_DIR) / "original"
        image_path = original_dir / request.filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Train image {request.filename} not found")
        
        # Importer les fonctions nécessaires
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        import base64
        from utils.utils import get_preprocessing_fn, apply_random_augmentation
        
        # Charger l'image originale
        original_pil = Image.open(image_path).convert('RGB')
        original_np = np.array(original_pil)
        
        # UTILISER LA RÉSOLUTION DU MEILLEUR MODÈLE au lieu de (512, 512)
        target_size = best_model_info.get("input_size", (512, 512))
        print(f"🎯 Utilisation de la taille du meilleur modèle: {target_size}")
        
        # Appliquer le preprocessing avec la résolution du meilleur modèle
        preprocess_fn = get_preprocessing_fn(img_size=target_size, encoder=None)
        normalized_tensor, _ = preprocess_fn(original_np, None)
        
        # Convertir le tensor en image pour affichage
        normalized_np = normalized_tensor.permute(1, 2, 0).numpy()
        # Dénormaliser pour l'affichage (tensor était normalisé 0-1)
        normalized_display = (normalized_np * 255).astype(np.uint8)
        
        # Créer les images en base64 pour l'envoi
        def img_to_base64(img_array):
            img_pil = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        # Image originale redimensionnée pour comparaison (avec la taille du meilleur modèle)
        original_resized = np.array(original_pil.resize(target_size))
        
        result = PreprocessingResult(
            success=True,
            message="Preprocessing completed successfully",
            original_image=img_to_base64(original_resized),
            normalized_image=img_to_base64(normalized_display),
            transformations_applied=[f"Resize {target_size}", "Normalization (0-1)"]
        )
        
        # Appliquer l'augmentation si demandée
        if request.apply_augmentation:
            augmented_np, transforms_applied = apply_random_augmentation(original_np)
            # Appliquer aussi le preprocessing à l'image augmentée
            augmented_tensor, _ = preprocess_fn(augmented_np, None)
            augmented_display = (augmented_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            result.augmented_image = img_to_base64(augmented_display)
            result.transformations_applied.extend(transforms_applied)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.get("/best-model-resolution")
def get_best_model_resolution():
    """Retourne la résolution utilisée par le meilleur modèle"""
    if best_model is None or not best_model_info:
        raise HTTPException(status_code=503, detail="Best model not loaded")
    
    return {
        "model_name": best_model_info.get("name", "Unknown"),
        "encoder_name": best_model_info.get("encoder_name", "Unknown"),
        "input_size": best_model_info.get("input_size", (512, 512)),
        "architecture": best_model_info.get("architecture", "Unknown")
    }

@app.get("/sample-images", response_model=AvailableImagesResponse)
def get_sample_images():
    try:
        original_dir = Path(SAMPLE_IMAGES_DIR) / "original"
        mask_dir = Path(SAMPLE_IMAGES_DIR) / "mask"
        if not original_dir.exists():
            raise HTTPException(status_code=404, detail="Sample images directory not found")
        images = []
        for img_path in original_dir.glob("*leftImg8bit.png"):
            base_name = img_path.stem.replace("_leftImg8bit", "")
            mask_name = f"{base_name}_gtFine_labelIds.png"
            mask_path = mask_dir / mask_name
            display_name = base_name.replace("_", " ").title()
            images.append(ImageInfo(
                filename=img_path.name,
                display_name=display_name,
                has_ground_truth=mask_path.exists()
            ))
        images.sort(key=lambda x: x.filename)
        return AvailableImagesResponse(images=images, total_count=len(images))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sample images: {str(e)}")

@app.get("/sample-image/{filename}")
def get_sample_image(filename: str):
    try:
        original_dir = Path(SAMPLE_IMAGES_DIR) / "original"
        image_path = original_dir / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample image {filename} not found")
        if not filename.endswith('_leftImg8bit.png'):
            raise HTTPException(status_code=400, detail="Invalid image filename format")
        return FileResponse(
            image_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@app.post("/compare-sample", response_model=ComparisonResult)
def compare_sample_image(request: SampleImageRequest):
    if best_model is None or second_best_model is None or mapping_config is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        original_dir = Path(SAMPLE_IMAGES_DIR) / "original"
        image_path = original_dir / request.filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample image {request.filename} not found")
        base_name = image_path.stem.replace("_leftImg8bit", "")
        mask_name = f"{base_name}_gtFine_labelIds.png"
        mask_path = Path(SAMPLE_IMAGES_DIR) / "mask" / mask_name
        mask_path_str = str(mask_path) if mask_path.exists() else None
        
        # Inférence modèle 1 avec sa résolution spécifique
        pred1, t1, fig1 = run_inference_and_visualize(
            image_path=str(image_path),
            mask_path=mask_path_str,
            model=best_model,
            encoder_name=None,  # Normalisé de la même manière
            img_size=best_model_info["input_size"],  # Utilise la résolution du modèle
            mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False
        )
        # Inférence modèle 2 avec sa résolution spécifique
        pred2, t2, fig2 = run_inference_and_visualize(
            image_path=str(image_path),
            mask_path=mask_path_str,
            model=second_best_model,
            encoder_name=None,  # Normalisé de la même manière
            img_size=second_best_model_info["input_size"],  # Utilise la résolution du modèle
            mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False
        )
        
        # Créer une figure combinée avec les deux modèles
        # Nombre de colonnes : 3 si pas de GT, 4 si GT disponible
        num_cols = 3 if not (mask_path_str and os.path.exists(mask_path_str)) else 4
        fig, axes = plt.subplots(2, num_cols, figsize=(5*num_cols, 8))
        
        # Récupérer les images depuis les figures individuelles
        axes1 = fig1.get_axes()
        axes2 = fig2.get_axes()
        
        # Ligne 1 : Modèle 1
        for i in range(min(len(axes1), num_cols)):
            ax_src = axes1[i]
            ax_dst = axes[0, i]
            
            # Copier toutes les images de l'axe source
            for img_obj in ax_src.get_images():
                ax_dst.imshow(
                    img_obj.get_array(),
                    cmap=img_obj.get_cmap(),
                    alpha=img_obj.get_alpha(),
                    vmin=img_obj.get_clim()[0] if img_obj.get_clim() else None,
                    vmax=img_obj.get_clim()[1] if img_obj.get_clim() else None
                )
            
            # Copier le titre avec nom du modèle
            original_title = ax_src.get_title()
            if i == 0:
                new_title = "Image originale"
            elif "Prédit" in original_title:
                new_title = f"Prédiction {best_model_info['architecture']} - {best_model_info['encoder_name']}"
            else:
                new_title = original_title
            
            ax_dst.set_title(new_title)
            ax_dst.axis('off')
        
        # Ligne 2 : Modèle 2
        for i in range(min(len(axes2), num_cols)):
            ax_src = axes2[i]
            ax_dst = axes[1, i]
            
            # Copier toutes les images de l'axe source
            for img_obj in ax_src.get_images():
                ax_dst.imshow(
                    img_obj.get_array(),
                    cmap=img_obj.get_cmap(),
                    alpha=img_obj.get_alpha(),
                    vmin=img_obj.get_clim()[0] if img_obj.get_clim() else None,
                    vmax=img_obj.get_clim()[1] if img_obj.get_clim() else None
                )
            
            # Copier le titre avec nom du modèle
            original_title = ax_src.get_title()
            if i == 0:
                new_title = "Image originale"
            elif "Prédit" in original_title:
                new_title = f"Prédiction {second_best_model_info['architecture']} - {second_best_model_info['encoder_name']}"
            else:
                new_title = original_title
            
            ax_dst.set_title(new_title)
            ax_dst.axis('off')
        
        plt.tight_layout()
        
        # Fermer les figures individuelles pour libérer la mémoire
        plt.close(fig1)
        plt.close(fig2)
        
        fig_b64 = base64.b64encode(pickle.dumps(fig)).decode()
        # Stats
        def calc(mask):
            uc, cnts = np.unique(mask, return_counts=True)
            stats = {}
            for cid, c in zip(uc, cnts):
                if cid < len(mapping_config['group_names']):
                    stats[mapping_config['group_names'][cid]] = {
                        "pixels": int(c),
                        "percentage": round((c/mask.size)*100, 1)
                    }
            return stats
        stats1, stats2 = calc(pred1), calc(pred2)
        faster = best_model_info['encoder_name'] if t1 < t2 else second_best_model_info['encoder_name']
        speedup = round(max(t1, t2) / min(t1, t2), 2)
        speed_comp = {"faster_model": faster, "speedup_ratio": speedup, "time_difference_ms": abs(t1-t2)*1000}
        return ComparisonResult(
            success=True, message="Comparison completed successfully",
            model1_info=ModelInfo(**best_model_info), model2_info=ModelInfo(**second_best_model_info),
            image_path=str(image_path), mask_path=mask_path_str,
            model1_stats=stats1, model2_stats=stats2,
            model1_inference_time=t1, model2_inference_time=t2,
            speed_comparison=speed_comp, ground_truth_available=mask_path.exists(),
            figure_data=fig_b64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@app.post("/compare-upload", response_model=ComparisonResult)
def compare_uploaded_image(file: UploadFile = File(...)):
    if best_model is None or second_best_model is None or mapping_config is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}")
    temp_file_path = None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}_{file.filename}"
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Inférences avec résolutions spécifiques
        pred1, t1, fig1 = run_inference_and_visualize(
            image_path=temp_file_path, mask_path=None,
            model=best_model, encoder_name=None,  # Normalisé de la même manière
            img_size=best_model_info["input_size"], mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False,
        )
        pred2, t2, fig2 = run_inference_and_visualize(
            image_path=temp_file_path, mask_path=None,
            model=second_best_model, encoder_name=None,  # Normalisé de la même manière
            img_size=second_best_model_info["input_size"], mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False,
        )
        
        # Figure combinée pour uploads (pas de GT, donc 2 colonnes seulement)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Récupérer les images depuis les figures individuelles
        axes1 = fig1.get_axes()
        axes2 = fig2.get_axes()
        
        # Ligne 1 : Modèle 1 (Image originale + Prédiction)
        for i in range(min(len(axes1), 2)):
            ax_src = axes1[i]
            ax_dst = axes[0, i]
            
            # Copier toutes les images de l'axe source
            for img_obj in ax_src.get_images():
                ax_dst.imshow(
                    img_obj.get_array(),
                    cmap=img_obj.get_cmap(),
                    alpha=img_obj.get_alpha(),
                    vmin=img_obj.get_clim()[0] if img_obj.get_clim() else None,
                    vmax=img_obj.get_clim()[1] if img_obj.get_clim() else None
                )
            
            # Copier le titre avec nom du modèle
            original_title = ax_src.get_title()
            if i == 0:
                new_title = "Image originale"
            else:
                new_title = f"Prédiction {best_model_info['architecture']} - {best_model_info['encoder_name']}"
            
            ax_dst.set_title(new_title)
            ax_dst.axis('off')
        
        # Ligne 2 : Modèle 2 (Image originale + Prédiction)  
        for i in range(min(len(axes2), 2)):
            ax_src = axes2[i]
            ax_dst = axes[1, i]
            
            # Copier toutes les images de l'axe source
            for img_obj in ax_src.get_images():
                ax_dst.imshow(
                    img_obj.get_array(),
                    cmap=img_obj.get_cmap(),
                    alpha=img_obj.get_alpha(),
                    vmin=img_obj.get_clim()[0] if img_obj.get_clim() else None,
                    vmax=img_obj.get_clim()[1] if img_obj.get_clim() else None
                )
            
            # Copier le titre avec nom du modèle
            original_title = ax_src.get_title()
            if i == 0:
                new_title = "Image originale"
            else:
                new_title = f"Prédiction {second_best_model_info['architecture']} - {second_best_model_info['encoder_name']}"
            
            ax_dst.set_title(new_title)
            ax_dst.axis('off')
        
        plt.tight_layout()
        
        # Fermer les figures individuelles pour libérer la mémoire
        plt.close(fig1)
        plt.close(fig2)
        
        fig_b64 = base64.b64encode(pickle.dumps(fig)).decode()
        # Stats
        def calc(mask):
            uc, cnts = np.unique(mask, return_counts=True)
            stats = {}
            for cid, c in zip(uc, cnts):
                if cid < len(mapping_config['group_names']):
                    stats[mapping_config['group_names'][cid]] = {
                        "pixels": int(c),
                        "percentage": round((c/mask.size)*100, 1)
                    }
            return stats
        stats1, stats2 = calc(pred1), calc(pred2)
        faster = best_model_info['encoder_name'] if t1 < t2 else second_best_model_info['encoder_name']
        speedup = round(max(t1, t2) / min(t1, t2), 2)
        speed_comp = {"faster_model": faster, "speedup_ratio": speedup, "time_difference_ms": abs(t1-t2)*1000}
        return ComparisonResult(
            success=True, message="Comparison completed successfully",
            model1_info=ModelInfo(**best_model_info), model2_info=ModelInfo(**second_best_model_info),
            image_path=temp_file_path,
            model1_stats=stats1, model2_stats=stats2,
            model1_inference_time=t1, model2_inference_time=t2,
            speed_comparison=speed_comp, ground_truth_available=False,
            figure_data=fig_b64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except Exception: pass

@app.get("/metrics")
def get_metrics():
    return {
        "models_loaded": best_model is not None and second_best_model is not None,
        "best_model_info": best_model_info,
        "second_best_model_info": second_best_model_info,
        "num_classes": mapping_config['num_classes'] if mapping_config else None,
        "timestamp": datetime.utcnow().isoformat()
    }

# Ajouter cette route après les autres routes :
@app.get("/accessibility-info")
def get_accessibility_information():
    """Retourne les informations sur la conformité d'accessibilité"""
    return get_accessibility_info()