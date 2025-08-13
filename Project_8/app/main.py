import os
import sys
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import mlflow
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
import base64

# Configuration MLflow avec variables d'environnement
def configure_mlflow():
    """Configure MLflow avec les variables d'environnement"""
    print("=== DEBUG VARIABLES D'ENVIRONNEMENT ===")
    
    # ✅ NETTOYAGE automatique des variables d'environnement
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
    
    # Configuration MLflow avec URLs nettoyées
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    
    # ✅ CORRECTION: Configuration EXPLICITE des identifiants avec nettoyage
    if not mlflow_s3_endpoint_url or not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("Variables d'environnement MLflow manquantes (S3_ENDPOINT, ACCESS_KEY, SECRET_KEY)")
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    print(f"MLflow S3 Endpoint: {mlflow_s3_endpoint_url}")
    print("✅ Identifiants AWS configurés")
    
    # ✅ SKIP set_experiment pour éviter le problème 404
    # mlflow.set_experiment("OC Projet 8")  # Commenté temporairement
    print("✅ Configuration MLflow terminée (sans set_experiment)")

# Import des fonctions utils
sys.path.append('.')
from utils.utils import (
    load_cityscapes_config, 
    load_best_model_from_registry,
    run_inference_and_visualize,
    colorize_mask
)

# ✅ AJOUT: Import segmentation_models pour le chargement des modèles
try:
    import segmentation_models as sm
    print("✅ segmentation_models importé - modèles compatibles")
except ImportError as e:
    print(f"⚠️ segmentation_models non disponible: {e}")
    print("💡 Installez avec: pip install git+https://github.com/qubvel/segmentation_models.git")

# Configuration
MODEL_NAME = "StreetsSegmentation"
SAMPLE_IMAGES_DIR = "./notebooks/content/data/test_images_sample"
CITYSCAPES_CONFIG_PATH = "./cityscapes_config.json"
TEMP_UPLOAD_DIR = "/tmp/uploads"

# Modèles de données
class ImageInfo(BaseModel):
    filename: str
    display_name: str
    has_ground_truth: bool

class PredictionResult(BaseModel):
    success: bool
    message: str
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    prediction_stats: Optional[dict] = None
    ground_truth_available: bool = False
    figure_data: Optional[str] = None

class AvailableImagesResponse(BaseModel):
    images: List[ImageInfo]
    total_count: int

# Variables globales pour le modèle
model = None
encoder_name = None
script_img_size = None
mapping_config = None
run_id = None

def initialize_model():
    """Initialise le modèle et la configuration au démarrage"""
    global model, encoder_name, script_img_size, mapping_config, run_id
    
    try:
        # Configuration MLflow
        print("🔄 Configuration de MLflow...")
        configure_mlflow()
        
        print("🔄 Chargement de la configuration Cityscapes...")
        mapping_config = load_cityscapes_config(CITYSCAPES_CONFIG_PATH, verbose=False)
        print("✅ Configuration Cityscapes chargée")
        
        print("🔄 Chargement du modèle depuis MLflow...")
        print(f"🔍 Recherche du modèle: {MODEL_NAME}")
        model, run_id, encoder_name, script_img_size = load_best_model_from_registry(MODEL_NAME)
        print("✅ Modèle téléchargé depuis MLflow")
        
        print(f"✅ Modèle chargé du run {run_id}")
        print(f"📐 Encoder: {encoder_name}")
        print(f"📐 Input shape: {script_img_size}")
        print(f"🎨 Nombre de classes: {mapping_config['num_classes']}")
        
        # Créer le dossier temporaire
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        raise e

# Initialisation de l'API
app = FastAPI(
    title="Multi-Class Segmentation API",
    description="API de segmentation sémantique pour véhicules autonomes",
    version="1.0.0"
)

# ✅ AJOUT: Initialiser le modèle de manière plus robuste
@app.on_event("startup")
async def startup_event():
    """Initialise le modèle au démarrage de l'API"""
    try:
        initialize_model()
        print("🚀 API démarrée avec succès - Modèle chargé!")
    except Exception as e:
        print(f"⚠️ ATTENTION: Impossible de charger le modèle au démarrage: {e}")
        print("L'API démarre quand même mais les prédictions ne fonctionneront pas.")

@app.get("/")
def root():
    return {
        "message": "Multi-Class Segmentation API - Online",
        "model_loaded": model is not None,
        "model_run_id": run_id,
        "encoder": encoder_name,
        "num_classes": mapping_config['num_classes'] if mapping_config else None
    }

@app.get("/health")
def health_check():
    """Health check endpoint pour nginx"""
    if model is None or mapping_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/models", response_model=dict)
def get_model_info():
    """Retourne les informations du modèle chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "encoder": encoder_name,
        "input_size": script_img_size,
        "num_classes": mapping_config['num_classes'],
        "class_names": mapping_config['group_names'],
        "class_colors": mapping_config['group_colors'].tolist()
    }

@app.get("/sample-images", response_model=AvailableImagesResponse)
def get_sample_images():
    """Retourne la liste des images d'exemple disponibles"""
    try:
        original_dir = Path(SAMPLE_IMAGES_DIR) / "original"
        mask_dir = Path(SAMPLE_IMAGES_DIR) / "mask"
        
        if not original_dir.exists():
            raise HTTPException(status_code=404, detail="Sample images directory not found")
        
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
        raise HTTPException(status_code=500, detail=f"Error listing sample images: {str(e)}")

@app.get("/sample-image/{filename}")
def get_sample_image(filename: str):
    """Sert une image d'exemple pour prévisualisation"""
    try:
        print(f"🔍 DEBUG: Demande d'image: {filename}")
        
        original_dir = Path(SAMPLE_IMAGES_DIR) / "original"
        image_path = original_dir / filename
        
        print(f"🔍 DEBUG: Chemin construit: {image_path}")
        print(f"🔍 DEBUG: Fichier existe: {image_path.exists()}")
        
        if not image_path.exists():
            print(f"❌ DEBUG: Image non trouvée: {image_path}")
            raise HTTPException(status_code=404, detail=f"Sample image {filename} not found")
        
        # Vérifier que c'est bien un fichier image attendu (sécurité)
        if not filename.endswith('_leftImg8bit.png'):
            print(f"❌ DEBUG: Format invalide: {filename}")
            raise HTTPException(status_code=400, detail="Invalid image filename format")
        
        print(f"✅ DEBUG: Envoi de l'image: {filename}")
        return FileResponse(
            image_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
        
    except HTTPException:
        # Re-raise les HTTPException pour qu'elles soient gérées correctement
        raise
    except Exception as e:
        print(f"❌ Erreur lors du service de l'image {filename}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

class SampleImageRequest(BaseModel):
    filename: str

@app.post("/predict-sample", response_model=PredictionResult)
def predict_sample_image(request: SampleImageRequest):
    """Lance une prédiction sur une image d'exemple"""
    if model is None or mapping_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Vérifier que l'image existe
        original_dir = Path(SAMPLE_IMAGES_DIR) / "original"
        image_path = original_dir / request.filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample image {request.filename} not found")
        
        # Construire le chemin du masque de vérité
        base_name = image_path.stem.replace("_leftImg8bit", "")
        mask_name = f"{base_name}_gtFine_labelIds.png"
        mask_path = Path(SAMPLE_IMAGES_DIR) / "mask" / mask_name
        
        mask_path_str = str(mask_path) if mask_path.exists() else None
        
        print(f"🔮 Lancement de l'inférence sur {request.filename}")
        
        # Lancer l'inférence (sans affichage pour l'API)
        predicted_mask, fig = run_inference_and_visualize(
            image_path=str(image_path),
            mask_path=mask_path_str,
            model=model,
            encoder_name=encoder_name,
            img_size=script_img_size,
            mapping_config=mapping_config,
            save_dir=None,
            show_stats=False,
            show_plot=False  # ← AJOUTÉ pour récupérer la figure
        )
        
        # Sérialiser la figure
        fig_bytes = pickle.dumps(fig)
        fig_b64 = base64.b64encode(fig_bytes).decode()
        
        # Calculer les statistiques
        unique_classes, counts = np.unique(predicted_mask, return_counts=True)
        total_pixels = predicted_mask.size
        
        stats = {}
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(mapping_config['group_names']):
                percentage = (count / total_pixels) * 100
                stats[mapping_config['group_names'][class_id]] = {
                    "pixels": int(count),
                    "percentage": round(percentage, 1)
                }
        
        return PredictionResult(
            success=True,
            message="Prediction completed successfully",
            image_path=str(image_path),
            mask_path=mask_path_str,
            prediction_stats=stats,
            ground_truth_available=mask_path.exists(),
            figure_data=fig_b64
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-upload", response_model=PredictionResult)
def predict_uploaded_image(file: UploadFile = File(...)):
    """Lance une prédiction sur une image uploadée"""
    if model is None or mapping_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Vérifier le type de fichier
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    temp_file_path = None
    try:
        # Créer un fichier temporaire
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}_{file.filename}"
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
        
        # Sauvegarder le fichier uploadé
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🔮 Lancement de l'inférence sur {file.filename}")
        
        # Lancer l'inférence (sans masque de vérité)
        predicted_mask, fig = run_inference_and_visualize(
            image_path=temp_file_path,
            mask_path=None,  # Pas de masque de vérité pour les uploads
            model=model,
            encoder_name=encoder_name,
            img_size=script_img_size,
            mapping_config=mapping_config,
            save_dir=None,
            show_stats=False,
            show_plot=False  # ← AJOUTÉ pour récupérer la figure
        )
        
        # Sérialiser la figure
        fig_bytes = pickle.dumps(fig)
        fig_b64 = base64.b64encode(fig_bytes).decode()
        
        # Calculer les statistiques
        unique_classes, counts = np.unique(predicted_mask, return_counts=True)
        total_pixels = predicted_mask.size
        
        stats = {}
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(mapping_config['group_names']):
                percentage = (count / total_pixels) * 100
                stats[mapping_config['group_names'][class_id]] = {
                    "pixels": int(count),
                    "percentage": round(percentage, 1)
                }
        
        return PredictionResult(
            success=True,
            message="Prediction completed successfully",
            image_path=temp_file_path,
            prediction_stats=stats,
            ground_truth_available=False,
            figure_data=fig_b64
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    finally:
        # Nettoyer le fichier temporaire
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"🗑️ Fichier temporaire supprimé: {temp_filename}")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer le fichier temporaire: {e}")

@app.get("/metrics")
def get_metrics():
    """Endpoint de métriques pour monitoring"""
    return {
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "num_classes": mapping_config['num_classes'] if mapping_config else None,
        "timestamp": datetime.utcnow().isoformat()
    }