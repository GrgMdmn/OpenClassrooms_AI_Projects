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
CITYSCAPES_CONFIG_PATH = "./cityscapes_config.json"
TEMP_UPLOAD_DIR = "/tmp/uploads"

# Modèles de données mis à jour
class ModelInfo(BaseModel):
    name: str
    run_id: str
    encoder_name: str
    input_size: tuple
    rank: int

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

# Variables globales pour les deux modèles
best_model = None
second_best_model = None
best_model_info = {}
second_best_model_info = {}
mapping_config = None

def initialize_models():
    """Initialise les deux meilleurs modèles au démarrage"""
    global best_model, second_best_model, best_model_info, second_best_model_info, mapping_config
    
    try:
        print("🔄 Configuration de MLflow...")
        configure_mlflow()
        
        print("🔄 Chargement de la configuration Cityscapes...")
        mapping_config = load_cityscapes_config(CITYSCAPES_CONFIG_PATH, verbose=False)
        print("✅ Configuration Cityscapes chargée")
        
        print("🔄 Chargement du meilleur modèle...")
        best_model, best_run_id, best_encoder_name, best_img_size = load_model(
            experiment_name="OC Projet 9", 
            metric="test_mean_iou", 
            top_n=1
        )
        best_model_info = {
            "name": "Meilleur modèle",
            "run_id": best_run_id,
            "encoder_name": best_encoder_name,
            "input_size": best_img_size,
            "rank": 1
        }
        
        print("🔄 Chargement du deuxième meilleur modèle...")
        second_best_model, second_run_id, second_encoder_name, second_img_size = load_model(
            experiment_name="OC Projet 9", 
            metric="test_mean_iou", 
            top_n=2
        )
        second_best_model_info = {
            "name": "Deuxième modèle",
            "run_id": second_run_id,
            "encoder_name": second_encoder_name,
            "input_size": second_img_size,
            "rank": 2
        }
        
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        traceback.print_exc()
        raise e

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

@app.get("/models", response_model=dict)
def get_models_info():
    if best_model is None or second_best_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "best_model": best_model_info,
        "second_best_model": second_best_model_info,
        "num_classes": mapping_config['num_classes'],
        "class_names": mapping_config['group_names'],
        "class_colors": mapping_config['group_colors'].tolist()
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

class SampleImageRequest(BaseModel):
    filename: str

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
        
        # Inférence modèle 1
        pred1, t1, _ = run_inference_and_visualize(
            image_path=str(image_path),
            mask_path=mask_path_str,
            model=best_model,
            encoder_name=None,
            img_size=best_model_info["input_size"],
            mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False
        )
        # Inférence modèle 2
        pred2, t2, _ = run_inference_and_visualize(
            image_path=str(image_path),
            mask_path=mask_path_str,
            model=second_best_model,
            encoder_name=None,
            img_size=second_best_model_info["input_size"],
            mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False
        )
        # Figure combinée
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        img_arr = mpimg.imread(str(image_path))
        axes[0, 0].imshow(img_arr); axes[0, 0].set_title("Image originale"); axes[0, 0].axis('off')
        axes[1, 0].imshow(img_arr); axes[1, 0].set_title("Image originale"); axes[1, 0].axis('off')
        axes[0, 1].imshow(colorize_mask(pred1, mapping_config['group_colors']))
        axes[0, 1].set_title(f"Prédiction {best_model_info['encoder_name']}"); axes[0, 1].axis('off')
        axes[1, 1].imshow(colorize_mask(pred2, mapping_config['group_colors']))
        axes[1, 1].set_title(f"Prédiction {second_best_model_info['encoder_name']}"); axes[1, 1].axis('off')
        if mask_path_str and os.path.exists(mask_path_str):
            gt = mpimg.imread(mask_path_str)
            axes[0, 2].imshow(gt); axes[0, 2].set_title("Ground Truth"); axes[0, 2].axis('off')
            axes[1, 2].imshow(gt); axes[1, 2].set_title("Ground Truth"); axes[1, 2].axis('off')
        plt.tight_layout()
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
        # Inférences
        pred1, t1, _ = run_inference_and_visualize(
            image_path=temp_file_path, mask_path=None,
            model=best_model, encoder_name=None,
            img_size=best_model_info["input_size"], mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False,
        )
        pred2, t2, _ = run_inference_and_visualize(
            image_path=temp_file_path, mask_path=None,
            model=second_best_model, encoder_name=None,
            img_size=second_best_model_info["input_size"], mapping_config=mapping_config,
            save_dir=None, show_stats=False, show_plot=False,
        )
        # Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        img_arr = mpimg.imread(temp_file_path)
        axes[0].imshow(img_arr); axes[0].set_title("Image originale"); axes[0].axis('off')
        axes[1].imshow(colorize_mask(pred1, mapping_config['group_colors']))
        axes[1].set_title(f"Prédiction {best_model_info['encoder_name']}"); axes[1].axis('off')
        axes[2].imshow(colorize_mask(pred2, mapping_config['group_colors']))
        axes[2].set_title(f"Prédiction {second_best_model_info['encoder_name']}"); axes[2].axis('off')
        plt.tight_layout()
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
