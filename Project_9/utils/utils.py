import os
import json
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import mlflow
import re
import ast
import time
import random

# Visualisation
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Preprocessing/Images
import albumentations as A
from albumentations.pytorch import ToTensorV2


def update_colors_to_wcag(mapping_config):
    """
    Met à jour les couleurs dans mapping_config pour utiliser les couleurs WCAG accessibles
    avec redistribution sémantique intuitive finale
    """
    # Redistribution sémantique finale :
    # Flat → gris moyen, Human → rouge, Vehicle → violet, Construction → orange,
    # Object → teal, Nature → vert, Sky → bleu, Void → gris foncé
    wcag_colors = np.array([
        [102, 102, 102],  # Flat: gris moyen (routes, trottoirs)
        [204, 0, 0],      # Human: rouge (piétons, cyclistes)  
        [102, 51, 153],   # Vehicle: violet (véhicules)
        [204, 102, 0],    # Construction: orange (bâtiments)
        [0, 102, 102],    # Object: teal (objets, signalétique)
        [0, 102, 0],      # Nature: vert (végétation)
        [0, 102, 204],    # Sky: bleu (ciel)
        [51, 51, 51]      # Void: gris foncé (zones non-définies)
    ], dtype=np.uint8)
    
    # S'assurer qu'on a assez de couleurs pour le nombre de classes
    num_classes = mapping_config['num_classes']
    if len(wcag_colors) < num_classes:
        print(f"⚠️ Attention: {num_classes} classes mais seulement {len(wcag_colors)} couleurs WCAG disponibles")
        # Compléter avec les couleurs originales si nécessaire
        original_colors = mapping_config['group_colors']
        while len(wcag_colors) < num_classes:
            wcag_colors = np.vstack([wcag_colors, original_colors[len(wcag_colors):len(wcag_colors)+1]])
    
    mapping_config['group_colors'] = wcag_colors[:num_classes]
    print(f"✅ Couleurs mises à jour vers WCAG pour {num_classes} classes")
    return mapping_config


def load_cityscapes_config(config_path="../cityscapes_config.json", verbose=True, use_wcag_colors=False):
    """
    Charge la configuration Cityscapes depuis un fichier JSON.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        verbose: Affichage détaillé
        use_wcag_colors: Utiliser les couleurs WCAG accessibles
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_groups = config["class_groups"]
    cityscapes_mapping = {int(k): v for k, v in config["cityscapes_mapping"].items()}
    id_to_label = {int(k): v for k, v in config["id_to_label"].items()}
    group_colors = np.array(config["group_colors"], dtype=np.uint8)
    group_names = config["group_names"]

    id_to_group = np.zeros(256, dtype=np.uint8)
    for id_, group_id in cityscapes_mapping.items():
        id_to_group[id_] = group_id

    num_classes = len(group_names)

    mapping_config = {
        "class_groups": class_groups,
        "cityscapes_mapping": cityscapes_mapping,
        "id_to_label": id_to_label,
        "group_colors": group_colors,
        "group_names": group_names,
        "id_to_group": id_to_group,
        "num_classes": num_classes
    }
    
    # Appliquer les couleurs WCAG si demandé
    if use_wcag_colors:
        mapping_config = update_colors_to_wcag(mapping_config)

    if verbose:
        print("== Groupes de classes Cityscapes ==")
        for group, classes in class_groups.items():
            print(f"  - {group} : {classes}")

        print("\n== Mapping ID Cityscapes → Groupe ==")
        for id_ in sorted(cityscapes_mapping.keys()):
            label = id_to_label.get(id_, "Label inconnu")
            group_id = cityscapes_mapping[id_]
            print(f"  ID {id_:2d} ('{label}') → groupe '{group_names[group_id]}' ({group_id})")

        print("\n== Liste des groupes et couleurs associées ==")
        for i, (name, color) in enumerate(zip(group_names, mapping_config['group_colors'])):
            print(f"  Groupe {i}: {name} - Couleur RGB : {color}")

        print(f"\nNombre total de groupes/classes : {num_classes}")

    return mapping_config


def map_mask_ids(mask, id_to_group_mapping):
    """
    Applique le mapping des IDs Cityscapes vers les groupes
    ADAPTÉ PYTORCH : Utilise torch au lieu de tf
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    if isinstance(id_to_group_mapping, np.ndarray):
        if len(id_to_group_mapping) != 256:
            raise ValueError(f"id_to_group_mapping doit avoir 256 éléments, reçu: {len(id_to_group_mapping)}")
        mapping_tensor = torch.tensor(id_to_group_mapping, dtype=torch.long)
    else:
        mapping_tensor = torch.tensor(list(id_to_group_mapping), dtype=torch.long)
    
    # S'assurer que les valeurs du masque sont dans la plage valide [0, 255]
    mask_clipped = torch.clamp(mask, 0, 255)
    mapped_mask = mapping_tensor[mask_clipped.long()]
    return mapped_mask
    
def colorize_mask(mask, group_colors):
    """
    Applique la colorisation du masque avec les couleurs définies
    INCHANGÉ - Compatible PyTorch (prend numpy en entrée)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_classes = len(group_colors)
    
    for group_id in range(num_classes):
        colored_mask[mask == group_id] = group_colors[group_id]
    
    return colored_mask


def get_preprocessing_fn(img_size=(512, 512), encoder=None):
    """
    Retourne une fonction qui prétraite à la fois les images et les masques
    pour l'inférence, en reproduisant exactement le pipeline CityscapesDataset.
    """
    
    if encoder is None:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Pipeline unifié (gérera image ET masque ensemble)
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    def preprocess(image_np, mask_np=None):
        if mask_np is not None:
            # Traitement simultané image + masque
            transformed = transform(image=image_np, mask=mask_np)
            return transformed['image'], transformed['mask'].long()
        else:
            # Image seule
            transformed = transform(image=image_np)
            return transformed['image'], None

    return preprocess

def apply_random_augmentation(image_np, prob=0.7):
    """
    Applique des transformations d'augmentation aléatoires à une image,
    similaires à celles utilisées pendant l'entraînement.
    
    Args:
        image_np (np.ndarray): Image numpy array (H, W, 3)
        prob (float): Probabilité d'appliquer chaque transformation
        
    Returns:
        tuple: (image_augmentée, liste_des_transformations_appliquées)
    """
    transforms_applied = []
    
    # Définir les transformations possibles
    possible_transforms = [
        {
            "name": "Rotation aléatoire",
            "transform": A.Rotate(limit=15, p=prob),
            "description": "Rotation de -15° à +15°"
        },
        {
            "name": "Flip horizontal", 
            "transform": A.HorizontalFlip(p=0.5),
            "description": "Miroir horizontal (50% chance)"
        },
        {
            "name": "Changement de luminosité",
            "transform": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=prob),
            "description": "Luminosité et contraste ±20%"
        },
        {
            "name": "Flou gaussien léger",
            "transform": A.GaussianBlur(blur_limit=(1, 3), p=prob*0.3),
            "description": "Flou gaussien subtil"
        },
        {
            "name": "Décalage couleur",
            "transform": A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=prob),
            "description": "Ajustement teinte/saturation/valeur"
        },
        {
            "name": "Zoom aléatoire",
            "transform": A.RandomScale(scale_limit=0.1, p=prob*0.5),
            "description": "Zoom de -10% à +10%"
        }
    ]
    
    # Sélectionner aléatoirement quelques transformations
    num_transforms = random.randint(1, min(4, len(possible_transforms)))
    selected_transforms = random.sample(possible_transforms, num_transforms)
    
    # Créer le pipeline d'augmentation
    augmentation_pipeline = []
    for transform_info in selected_transforms:
        augmentation_pipeline.append(transform_info["transform"])
        transforms_applied.append(f"{transform_info['name']}: {transform_info['description']}")
    
    # Appliquer les transformations
    compose = A.Compose(augmentation_pipeline)
    
    try:
        augmented = compose(image=image_np)
        augmented_image = augmented['image']
    except Exception as e:
        print(f"Erreur lors de l'augmentation: {e}")
        # En cas d'erreur, retourner l'image originale
        augmented_image = image_np.copy()
        transforms_applied = ["Erreur - image originale conservée"]
    
    return augmented_image, transforms_applied
    
# Version simplifiée de preprocess_image_and_mask utilisant la nouvelle get_preprocessing_fn
def preprocess_image_and_mask(image_path, mask_path=None, img_size=(512, 512),
                                    encoder_name=None, id_to_group_mapping=None):
    """
    Version simplifiée utilisant get_preprocessing_fn pour la cohérence avec CityscapesDataset
    """
    import numpy as np
    from PIL import Image
    import torch
    
    print(f"🔄 Préprocessing de l'image: {os.path.basename(image_path)}")
    
    # Chargement des données
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil)
    img_display = np.array(image_pil.resize(img_size, Image.BILINEAR))  # Pour affichage
    
    mask_np = None
    if mask_path and os.path.exists(mask_path):
        print(f"🔄 Préprocessing du masque: {os.path.basename(mask_path)}")
        mask_pil = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_pil)
        print(f"   Valeurs uniques avant mapping: {np.unique(mask_np)[:10]}...")
    
    # Utiliser notre preprocessing unifié
    preprocess_fn = get_preprocessing_fn(img_size=img_size, encoder=encoder_name)
    img_preprocessed, mask_preprocessed = preprocess_fn(image_np, mask_np)
    
    # Appliquer le mapping si fourni ET si on a un masque
    if mask_preprocessed is not None and id_to_group_mapping is not None:
        mask_preprocessed = map_mask_ids(mask_preprocessed, id_to_group_mapping)
        print(f"   Valeurs uniques après mapping: {torch.unique(mask_preprocessed)}")
    
    return img_preprocessed, img_display, mask_preprocessed

# Version mise à jour de run_inference_and_visualize
def run_inference_and_visualize(image_path, model, encoder_name, img_size, mapping_config, 
                               mask_path=None, save_dir=None, show_stats=True, show_plot=True):
    """
    Version simplifiée utilisant preprocess_image_and_mask
    """
    print(f"=== INFÉRENCE PYTORCH SUR {os.path.basename(image_path)} ===")

    # Utiliser la version simplifiée du preprocessing
    img_preprocessed, img_display, mask_gt = preprocess_image_and_mask(
        image_path=image_path,
        mask_path=mask_path,
        img_size=img_size or (512, 512),
        encoder_name=encoder_name,
        id_to_group_mapping=mapping_config.get('id_to_group')
    )

    print("🔮 Exécution de l'inférence PyTorch...")
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Ajouter dimension batch
        img_batch = img_preprocessed.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # ⏱️ MESURE DU TEMPS D'INFÉRENCE
        start_time = time.time()
        predictions = model(img_batch)  # [1, num_classes, H, W]
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        predicted_mask = torch.argmax(predictions[0], dim=0).cpu().numpy()  # [H, W]

    print(f"✅ Inférence terminée - Shape: {predicted_mask.shape}")
    print(f"⏱️ Temps d'inférence: {inference_time*1000:.1f} ms ({inference_time:.4f} s)")

    # [Le reste du code de visualisation reste identique]
    # Visualisation
    num_plots = 2 if mask_gt is None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    axes[0].imshow(img_display)
    axes[0].set_title("Image Originale")
    axes[0].axis('off')

    predicted_colored = colorize_mask(predicted_mask, mapping_config['group_colors'])
    axes[1].imshow(predicted_colored)
    axes[1].set_title("Masque Prédit")
    axes[1].axis('off')

    if mask_gt is not None:
        mask_gt_np = mask_gt.numpy() if isinstance(mask_gt, torch.Tensor) else mask_gt
        gt_colored = colorize_mask(mask_gt_np, mapping_config['group_colors'])
        axes[2].imshow(gt_colored)
        axes[2].set_title("Masque de Vérité")
        axes[2].axis('off')

        # Visualisation des erreurs
        diff_mask = (mask_gt_np != predicted_mask).astype(float)
        axes[3].imshow(img_display)
        axes[3].imshow(diff_mask, alpha=0.5, cmap='Reds')
        axes[3].set_title("Erreurs (en rouge)")
        axes[3].axis('off')

    colors_normalized = mapping_config['group_colors'] / 255.0
    custom_cmap = ListedColormap(colors_normalized)

    legend_elements = []
    for i, (name, color) in enumerate(zip(mapping_config['group_names'], colors_normalized)):
        legend_elements.append(mpatches.Patch(color=color, label=f'{i}: {name}'))

    fig.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(1.01, 0.5),
              fontsize=9,
              frameon=True,
              fancybox=True,
              shadow=True)

    plt.tight_layout()
    fig.subplots_adjust(right=0.95)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        fig_path = os.path.join(save_dir, f'{img_name}_inference.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"💾 Résultat sauvegardé: {fig_path}")


    # Statistiques
    if show_stats:
        print("\n📊 Statistiques du masque prédit:")
        unique_classes, counts = np.unique(predicted_mask, return_counts=True)
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(mapping_config['group_names']):
                percentage = (count / predicted_mask.size) * 100
                print(f"   {mapping_config['group_names'][class_id]}: {count} pixels ({percentage:.1f}%)")

        if mask_gt is not None:
            mask_gt_np = mask_gt.numpy() if isinstance(mask_gt, torch.Tensor) else mask_gt
            print("\n📊 Statistiques du masque de vérité:")
            unique_classes, counts = np.unique(mask_gt_np, return_counts=True)
            for class_id, count in zip(unique_classes, counts):
                if class_id < len(mapping_config['group_names']):
                    percentage = (count / mask_gt_np.size) * 100
                    print(f"   {mapping_config['group_names'][class_id]}: {count} pixels ({percentage:.1f}%)")

    if show_plot:
        plt.show()
        return predicted_mask, inference_time
    else:
        return predicted_mask, inference_time, fig


def load_model(run_id=None, experiment_name="OC Projet 9", 
               metric="test_mean_iou", top_n=1):
    """
    Charge un modèle PyTorch depuis MLflow avec plusieurs méthodes de sélection
    
    Args:
        run_id (str, optional): Run ID spécifique (prioritaire si fourni)
        experiment_name (str): Nom de l'expérience MLflow
        metric (str): Métrique pour classer les runs
        top_n (int): Position dans le classement (1=meilleur, 2=deuxième, etc.)
        
    Returns:
        tuple: (model, source_run_id, encoder_name, script_img_size)
    """
    
    print(f"\n🔍 Chargement d'un modèle PyTorch depuis MLflow...")
    
    client = mlflow.tracking.MlflowClient()
    
    # 1. Sélection du run selon les paramètres
    if run_id:
        print(f"🎯 Mode run_id spécifique: {run_id}")
        try:
            run = mlflow.get_run(run_id)
            source_run_id = run_id
            print(f"✅ Run trouvé: {run.data.tags.get('mlflow.runName', 'Unknown')}")
        except Exception as e:
            raise RuntimeError(f"❌ Run {run_id} introuvable: {e}")
    
    else:
        print(f"🏆 Mode classement: top_{top_n} de '{metric}' dans '{experiment_name}'")
        try:
            # Récupérer l'expérience
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise RuntimeError(f"❌ Expérience '{experiment_name}' non trouvée")
            
            # Rechercher les runs avec la métrique, triés par ordre décroissant
            filter_string = f"status = 'FINISHED' and metrics.{metric} >= 0"
            order_by = f"metrics.{metric} DESC"
            
            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=[order_by],
                max_results=top_n
            )
            
            if runs_df.empty:
                raise RuntimeError(f"❌ Aucun run avec '{metric}' trouvé dans '{experiment_name}'")
            
            if len(runs_df) < top_n:
                raise RuntimeError(f"❌ Seulement {len(runs_df)} runs trouvés, impossible d'obtenir le top_{top_n}")
            
            # Prendre le run à la position top_n
            selected_run = runs_df.iloc[top_n - 1]
            source_run_id = selected_run['run_id']
            metric_value = selected_run[f'metrics.{metric}']
            run_name = selected_run.get('tags.mlflow.runName', 'Unknown')
            
            print(f"✅ Run sélectionné (position #{top_n}):")
            print(f"   • Nom: {run_name}")
            print(f"   • Run ID: {source_run_id}")
            print(f"   • {metric}: {metric_value:.4f}")
            
            # Afficher le contexte
            print(f"📊 Contexte du classement:")
            for i, (_, row) in enumerate(runs_df.head(min(5, len(runs_df))).iterrows()):
                indicator = "👑" if i == (top_n - 1) else f"{i+1}."
                name = row.get('tags.mlflow.runName', 'Unknown')[:30]
                value = row[f'metrics.{metric}']
                print(f"   {indicator} {name}: {value:.4f}")
            
            run = mlflow.get_run(source_run_id)
            
        except Exception as e:
            raise RuntimeError(f"❌ Erreur lors de la sélection: {e}")
    
    # 2-5. [Reste du code identique - extraction params, détection variante, chargement...]
    try:
        params = run.data.params
        
        encoder_name = params.get("encoder_name", None)
        architecture = params.get("architecture", "FPN")
        num_classes = int(params.get("num_classes", 8))
        script_img_size_raw = params.get("script_img_size", None)
        run_name = run.data.tags.get('mlflow.runName', 'Unknown')
        
        # Parser script_img_size
        script_img_size = None
        if script_img_size_raw:
            try:
                script_img_size = ast.literal_eval(script_img_size_raw)
                if isinstance(script_img_size, (tuple, list)) and len(script_img_size) == 2:
                    script_img_size = tuple(script_img_size)
            except:
                script_img_size = None
        
        print(f"\n📋 Paramètres du modèle:")
        print(f"🏗️ Architecture: {architecture}")
        print(f"🔧 Encodeur: {encoder_name}")
        print(f"📊 Nombre de classes: {num_classes}")
        print(f"🖼️ Script img size: {script_img_size}")
        
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de la récupération des paramètres: {e}")
    
    # Détecter la variante SegFormer
    def detect_segformer_variant_from_name(run_name, params):
        segformer_pattern = r'SegFormer[_-]?b(\d+)'
        match = re.search(segformer_pattern, run_name, re.IGNORECASE)
        if match:
            variant_num = match.group(1)
            detected = f"mit_b{variant_num}"
            print(f"🔍 Variante détectée: 'SegFormer_b{variant_num}' → {detected}")
            return detected
        
        if encoder_name and encoder_name.startswith("mit_b"):
            print(f"🔍 Variante depuis paramètres: {encoder_name}")
            return encoder_name
            
        b_pattern = r'\bb(\d+)\b'
        match = re.search(b_pattern, run_name, re.IGNORECASE)
        if match:
            variant_num = match.group(1)
            detected = f"mit_b{variant_num}"
            print(f"🔍 Pattern détecté: 'b{variant_num}' → {detected}")
            return detected
        
        print(f"🔍 Aucune variante détectée, utilisation de mit_b0")
        return "mit_b0"
    
    detected_variant = detect_segformer_variant_from_name(run_name, params)
    
    # Télécharger et charger le modèle
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            artifacts = client.list_artifacts(source_run_id)
            model_artifact_candidates = [a.path for a in artifacts if a.path.endswith('.pth')]
            
            if not model_artifact_candidates:
                raise RuntimeError("❌ Aucun fichier .pth trouvé")
            
            model_artifact = model_artifact_candidates[0]
            print(f"⬇️ Téléchargement de {model_artifact}...")
            
            local_path = mlflow.artifacts.download_artifacts(
                run_id=source_run_id,
                artifact_path=model_artifact,
                dst_path=tmpdir
            )
            
            checkpoint = torch.load(local_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
        except Exception as e:
            raise RuntimeError(f"❌ Erreur lors du téléchargement: {e}")
    
    # Créer le modèle
    try:
        is_transformer = any('patch_embed' in key for key in state_dict.keys())
        
        if is_transformer:
            print(f"🔧 Création SegFormer ({detected_variant})")
            model = smp.Segformer(encoder_name=detected_variant, classes=num_classes)
            encoder_name = detected_variant
        else:
            print("🔧 Création modèle CNN")
            if encoder_name is None or encoder_name == "None":
                encoder_name = "efficientnet-b0"
            
            if architecture.lower() == "fpn":
                model = smp.FPN(encoder_name=encoder_name, classes=num_classes)
            elif architecture.lower() == "unet":
                model = smp.Unet(encoder_name=encoder_name, classes=num_classes)
            elif architecture.lower() == "deeplabv3plus":
                model = smp.DeepLabV3Plus(encoder_name=encoder_name, classes=num_classes)
            else:
                raise ValueError(f"Architecture {architecture} non supportée")
        
        print("🔄 Chargement des poids...")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("✅ Modèle chargé avec succès!")
        
    except Exception as e:
        try:
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("✅ Chargement partiel réussi!")
        except Exception as e2:
            raise RuntimeError(f"❌ Erreur lors du chargement: {e2}")
    
    print(f"🎉 Modèle chargé depuis le run {source_run_id}")
    return model, source_run_id, encoder_name, script_img_size


def load_best_model_from_registry(model_name):
    """
    Charge le meilleur modèle depuis le MLflow Model Registry
    (Version refactorisée - wrapper autour de load_model)
    
    Args:
        model_name (str): Nom du modèle dans le Registry
        
    Returns:
        tuple: (model, source_run_id, encoder_name, script_img_size)
    """
    print(f"🏛️ Récupération du modèle '{model_name}' depuis le Registry...")
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Récupérer la version en Production
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not prod_versions:
            raise RuntimeError(f"❌ Aucune version en Production trouvée pour {model_name}")
        
        latest_prod = max(prod_versions, key=lambda v: int(v.version))
        print(f"📦 Version Registry: {latest_prod.version} (Stage: {latest_prod.current_stage})")
        
        # Récupérer le source_run_id depuis le tag du Registry
        source_run_id = latest_prod.tags.get("source_run_id")
        if not source_run_id:
            raise RuntimeError(f"❌ Tag 'source_run_id' absent pour la version {latest_prod.version}")
        
        print(f"🔗 Source run détecté: {source_run_id}")
        
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors de l'accès au Registry: {e}")
    
    # Déléguer le chargement à load_model avec le run_id trouvé
    print("➡️ Délégation vers load_model...")
    return load_model(run_id=source_run_id)


# Fonctions de convenance
def load_best_model(experiment_name="OC Projet 9", metric="test_mean_iou"):
    """Charge le meilleur modèle selon une métrique"""
    return load_model(experiment_name=experiment_name, metric=metric, top_n=1)

def load_second_best_model(experiment_name="OC Projet 9", metric="test_mean_iou"):
    """Charge le deuxième meilleur modèle selon une métrique"""
    return load_model(experiment_name=experiment_name, metric=metric, top_n=2)

def load_model_by_run_id(run_id):
    """Charge un modèle par run_id spécifique"""
    return load_model(run_id=run_id)