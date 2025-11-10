import os
import json
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import mlflow


# Au début de utils.py, après les imports
import segmentation_models as sm
import tensorflow.keras.backend as K

# Configuration du backend segmentation_models (comme dans le notebook)
sm.set_framework('tf.keras')
sm.backend = K


def load_cityscapes_config(config_path="../cityscapes_config.json", verbose=True):
    """
    Charge la configuration Cityscapes depuis un fichier JSON.
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
        for i, (name, color) in enumerate(zip(group_names, group_colors)):
            print(f"  Groupe {i}: {name} - Couleur RGB : {color}")

        print(f"\nNombre total de groupes/classes : {num_classes}")

    return {
        "class_groups": class_groups,
        "cityscapes_mapping": cityscapes_mapping,
        "id_to_label": id_to_label,
        "group_colors": group_colors,
        "group_names": group_names,
        "id_to_group": id_to_group,
        "num_classes": num_classes
    }


def get_preprocessing_fn(encoder_name=None):
    """
    Retourne la fonction de prétraitement adaptée au backbone utilisé.
    Si encoder_name est None (modèle vanilla), on normalise à la main.
    """
    print("ℹ️ Modèle vanilla - pas de preprocessing")
    return lambda x: x / 255.0
    # Peut être rajouté dans le cas d'un entraînement plus poussé avec un traitement particulier pour chaque feature encoder
    # if encoder_name is None:
    #     print("ℹ️ Modèle vanilla - pas de preprocessing")
    #     return lambda x: x / 255.0
    # else:
    #     print(f"ℹ️ Préprocessing du backbone {encoder_name}")
    #     return sm.get_preprocessing(encoder_name)


def load_best_model_from_registry(model_name):
    """
    Charge le meilleur modèle depuis le MLflow Model Registry
    """
    client = mlflow.tracking.MlflowClient()

    # Récupérer la dernière version en Production (votre logique)
    versions = client.search_model_versions(f"name='{model_name}'")
    
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if not prod_versions:
        raise RuntimeError(f"Aucune version en Production trouvée pour {model_name}")
    latest_prod = max(prod_versions, key=lambda v: int(v.version))

    # Récupérer le tag source_run_id
    source_run_id = latest_prod.tags.get("source_run_id")
    if not source_run_id:
        raise RuntimeError(f"Tag 'source_run_id' absent pour la version {latest_prod.version}")

    print(f"📦 Téléchargement du modèle {model_name} (run: {source_run_id})")
    
    # Télécharger le fichier best_model.keras (VOTRE approche)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=source_run_id,
            artifact_path="best_model.keras",
            dst_path=tmpdir
        )
        
        # ✅ SOLUTION : Charger SANS compiler d'abord
        print("🔄 Chargement sans compilation...")
        model = tf.keras.models.load_model(local_path, compile=False)
        
        # Puis recompiler avec la bonne métrique
        print("🔄 Recompilation avec MeanIoU...")
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=[tf.keras.metrics.MeanIoU(num_classes=8)]
        )
        print("✅ Modèle chargé et recompilé !")

    # Récupérer les paramètres (votre logique)
    run = mlflow.get_run(source_run_id)
    encoder_name = run.data.params.get("encoder", None)

    script_img_size_str = run.data.params.get("script_img_size", None)
    script_img_size = None
    if script_img_size_str:
        try:
            script_img_size = tuple(int(x.replace(' ','').replace('(','').replace(')','').strip()) for x in script_img_size_str.split(","))
        except Exception as e:
            print(f"⚠️ Impossible de parser 'script_img_size': {e}")

    return model, source_run_id, encoder_name, script_img_size


def map_mask_ids(mask, id_to_group_mapping):
    """
    Applique le mapping des IDs Cityscapes vers les groupes
    """
    if isinstance(id_to_group_mapping, np.ndarray):
        # Vérifier que le mapping a bien 256 éléments
        if len(id_to_group_mapping) != 256:
            raise ValueError(f"id_to_group_mapping doit avoir 256 éléments, reçu: {len(id_to_group_mapping)}")
        mapping_tensor = tf.constant(id_to_group_mapping, dtype=tf.int32)
    else:
        mapping_tensor = tf.constant(list(id_to_group_mapping), dtype=tf.int32)
    
    # S'assurer que les valeurs du masque sont dans la plage valide [0, 255]
    mask_clipped = tf.clip_by_value(mask, 0, 255)
    mapped_mask = tf.gather(mapping_tensor, mask_clipped)
    return mapped_mask


def preprocess_image_and_mask(image_path, mask_path=None, img_size=(224, 224), 
                             encoder_name=None, id_to_group_mapping=None):
    """
    Préprocesse une image (et optionnellement un masque) avec le même pipeline que l'entraînement
    
    Args:
        image_path (str): Chemin vers l'image
        mask_path (str, optional): Chemin vers le masque de vérité
        img_size (tuple): Taille de redimensionnement (H, W)
        encoder_name (str): Nom du backbone pour le preprocessing
        id_to_group_mapping: Mapping des IDs Cityscapes vers les groupes
    
    Returns:
        tuple: (image_preprocessed, mask_preprocessed) ou (image_preprocessed, None)
    """
    print(f"🔄 Préprocessing de l'image: {os.path.basename(image_path)}")
    
    # Preprocessing de l'image (même logique que l'entraînement)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)  # Support PNG et JPG automatique
    img_original = tf.image.resize(img, img_size, method='bilinear')  # Pour affichage
    img = tf.cast(img_original, tf.float32)
    
    # Appliquer le preprocessing du backbone
    preprocessing_fn = get_preprocessing_fn(encoder_name)
    img_preprocessed = preprocessing_fn(img)
    
    # Preprocessing du masque si fourni
    mask_preprocessed = None
    if mask_path and os.path.exists(mask_path):
        print(f"🔄 Préprocessing du masque: {os.path.basename(mask_path)}")
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1)  # Support PNG et JPG pour les masques aussi
        mask = tf.image.resize(mask, img_size, method='nearest')
        mask = tf.cast(mask, tf.int32)
        
        print(f"   Valeurs uniques avant mapping: {tf.unique(tf.reshape(mask, [-1]))[0][:10].numpy()}...")  # Debug
        
        # Appliquer le mapping si fourni
        if id_to_group_mapping is not None:
            mask = map_mask_ids(mask, id_to_group_mapping)
            print(f"   Valeurs uniques après mapping: {tf.unique(tf.reshape(mask, [-1]))[0].numpy()}")  # Debug
        
        mask_preprocessed = tf.squeeze(mask, axis=-1)
    
    return img_preprocessed, img_original, mask_preprocessed


def colorize_mask(mask, group_colors):
    """
    Applique la colorisation du masque avec les couleurs définies
    
    Args:
        mask (np.ndarray): Masque avec les IDs de classes
        group_colors (np.ndarray): Couleurs RGB pour chaque classe
    
    Returns:
        np.ndarray: Masque colorisé RGB
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_classes = len(group_colors)
    
    for group_id in range(num_classes):
        colored_mask[mask == group_id] = group_colors[group_id]
    
    return colored_mask


def run_inference_and_visualize(image_path, model, encoder_name, img_size, mapping_config, 
                               mask_path=None, save_dir=None, show_stats=True, show_plot=True):
    print(f"=== INFÉRENCE SUR {os.path.basename(image_path)} ===")
    
    img_preprocessed, img_display, mask_gt = preprocess_image_and_mask(
        image_path=image_path,
        mask_path=mask_path,
        img_size=img_size or (224, 224),
        encoder_name=encoder_name,
        id_to_group_mapping=mapping_config.get('id_to_group')
    )
    
    print("🔮 Exécution de l'inférence...")
    img_batch = tf.expand_dims(img_preprocessed, axis=0)
    predictions = model.predict(img_batch, verbose=0)
    predicted_mask = tf.argmax(predictions[0], axis=-1).numpy()
    print(f"✅ Inférence terminée - Shape: {predicted_mask.shape}")
    
    # Nombre de plots : 2 si pas de masque, sinon 4 (image, prédiction, vérité, erreurs)
    num_plots = 2 if mask_gt is None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    axes[0].imshow(img_display.numpy().astype(np.uint8))
    axes[0].set_title("Image Originale")
    axes[0].axis('off')
    
    predicted_colored = colorize_mask(predicted_mask, mapping_config['group_colors'])
    axes[1].imshow(predicted_colored)
    axes[1].set_title("Masque Prédit")
    axes[1].axis('off')
    
    if mask_gt is not None:
        gt_colored = colorize_mask(mask_gt.numpy(), mapping_config['group_colors'])
        axes[2].imshow(gt_colored)
        axes[2].set_title("Masque de Vérité")
        axes[2].axis('off')
        
        # Visualisation des erreurs en rouge translucide
        diff_mask = (mask_gt.numpy() != predicted_mask).astype(float)
        axes[3].imshow(img_display.numpy().astype(np.uint8))
        axes[3].imshow(diff_mask, alpha=0.5, cmap='Reds')
        axes[3].set_title("Erreurs (en rouge)")
        axes[3].axis('off')
    
    # Légende colorée
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
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
    
    
    if show_stats:
        print("\n📊 Statistiques du masque prédit:")
        unique_classes, counts = np.unique(predicted_mask, return_counts=True)
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(mapping_config['group_names']):
                percentage = (count / predicted_mask.size) * 100
                print(f"   {mapping_config['group_names'][class_id]}: {count} pixels ({percentage:.1f}%)")
        
        if mask_gt is not None:
            print("\n📊 Statistiques du masque de vérité:")
            unique_classes, counts = np.unique(mask_gt.numpy(), return_counts=True)
            for class_id, count in zip(unique_classes, counts):
                if class_id < len(mapping_config['group_names']):
                    percentage = (count / mask_gt.numpy().size) * 100
                    print(f"   {mapping_config['group_names'][class_id]}: {count} pixels ({percentage:.1f}%)")
                    
    if show_plot:
        plt.show()
        return predicted_mask
    else:
        return predicted_mask, fig