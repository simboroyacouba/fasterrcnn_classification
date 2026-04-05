"""
Entraînement Faster R-CNN pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: Chargées depuis classes.yaml
Configuration: Chargée depuis .env

NOUVEAUTÉS :
  - Augmentation différentielle (classes rares sur-représentées)
  - WeightedRandomSampler : batiment_peint ×3, batiment_non_enduit ×2, batiment_enduit ×1.5
  - Augmentations géométriques : flip H/V, rotation 90°, RandomResizedCrop
  - Augmentations photométriques : ColorJitter, GaussianBlur, bruit additif, gris aléatoire
  - CutOut ciblé sur les images contenant des classes rares
"""

import os
import json
import yaml
import shutil
import random
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pycocotools.coco import COCO
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CHARGEMENT DES CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    print(f"📋 Classes chargées depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "images_dir":        os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
    "annotations_file":  os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "output_dir":        os.getenv("OUTPUT_DIR", "./output"),
    "classes_file":      os.getenv("CLASSES_FILE", "classes.yaml"),
    "classes":           None,

    # Hyperparamètres
    "num_epochs":        int(os.getenv("NUM_EPOCHS", "25")),
    "batch_size":        int(os.getenv("BATCH_SIZE", "2")),
    "learning_rate":     float(os.getenv("LEARNING_RATE", "0.005")),
    "momentum":          float(os.getenv("MOMENTUM", "0.9")),
    "weight_decay":      float(os.getenv("WEIGHT_DECAY", "0.0005")),
    "image_size":        int(os.getenv("IMAGE_SIZE", "640")),
    "train_split":       float(os.getenv("TRAIN_SPLIT", "0.70")),
    "val_split":         float(os.getenv("VAL_SPLIT", "0.20")),
    "test_split":        float(os.getenv("TEST_SPLIT", "0.10")),
    "save_every":        int(os.getenv("SAVE_EVERY", "5")),
    "score_threshold":   float(os.getenv("SCORE_THRESHOLD", "0.5")),
    "pretrained":        os.getenv("PRETRAINED", "true").lower() == "true",

    # Poids de sur-échantillonnage par classe (1-based index)
    # Augmenté si l'image contient au moins une instance de cette classe.
    # Quand une image contient plusieurs classes rares, on prend le max.
    "rare_class_weights": {
        "batiment_peint":      3.0,
        "batiment_non_enduit": 2.0,
        "batiment_enduit":     1.5,
        # panneau_solaire reste à 1.0 (classe dominante)
    },
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def stratified_split(coco, train_split, val_split, test_split, seed=42):
    np.random.seed(seed)
    all_image_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_image_ids)

    n_total = len(all_image_ids)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val

    if n_test < 1 and n_total > 2:
        n_test  = max(1, int(n_total * 0.10))
        n_train = n_total - n_val - n_test

    print(f"\n   📊 Split des IMAGES (total: {n_total}):")
    print(f"      Train: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"      Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"      Test:  {n_test}  ({n_test/n_total*100:.1f}%)")

    train_ids = all_image_ids[:n_train]
    val_ids   = all_image_ids[n_train:n_train + n_val]
    test_ids  = all_image_ids[n_train + n_val:]

    stats = {'train': {}, 'val': {}, 'test': {}}
    for cat_id in coco.getCatIds():
        for split in stats:
            stats[split][cat_id] = 0

    for img_id in train_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['train'][ann['category_id']] += 1
    for img_id in val_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['val'][ann['category_id']] += 1
    for img_id in test_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['test'][ann['category_id']] += 1

    return train_ids, val_ids, test_ids, stats


def print_split_stats(coco, stats):
    print("\n   📊 Distribution des classes (split 70/20/10):")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = "⚠️" if val == 0 or test == 0 else "✅"
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8} {ok}")
    print(f"   {'-'*70}")


# =============================================================================
# AUGMENTATIONS
# =============================================================================

def apply_cutout(image_tensor, boxes, n_holes=1, max_size=32, min_size=16):
    """
    CutOut : masque aléatoire sur l'image (ne touche pas les boxes GT).
    Fonctionne sur tensor [C, H, W] dans [0, 1].
    """
    c, h, w = image_tensor.shape
    result = image_tensor.clone()
    for _ in range(n_holes):
        hole_h = random.randint(min_size, max_size)
        hole_w = random.randint(min_size, max_size)
        y = random.randint(0, h - hole_h)
        x = random.randint(0, w - hole_w)
        result[:, y:y + hole_h, x:x + hole_w] = 0.0
    return result


def augment_image_and_boxes(image: Image.Image, boxes: list, labels: list,
                             image_size: int, is_rare: bool):
    """
    Applique des augmentations à une image PIL et met à jour les boxes.

    Paramètres
    ----------
    image      : PIL.Image RGB, déjà resizée à (image_size × image_size)
    boxes      : list de [x1, y1, x2, y2] en pixels (image_size coords)
    labels     : list de int
    image_size : taille de l'image (carré)
    is_rare    : True si l'image contient au moins une classe rare

    Retourne
    --------
    image_tensor : torch.Tensor [C, H, W]
    boxes_out    : list de [x1, y1, x2, y2]
    labels_out   : list de int (identiques, pas de filtrage)
    """
    W = H = image_size

    # --- Flip horizontal ---
    if random.random() < 0.5:
        image = TF.hflip(image)
        boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]

    # --- Flip vertical ---
    if random.random() < 0.5:
        image = TF.vflip(image)
        boxes = [[x1, H - y2, x2, H - y1] for x1, y1, x2, y2 in boxes]

    # --- Rotation 90° (multiple de 90) ---
    if random.random() < 0.3:
        k = random.choice([1, 2, 3])   # 90, 180, 270 degrés
        image = image.rotate(-90 * k, expand=False)
        new_boxes = []
        for x1, y1, x2, y2 in boxes:
            for _ in range(k):
                x1, y1, x2, y2 = y1, W - x2, y2, W - x1
            new_boxes.append([
                max(0, min(x1, x2)), max(0, min(y1, y2)),
                min(W, max(x1, x2)), min(H, max(y1, y2))
            ])
        boxes = new_boxes

    # --- RandomResizedCrop : recadrage aléatoire + resize ---
    # Utilisé seulement si on a des boxes (sinon risque de crop vide)
    if random.random() < 0.4 and len(boxes) > 0:
        scale = random.uniform(0.65, 1.0)
        crop_size = int(W * scale)
        max_x = W - crop_size
        max_y = H - crop_size
        cx = random.randint(0, max_x)
        cy = random.randint(0, max_y)

        image = image.crop((cx, cy, cx + crop_size, cy + crop_size))
        image = image.resize((W, H), Image.BILINEAR)

        ratio = W / crop_size
        new_boxes = []
        new_labels = []
        for (x1, y1, x2, y2), lbl in zip(boxes, labels):
            # Recalculer dans le crop
            nx1 = max(0, (x1 - cx) * ratio)
            ny1 = max(0, (y1 - cy) * ratio)
            nx2 = min(W, (x2 - cx) * ratio)
            ny2 = min(H, (y2 - cy) * ratio)
            # Garder si la box résultante est assez grande (>= 8px de côté)
            if nx2 - nx1 >= 8 and ny2 - ny1 >= 8:
                new_boxes.append([nx1, ny1, nx2, ny2])
                new_labels.append(lbl)
        # Si toutes les boxes ont disparu du crop, on abandonne ce crop
        if len(new_boxes) > 0:
            boxes  = new_boxes
            labels = new_labels
        else:
            # Revenir à l'image resizée sans crop
            image = image.resize((W, H), Image.BILINEAR)

    # --- Augmentations photométriques (sur PIL avant conversion tensor) ---

    # ColorJitter : luminosité, contraste, saturation, teinte
    if random.random() < 0.7:
        brightness = random.uniform(0.6, 1.4)
        contrast   = random.uniform(0.7, 1.3)
        saturation = random.uniform(0.7, 1.3)
        hue        = random.uniform(-0.05, 0.05)
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)

    # Gris aléatoire (simule images IR ou overcast)
    if random.random() < 0.1:
        image = TF.to_grayscale(image, num_output_channels=3)

    # Conversion en tensor
    tensor = TF.to_tensor(image)   # [C, H, W], float32 dans [0, 1]

    # Flou gaussien (après conversion pour garder la précision)
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        tensor = TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=sigma)

    # Bruit gaussien additif
    if random.random() < 0.2:
        noise = torch.randn_like(tensor) * random.uniform(0.01, 0.04)
        tensor = (tensor + noise).clamp(0.0, 1.0)

    # CutOut — priorité aux images de classes rares
    cutout_prob = 0.5 if is_rare else 0.15
    if random.random() < cutout_prob and len(boxes) > 0:
        tensor = apply_cutout(tensor, boxes, n_holes=random.randint(1, 2),
                              max_size=32, min_size=16)

    return tensor, boxes, labels


# =============================================================================
# CALCUL DES POIDS DE SAMPLER
# =============================================================================

def compute_sample_weights(coco, image_ids, cat_mapping, rare_class_weights, classes):
    """
    Calcule un poids par image pour WeightedRandomSampler.
    Le poids d'une image est le maximum des poids de ses classes présentes.
    """
    # Construire un mapping name -> weight
    # classes est ['__background__', 'cls1', ...]
    name_to_weight = {name: rare_class_weights.get(name, 1.0) for name in classes}

    # Mapping cat_id COCO -> nom de classe
    cat_id_to_name = {cat_id: coco.cats[cat_id]['name'] for cat_id in coco.getCatIds()}

    weights = []
    rare_count = 0

    for img_id in image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_weight = 1.0
        for ann in anns:
            cat_name = cat_id_to_name.get(ann['category_id'], '')
            w = name_to_weight.get(cat_name, 1.0)
            if w > img_weight:
                img_weight = w
        if img_weight > 1.0:
            rare_count += 1
        weights.append(img_weight)

    print(f"   🎲 Sampler: {rare_count}/{len(image_ids)} images avec classes rares sur-pondérées")
    for name, w in rare_class_weights.items():
        print(f"      {name:<30} ×{w:.1f}")

    return weights


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class CocoDetectionDataset(Dataset):
    """Dataset COCO pour Faster R-CNN avec augmentation différentielle."""

    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping,
                 image_size=640, augment=False, rare_class_names=None):
        self.images_dir       = images_dir
        self.coco             = COCO(annotations_file)
        self.image_ids        = image_ids
        self.cat_mapping      = cat_mapping
        self.image_size       = image_size
        self.augment          = augment
        self.rare_class_names = set(rare_class_names or [])

        # Pré-calculer quelles images contiennent des classes rares
        self._rare_flags = {}
        if augment and self.rare_class_names:
            cat_id_to_name = {cat_id: self.coco.cats[cat_id]['name']
                              for cat_id in self.coco.getCatIds()}
            for img_id in image_ids:
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
                is_rare = any(cat_id_to_name.get(a['category_id'], '') in self.rare_class_names
                              for a in anns)
                self._rare_flags[img_id] = is_rare

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Redimensionner à image_size × image_size
        image    = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        scale_x  = self.image_size / orig_w
        scale_y  = self.image_size / orig_h

        # Charger les annotations et adapter les boxes
        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes  = []
        labels = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            class_id = self.cat_mapping.get(ann['category_id'])
            if class_id is None:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x1 = max(0, x * scale_x)
            y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        # --- Augmentation ---
        if self.augment and len(boxes) > 0:
            is_rare = self._rare_flags.get(img_id, False)
            image_tensor, boxes, labels = augment_image_and_boxes(
                image, boxes, labels, self.image_size, is_rare
            )
        else:
            image_tensor = TF.to_tensor(image)

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =============================================================================
# MODÈLE
# =============================================================================

def build_model(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# =============================================================================
# MÉTRIQUES
# =============================================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0


def compute_map(predictions, ground_truths, iou_threshold=0.5):
    all_classes = set()
    for gt in ground_truths:
        all_classes.update(gt['labels'].tolist())

    aps = []
    for cls in all_classes:
        tps, fps, scores_list = [], [], []
        n_gt = sum((gt['labels'] == cls).sum().item() for gt in ground_truths)
        if n_gt == 0:
            continue

        for pred, gt in zip(predictions, ground_truths):
            mask_p  = pred['labels'] == cls
            mask_g  = gt['labels']   == cls
            p_boxes  = pred['boxes'][mask_p].cpu().numpy()
            p_scores = pred['scores'][mask_p].cpu().numpy()
            g_boxes  = gt['boxes'][mask_g].cpu().numpy()

            matched = set()
            for i in np.argsort(-p_scores):
                scores_list.append(p_scores[i])
                if len(g_boxes) == 0:
                    tps.append(0); fps.append(1)
                    continue
                ious   = [calculate_iou(p_boxes[i], g) for g in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j)
                    tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)

        if not scores_list:
            continue
        order     = np.argsort(-np.array(scores_list))
        tp_cum    = np.cumsum(np.array(tps)[order])
        fp_cum    = np.cumsum(np.array(fps)[order])
        precision = tp_cum / (tp_cum + fp_cum + 1e-10)
        recall    = tp_cum / (n_gt + 1e-10)

        ap = 0
        for r_thresh in np.arange(0, 1.1, 0.1):
            mask = recall >= r_thresh
            ap  += np.max(precision[mask]) if mask.any() else 0
        aps.append(ap / 11)

    return float(np.mean(aps)) if aps else 0.0


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, device, epoch, num_epochs):
    model.train()
    total_loss  = 0
    losses_dict = {'loss_classifier': 0, 'loss_box_reg': 0,
                   'loss_objectness': 0, 'loss_rpn_box_reg': 0}
    num_batches = 0

    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if all(len(t['boxes']) == 0 for t in targets):
            continue

        try:
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses.item()
            for k in losses_dict:
                losses_dict[k] += loss_dict.get(k, torch.tensor(0)).item()
            num_batches += 1

        except Exception as e:
            print(f"   ⚠️ Erreur batch: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    for k in losses_dict:
        losses_dict[k] /= max(num_batches, 1)

    return avg_loss, losses_dict


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, score_threshold=0.5):
    model.eval()
    all_preds = []
    all_gts   = []

    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            keep = output['scores'] >= score_threshold
            all_preds.append({
                'boxes':  output['boxes'][keep].cpu(),
                'labels': output['labels'][keep].cpu(),
                'scores': output['scores'][keep].cpu(),
            })
            all_gts.append({
                'boxes':  target['boxes'],
                'labels': target['labels'],
            })

    return compute_map(all_preds, all_gts, iou_threshold=0.5)


# =============================================================================
# MAIN
# =============================================================================

def train_fasterrcnn():
    CONFIG["classes"] = load_classes(CONFIG["classes_file"])
    num_classes       = len(CONFIG["classes"])

    print("=" * 70)
    print("   Faster R-CNN (ResNet-50 FPN) - Détection des Toitures")
    print("   Augmentation différentielle activée")
    print("=" * 70)
    print(f"\n📋 CONFIG (.env)")
    print(f"   Images:      {CONFIG['images_dir']}")
    print(f"   Annotations: {CONFIG['annotations_file']}")
    print(f"   Classes:     {num_classes} (avec __background__)")
    print(f"   Epochs:      {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = os.path.join(CONFIG["output_dir"], f"fasterrcnn_{timestamp}")
    os.makedirs(train_dir, exist_ok=True)
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------
    coco = COCO(CONFIG["annotations_file"])
    cat_ids     = coco.getCatIds()
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, CONFIG["train_split"], CONFIG["val_split"], CONFIG["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    # Sauvegarder les IDs du test set
    test_info = {
        'test_image_ids':  test_ids,
        'cat_mapping':     {str(k): v for k, v in cat_mapping.items()},
        'images_dir':      CONFIG["images_dir"],
        'annotations_file':CONFIG["annotations_file"],
        'num_test_images': len(test_ids),
    }
    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    # -------------------------------------------------------------------------
    # Poids du sampler
    # -------------------------------------------------------------------------
    rare_class_names   = list(CONFIG["rare_class_weights"].keys())
    sample_weights_arr = compute_sample_weights(
        coco, train_ids, cat_mapping, CONFIG["rare_class_weights"], CONFIG["classes"]
    )
    sampler = WeightedRandomSampler(
        weights     = sample_weights_arr,
        num_samples = len(sample_weights_arr),
        replacement = True,
    )

    # -------------------------------------------------------------------------
    # Datasets & DataLoaders
    # -------------------------------------------------------------------------
    train_dataset = CocoDetectionDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        train_ids, cat_mapping, CONFIG["image_size"],
        augment=True, rare_class_names=rare_class_names,
    )
    val_dataset = CocoDetectionDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        val_ids, cat_mapping, CONFIG["image_size"],
        augment=False,
    )

    # shuffle=False car on utilise un sampler custom
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        sampler=sampler, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"\n   Train: {len(train_dataset)} images | Val: {len(val_dataset)} images | Test: {len(test_ids)} images")

    # -------------------------------------------------------------------------
    # Modèle & Optimiseur
    # -------------------------------------------------------------------------
    print(f"\n🧠 Chargement Faster R-CNN ResNet-50 FPN (pretrained={CONFIG['pretrained']})...")
    model     = build_model(num_classes, pretrained=CONFIG["pretrained"])
    model.to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=CONFIG["learning_rate"],
                                momentum=CONFIG["momentum"],
                                weight_decay=CONFIG["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # -------------------------------------------------------------------------
    # Boucle d'entraînement
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"   🚀 ENTRAÎNEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history = {
        'train_loss': [], 'val_map50': [],
        'loss_classifier': [], 'loss_box_reg': [],
        'loss_objectness': [], 'loss_rpn_box_reg': [],
        'lr': [],
    }

    best_map50 = 0.0
    start_time = time.time()

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\n📅 Epoch [{epoch}/{CONFIG['num_epochs']}]")

        avg_loss, losses_dict = train_one_epoch(
            model, optimizer, train_loader, device, epoch, CONFIG["num_epochs"]
        )
        val_map50    = evaluate_epoch(model, val_loader, device, CONFIG["score_threshold"])
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss)
        history['val_map50'].append(val_map50)
        history['loss_classifier'].append(losses_dict['loss_classifier'])
        history['loss_box_reg'].append(losses_dict['loss_box_reg'])
        history['loss_objectness'].append(losses_dict['loss_objectness'])
        history['loss_rpn_box_reg'].append(losses_dict['loss_rpn_box_reg'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"   Loss: {avg_loss:.4f} | mAP@50: {val_map50:.4f} | LR: {current_lr:.6f} | ⏱️ {format_time(epoch_time)}")

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':                best_map50,
                'num_classes':          num_classes,
                'classes':              CONFIG["classes"],
                'cat_mapping':          cat_mapping,
            }, os.path.join(weights_dir, "best.pth"))
            print(f"   💾 Meilleur modèle sauvegardé (mAP@50: {best_map50:.4f})")

        if epoch % CONFIG["save_every"] == 0 or epoch == CONFIG["num_epochs"]:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':                val_map50,
                'num_classes':          num_classes,
                'classes':              CONFIG["classes"],
                'cat_mapping':          cat_mapping,
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # Copier les modèles finaux
    # -------------------------------------------------------------------------
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   ✅ {dst} ({os.path.getsize(dst_path) / 1024 / 1024:.1f} MB)")

    # -------------------------------------------------------------------------
    # Historique + graphiques + rapport
    # -------------------------------------------------------------------------
    history['time_stats'] = {
        'total_time':              total_time,
        'total_time_formatted':    format_time(total_time),
        'avg_epoch_time_formatted':format_time(total_time / CONFIG["num_epochs"]),
    }
    history['config']     = CONFIG
    history['best_map50'] = best_map50

    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_title('Loss totale'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['val_map50'], 'g-', label='mAP@50')
    axes[0, 1].set_title('mAP@50 (Validation)'); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(epochs, history['loss_classifier'], label='Classifier')
    axes[1, 0].plot(epochs, history['loss_box_reg'],    label='Box Reg')
    axes[1, 0].set_title('Losses classification & box')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['loss_objectness'],  label='Objectness')
    axes[1, 1].plot(epochs, history['loss_rpn_box_reg'], label='RPN Box Reg')
    axes[1, 1].set_title('Losses RPN')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
    plt.close()

    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write("Faster R-CNN (ResNet-50 FPN) - Rapport\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {CONFIG['images_dir']}\n")
        f.write(f"Classes: {CONFIG['classes']}\n")
        f.write(f"Augmentation différentielle: activée\n")
        f.write(f"Classes rares sur-échantillonnées: {rare_class_names}\n")
        f.write(f"Epochs: {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}\n\n")
        f.write(f"Meilleur mAP@50: {best_map50:.4f}\n")
        f.write(f"Temps total: {format_time(total_time)}\n")
        f.write(f"Chemin: {train_dir}\n")

    print("\n" + "=" * 70)
    print("   🎉 TERMINÉ")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   ⏱️  Temps: {format_time(total_time)}")
    print(f"   📁 Modèles: {train_dir}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    train_fasterrcnn()