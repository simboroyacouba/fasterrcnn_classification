"""
Entraînement Faster R-CNN pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)

Modes d'entraînement:
  simple    : Faster R-CNN standard, hyperparamètres fixes
  attention : Faster R-CNN + CBAM sur le FPN, hyperparamètres fixes
  optimize  : Faster R-CNN standard + recherche bayésienne des hyperparamètres (Optuna)
  dual      : entraînement séquentiel nadir (panneau_solaire) puis oblique (batiments)
"""

import copy
import os
import json
import yaml
import shutil
import random
import argparse
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ColorJitter
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

MODE_CLASSES = {
    "nadir":   ["panneau_solaire"],
    "oblique": [
        "batiment_peint", "batiment_non_enduit",
        "batiment_enduit", "menuiserie_metallique",
    ],
}

OPTUNA_CONFIG = {
    "n_trials":          30,
    "n_epochs_per_trial": 5,
    "study_name":        "fasterrcnn_cadastral",
    "output_dir":        "./optuna_output",
}


def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    print(f"Classes chargees depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


def load_classes_for_mode(yaml_path, mode):
    all_classes = load_classes(yaml_path)
    if mode in MODE_CLASSES:
        keep = set(MODE_CLASSES[mode])
        filtered = [c for c in all_classes if c == '__background__' or c in keep]
        return filtered
    return all_classes


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "images_dir":        os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
    "annotations_file":  os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "output_dir":        os.getenv("OUTPUT_DIR", "./output"),
    "classes_file":      os.getenv("CLASSES_FILE", "classes.yaml"),
    "classes":           None,
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
    # Chemins specifiques par mode
    "nadir_annotations_file":   os.getenv("NADIR_ANNOTATIONS_FILE",   ""),
    "nadir_classes_file":       os.getenv("NADIR_CLASSES_FILE",       "classes_nadir.yaml"),
    "oblique_annotations_file": os.getenv("OBLIQUE_ANNOTATIONS_FILE", ""),
    "oblique_classes_file":     os.getenv("OBLIQUE_CLASSES_FILE",     "classes_oblique.yaml"),
    # CBAM (mode attention)
    "cbam_reduction":   16,
    "cbam_kernel_size": 7,
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
    print(f"\n   Split des IMAGES (total: {n_total}):")
    print(f"      Train: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"      Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"      Test:  {n_test}  ({n_test/n_total*100:.1f}%)")
    train_ids = all_image_ids[:n_train]
    val_ids   = all_image_ids[n_train:n_train + n_val]
    test_ids  = all_image_ids[n_train + n_val:]
    stats = {'train': {}, 'val': {}, 'test': {}}
    for cat_id in coco.getCatIds():
        stats['train'][cat_id] = 0; stats['val'][cat_id] = 0; stats['test'][cat_id] = 0
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
    print("\n   Distribution des classes (split 70/20/10):")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = " !" if val == 0 or test == 0 else ""
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8}{ok}")
    print(f"   {'-'*70}")


# =============================================================================
# AUGMENTATION PAR CLASSE
# =============================================================================

def parse_aug_coeffs(aug_args, classes):
    """
    Parse les coefficients d'augmentation depuis les arguments CLI.
    Format: nom_partiel:coeff  (ex: --aug panneau:2 batiment:3)
    Correspondance partielle insensible a la casse.
    Valeur par defaut: 1 (aucune augmentation).
    """
    real_classes = [c for c in classes if c != '__background__']
    coeffs = {cls: 1 for cls in real_classes}
    for item in (aug_args or []):
        if ':' not in item:
            print(f"Format invalide (ignore): {item!r} — attendu: classe:coeff")
            continue
        key, val = item.rsplit(':', 1)
        try:
            coeff = max(1, int(val))
        except ValueError:
            print(f"Coefficient invalide (ignore): {val!r}")
            continue
        matched = [c for c in real_classes if key.lower() in c.lower()]
        if not matched:
            print(f"Classe non trouvee (ignoree): {key!r}  — classes dispo: {real_classes}")
            continue
        for cls in matched:
            coeffs[cls] = coeff
    return coeffs


# =============================================================================
# MÉCANISME D'ATTENTION (CBAM) — mode "attention" uniquement
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.fc[-1].weight)

    def forward(self, x):
        b, c = x.shape[:2]
        avg   = self.fc(self.avg_pool(x).view(b, c))
        mx    = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class AttentionFPN(nn.Module):
    """Enveloppe FPN qui applique CBAM a chaque niveau de features."""

    def __init__(self, fpn, out_channels=256, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        self.fpn = fpn
        self.cbam_modules = nn.ModuleDict({
            '0':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '1':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '2':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '3':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            'pool': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
        })

    def forward(self, x):
        features = self.fpn(x)
        attended = OrderedDict()
        for key, feat in features.items():
            attended[key] = self.cbam_modules[key](feat) if key in self.cbam_modules else feat
        return attended


# =============================================================================
# MODÈLE
# =============================================================================

def get_model_simple(num_classes, pretrained=True):
    """Faster R-CNN standard."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model_attention(num_classes, pretrained=True, cbam_reduction=16, cbam_kernel_size=7):
    """Faster R-CNN avec CBAM sur le FPN."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    fpn_out = model.backbone.out_channels   # 256
    model.backbone.fpn = AttentionFPN(model.backbone.fpn, fpn_out, cbam_reduction, cbam_kernel_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# =============================================================================
# DATASET PYTORCH
# =============================================================================

def compute_sample_weights(coco, image_ids, cat_mapping, classes, aug_coeffs):
    """Poids par image pour WeightedRandomSampler basé sur les coefficients d'augmentation."""
    class_idx_to_coeff = {
        classes.index(name): coeff
        for name, coeff in aug_coeffs.items()
        if name in classes
    }
    weights = []
    for img_id in image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_weight = 1.0
        for ann in anns:
            cls_idx = cat_mapping.get(ann['category_id'])
            if cls_idx in class_idx_to_coeff:
                img_weight = max(img_weight, float(class_idx_to_coeff[cls_idx]))
        weights.append(img_weight)
    return weights


def augment_sample(image, boxes, labels, image_size):
    W = H = image_size
    if random.random() < 0.5:
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        image = jitter(image)
    if random.random() < 0.3:
        blur_radius = random.uniform(0.5, 1.5) * (image_size / 640)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if random.random() < 0.5:
        image = TF.hflip(image)
        boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]
    if random.random() < 0.5:
        image = TF.vflip(image)
        boxes = [[x1, H - y2, x2, H - y1] for x1, y1, x2, y2 in boxes]
    if random.random() < 0.5:
        image = TF.rotate(image, 180, expand=False)
        boxes = [[W - x2, H - y2, W - x1, H - y1] for x1, y1, x2, y2 in boxes]
    kept = [(i, [max(0, x1), max(0, y1), min(W, x2), min(H, y2)])
            for i, (x1, y1, x2, y2) in enumerate(boxes) if x2 > x1 and y2 > y1]
    if kept:
        idxs, boxes = zip(*kept)
        boxes  = list(boxes)
        labels = [labels[i] for i in idxs]
    else:
        boxes, labels = [], []
    return image, boxes, labels


class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping,
                 image_size=640, augment=False):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
        self.image_size  = image_size
        self.augment     = augment

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image    = image.resize((self.image_size, self.image_size))
        scale_x  = self.image_size / orig_w
        scale_y  = self.image_size / orig_h
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            if ann.get('iscrowd', 0): continue
            class_id = self.cat_mapping.get(ann['category_id'])
            if class_id is None: continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue
            x1 = max(0, x * scale_x); y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2]); labels.append(class_id)
        if self.augment and boxes:
            image, boxes, labels = augment_sample(image, boxes, labels, self.image_size)
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


def compute_map(predictions, ground_truths, classes, iou_threshold=0.5):
    all_cls = set(range(1, len(classes)))
    for gt in ground_truths:
        all_cls.update(gt['labels'].tolist())
    all_cls.discard(0)
    aps = []; per_class_ap = {}
    for cls in all_cls:
        tps, fps, scores_list = [], [], []
        n_gt = sum((gt['labels'] == cls).sum().item() for gt in ground_truths)
        if n_gt == 0: continue
        for pred, gt in zip(predictions, ground_truths):
            mask_p = pred['labels'] == cls; mask_g = gt['labels'] == cls
            p_boxes  = pred['boxes'][mask_p].cpu().numpy()
            p_scores = pred['scores'][mask_p].cpu().numpy()
            g_boxes  = gt['boxes'][mask_g].cpu().numpy()
            matched  = set()
            for i in np.argsort(-p_scores):
                scores_list.append(p_scores[i])
                if not len(g_boxes): tps.append(0); fps.append(1); continue
                ious   = [calculate_iou(p_boxes[i], g) for g in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j); tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)
        if not scores_list: aps.append(0.0); per_class_ap[cls] = 0.0; continue
        order  = np.argsort(-np.array(scores_list))
        tp_cum = np.cumsum(np.array(tps)[order])
        fp_cum = np.cumsum(np.array(fps)[order])
        prec   = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec    = tp_cum / (n_gt + 1e-10)
        ap = sum(np.max(prec[rec >= t]) if (rec >= t).any() else 0 for t in np.arange(0, 1.1, 0.1)) / 11
        aps.append(ap); per_class_ap[cls] = ap
    return float(np.mean(aps)) if aps else 0.0, per_class_ap


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, device, epoch, num_epochs):
    model.train()
    total_loss = 0; losses_dict = {'loss_classifier': 0, 'loss_box_reg': 0,
                                   'loss_objectness': 0, 'loss_rpn_box_reg': 0}
    num_batches = 0
    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if all(len(t['boxes']) == 0 for t in targets): continue
        try:
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad(); losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses.item()
            for k in losses_dict:
                losses_dict[k] += loss_dict.get(k, torch.tensor(0)).item()
            num_batches += 1
        except Exception as e:
            print(f"   Erreur batch: {e}"); continue
    n = max(num_batches, 1)
    avg_loss = total_loss / n
    for k in losses_dict: losses_dict[k] /= n
    return avg_loss, losses_dict


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, classes, score_threshold=0.5):
    model.eval()
    all_preds = []; all_gts = []
    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            keep = output['scores'] >= score_threshold
            all_preds.append({'boxes':  output['boxes'][keep].cpu(),
                              'labels': output['labels'][keep].cpu(),
                              'scores': output['scores'][keep].cpu()})
            all_gts.append({'boxes': target['boxes'], 'labels': target['labels']})
    map50, per_class_ap = compute_map(all_preds, all_gts, classes, iou_threshold=0.5)
    per_class_named = {classes[cls]: ap for cls, ap in per_class_ap.items() if cls < len(classes)}
    return map50, per_class_named


# =============================================================================
# OPTIMISATION BAYÉSIENNE (OPTUNA) — mode "optimize" uniquement
# =============================================================================

def _run_optimization(device, train_loader, val_loader, num_classes, pretrained):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    classes = CONFIG["classes"]

    ref_model     = get_model_simple(num_classes, pretrained)
    initial_state = copy.deepcopy(ref_model.state_dict())
    del ref_model

    def objective(trial):
        lr           = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)
        momentum     = trial.suggest_float("momentum",      0.80, 0.99)

        model = get_model_simple(num_classes, pretrained=False)
        model.load_state_dict(initial_state)
        model.to(device)
        params    = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        best_val = 0.0
        for epoch in range(OPTUNA_CONFIG["n_epochs_per_trial"]):
            train_one_epoch(model, optimizer, train_loader, device, epoch + 1, OPTUNA_CONFIG["n_epochs_per_trial"])
            val_map50, _ = evaluate_epoch(model, val_loader, device, classes, CONFIG["score_threshold"])
            trial.report(val_map50, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_val = max(best_val, val_map50)
        del model; gc.collect()
        return best_val

    os.makedirs(OPTUNA_CONFIG["output_dir"], exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name=OPTUNA_CONFIG["study_name"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print(f"\n{'='*70}")
    print(f"   OPTIMISATION BAYESIENNE — {OPTUNA_CONFIG['n_trials']} essais")
    print(f"   {OPTUNA_CONFIG['n_epochs_per_trial']} epochs/essai | sampler: TPE | pruner: Median")
    print(f"{'='*70}\n")

    study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"], show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"   MEILLEUR ESSAI #{best.number}  —  mAP@50: {best.value:.4f}")
    print(f"{'='*70}")
    for k, v in best.params.items():
        print(f"   {k}: {v}")

    report = {
        "best_trial": best.number, "best_map50": best.value, "best_params": best.params,
        "all_trials": [{"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                       for t in study.trials],
    }
    with open(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        values = [t.value for t in study.trials if t.value is not None]
        axes[0].plot(values, marker='o', linewidth=1.5)
        axes[0].set_xlabel("Essai"); axes[0].set_ylabel("mAP@50")
        axes[0].set_title("Historique Optuna"); axes[0].grid(True, alpha=0.3)
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys()), list(importances.values()))
        axes[1].set_xlabel("Importance"); axes[1].set_title("Importance des hyperparametres")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_results.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    return best.params


# =============================================================================
# ENTRAÎNEMENT D'UN SEUL MODÈLE
# =============================================================================

def _train_single(config, aug_coeffs, mode_label, training_mode, cbam_r, cbam_k):
    """
    Entraîne un modèle Faster R-CNN (mode simple ou attention).
    Retourne (model, history, train_dir).
    """
    classes     = config["classes"]
    num_classes = len(classes)

    print("=" * 70)
    print(f"   Faster R-CNN (ResNet-50 FPN) — Mode : {mode_label.upper()}")
    print("=" * 70)
    print(f"\n   Mode:        {mode_label}")
    print(f"   Images:      {config['images_dir']}")
    print(f"   Annotations: {config['annotations_file']}")
    print(f"   Classes:     {num_classes} — {classes}")
    print(f"   Epochs:      {config['num_epochs']} | Batch: {config['batch_size']} | LR: {config['learning_rate']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:      {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join(config["output_dir"], f"fasterrcnn_{mode_label}_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    coco = COCO(config["annotations_file"])
    coco_cats   = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    cat_mapping = {}
    for cat_id, cat_name in coco_cats.items():
        if cat_name in classes:
            cat_mapping[cat_id] = classes.index(cat_name)

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    test_info = {'test_image_ids': test_ids, 'cat_mapping': {str(k): v for k, v in cat_mapping.items()},
                 'images_dir': config["images_dir"], 'annotations_file': config["annotations_file"],
                 'num_test_images': len(test_ids)}
    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    train_dataset = CocoDetectionDataset(config["images_dir"], config["annotations_file"],
                                         train_ids, cat_mapping, config["image_size"], augment=True)
    val_dataset   = CocoDetectionDataset(config["images_dir"], config["annotations_file"],
                                         val_ids,   cat_mapping, config["image_size"])

    sample_weights = compute_sample_weights(coco, train_ids, cat_mapping, classes, aug_coeffs)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    n_aug   = sum(1 for w in sample_weights if w > 1.0)
    print(f"   WeightedSampler: {n_aug}/{len(sample_weights)} images surrepresentees")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    print(f"\n   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_ids)}")

    print(f"\n   Chargement Faster R-CNN (pretrained={config['pretrained']}, mode={training_mode})...")
    if training_mode == "attention":
        model = get_model_attention(num_classes, config["pretrained"], cbam_r, cbam_k)
        print(f"   CBAM injecte sur FPN (reduction={cbam_r}, kernel_size={cbam_k})")
    else:
        model = get_model_simple(num_classes, config["pretrained"])
    model.to(device)

    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.SGD(params, lr=config["learning_rate"],
                                   momentum=config["momentum"], weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-5
    )

    print("\n" + "=" * 70)
    print(f"   ENTRAINEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history   = {'train_loss': [], 'val_map50': [], 'lr': []}
    best_map50 = 0.0
    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch}/{config['num_epochs']}]")
        avg_loss, _ = train_one_epoch(model, optimizer, train_loader, device, epoch, config["num_epochs"])
        val_map50, per_class_ap = evaluate_epoch(model, val_loader, device, classes, config["score_threshold"])
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss); history['val_map50'].append(val_map50)
        history['lr'].append(current_lr)
        print(f"   Loss: {avg_loss:.4f} | mAP@50: {val_map50:.4f} | LR: {current_lr:.6f} | {format_time(time.time()-epoch_start)}")
        for cls_name, ap in per_class_ap.items():
            if cls_name != '__background__':
                print(f"      {cls_name:<30} AP@50: {ap:.4f}")
        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'map50': best_map50,
                        'num_classes': num_classes, 'classes': classes, 'cat_mapping': cat_mapping},
                       os.path.join(weights_dir, "best.pth"))
            print(f"   Meilleur modele sauvegarde (mAP@50: {best_map50:.4f})")
        if epoch % config["save_every"] == 0 or epoch == config["num_epochs"]:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'map50': val_map50,
                        'num_classes': num_classes, 'classes': classes, 'cat_mapping': cat_mapping},
                       os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    history['best_map50'] = best_map50
    history['time_stats'] = {'total_time': total_time, 'total_time_formatted': format_time(total_time),
                              'avg_epoch_time_formatted': format_time(total_time / config["num_epochs"])}
    history['config']     = {k: str(v) for k, v in config.items()}
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs_r = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs_r, history['train_loss'], 'b-'); axes[0].set_title('Loss (train)'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs_r, history['val_map50'],  'g-'); axes[1].set_title('mAP@50 (val)')
        axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150); plt.close()

    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Faster R-CNN — Mode {mode_label}\n{'='*50}\n\n")
        f.write(f"Classes: {classes}\nEpochs: {config['num_epochs']}\n")
        f.write(f"Meilleur mAP@50: {best_map50:.4f}\nTemps: {format_time(total_time)}\n")
        f.write(f"Chemin: {train_dir}\n")

    print("\n" + "=" * 70)
    print(f"   TERMINE — {mode_label.upper()}")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   Temps: {format_time(total_time)}")
    print(f"   Modele: {train_dir}")
    print("=" * 70)

    return model, history, train_dir


def _build_sub_config(base_config, mode_label):
    """Construit une config pour le mode nadir ou oblique en mode dual."""
    cfg = dict(base_config)
    ann_dir = os.path.dirname(os.path.abspath(cfg["annotations_file"]))
    if mode_label == "nadir":
        if cfg["nadir_annotations_file"]:
            cfg["annotations_file"] = cfg["nadir_annotations_file"]
        else:
            candidate = os.path.join(ann_dir, "instances_nadir.json")
            if os.path.exists(candidate):
                cfg["annotations_file"] = candidate
        cfg["output_dir"] = os.path.join(cfg["output_dir"], "nadir")
        cfg["classes"]    = load_classes_for_mode(cfg["classes_file"], "nadir")
    else:  # oblique
        if cfg["oblique_annotations_file"]:
            cfg["annotations_file"] = cfg["oblique_annotations_file"]
        else:
            candidate = os.path.join(ann_dir, "instances_oblique.json")
            if os.path.exists(candidate):
                cfg["annotations_file"] = candidate
        cfg["output_dir"] = os.path.join(cfg["output_dir"], "oblique")
        cfg["classes"]    = load_classes_for_mode(cfg["classes_file"], "oblique")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Faster R-CNN - Toitures Cadastrales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  simple    Faster R-CNN standard, hyperparametres fixes
  attention Faster R-CNN + CBAM sur le FPN
  optimize  Faster R-CNN standard + recherche bayesienne (Optuna)
  dual      Entrainement sequentiel nadir (panneau_solaire) + oblique (batiments)

  nadir / oblique / all  aliases deprecies — utiliser dual ou simple a la place
        """
    )
    parser.add_argument("--mode", default="simple",
                        choices=["simple", "attention", "optimize", "dual", "nadir", "oblique", "all"],
                        help="Mode d'entrainement")
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--freeze-epochs",    type=int, default=0,
                        help="Nombre d'epochs avec backbone gele (0=desactive)")
    parser.add_argument("--aug", nargs='*', default=[], metavar='CLASSE:COEFF',
                        help="Coefficients d'augmentation par classe (ex: --aug panneau:2 batiment:3)")
    parser.add_argument("--cbam-reduction",   type=int, default=CONFIG["cbam_reduction"],
                        help=f"CBAM reduction — mode attention (defaut: {CONFIG['cbam_reduction']})")
    parser.add_argument("--cbam-kernel-size", type=int, default=CONFIG["cbam_kernel_size"],
                        choices=[3, 5, 7],
                        help=f"CBAM kernel size — mode attention (defaut: {CONFIG['cbam_kernel_size']})")
    parser.add_argument("--n-trials",       type=int, default=OPTUNA_CONFIG["n_trials"],
                        help="Nombre d'essais Optuna — mode optimize")
    parser.add_argument("--n-epochs-trial", type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"],
                        help="Epochs par essai Optuna")
    args = parser.parse_args()

    OPTUNA_CONFIG["n_trials"]           = args.n_trials
    OPTUNA_CONFIG["n_epochs_per_trial"] = args.n_epochs_trial

    if args.images_dir:       CONFIG["images_dir"]        = args.images_dir
    if args.annotations_file: CONFIG["annotations_file"]  = args.annotations_file
    if args.classes_file:     CONFIG["classes_file"]      = args.classes_file
    if args.output_dir:       CONFIG["output_dir"]        = args.output_dir

    mode = args.mode

    # Aliases deprecies
    if mode == "nadir":
        print("   [AVERTISSEMENT] --mode nadir est deprecie — utilisez --mode dual")
        CONFIG["classes"] = load_classes_for_mode(CONFIG["classes_file"], "nadir")
        nadir_cfg = _build_sub_config(CONFIG, "nadir")
        aug_coeffs = parse_aug_coeffs(args.aug, CONFIG["classes"])
        _train_single(nadir_cfg, aug_coeffs, "nadir", "simple", args.cbam_reduction, args.cbam_kernel_size)
        return
    elif mode == "oblique":
        print("   [AVERTISSEMENT] --mode oblique est deprecie — utilisez --mode dual")
        CONFIG["classes"] = load_classes_for_mode(CONFIG["classes_file"], "oblique")
        oblique_cfg = _build_sub_config(CONFIG, "oblique")
        aug_coeffs = parse_aug_coeffs(args.aug, CONFIG["classes"])
        _train_single(oblique_cfg, aug_coeffs, "oblique", "simple", args.cbam_reduction, args.cbam_kernel_size)
        return
    elif mode == "all":
        print("   [AVERTISSEMENT] --mode all est deprecie — utilisez --mode simple")
        mode = "simple"

    CONFIG["classes"] = load_classes(CONFIG["classes_file"])

    print("=" * 70)
    print("   FASTER R-CNN - Detection des Toitures Cadastrales")
    print(f"   Mode: {mode.upper()}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"   GPU:  {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    aug_coeffs = parse_aug_coeffs(args.aug, CONFIG["classes"])

    if mode == "dual":
        print(f"\nMode dual : entrainement nadir puis oblique")
        nadir_cfg   = _build_sub_config(CONFIG, "nadir")
        oblique_cfg = _build_sub_config(CONFIG, "oblique")
        aug_nadir   = parse_aug_coeffs(args.aug, nadir_cfg["classes"])
        aug_oblique = parse_aug_coeffs(args.aug, oblique_cfg["classes"])
        print("\n--- Nadir ---")
        _train_single(nadir_cfg,   aug_nadir,   "nadir",   "simple", args.cbam_reduction, args.cbam_kernel_size)
        print("\n--- Oblique ---")
        _train_single(oblique_cfg, aug_oblique, "oblique", "simple", args.cbam_reduction, args.cbam_kernel_size)
        return

    if mode == "optimize":
        print(f"\nArchitecture: Faster R-CNN ResNet-50 FPN (standard)")
        print("Lancement de l'optimisation bayesienne des hyperparametres...")
        coco = COCO(CONFIG["annotations_file"])
        cat_ids   = coco.getCatIds()
        coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
        classes   = CONFIG["classes"]
        cat_mapping = {cat_id: classes.index(name) for cat_id, name in coco_cats.items() if name in classes}
        all_ids   = [i for i in coco.imgs if coco.getAnnIds(imgIds=i)]
        np.random.seed(42); np.random.shuffle(all_ids)
        n = len(all_ids); n_train = int(n * CONFIG["train_split"]); n_val = int(n * CONFIG["val_split"])
        train_ids = all_ids[:n_train]; val_ids = all_ids[n_train:n_train + n_val]
        train_ds = CocoDetectionDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                                        train_ids, cat_mapping, CONFIG["image_size"], augment=True)
        val_ds   = CocoDetectionDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                                        val_ids,   cat_mapping, CONFIG["image_size"])
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                                  collate_fn=collate_fn, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                                  collate_fn=collate_fn, num_workers=0)
        num_classes  = len(classes)
        best_params  = _run_optimization(device, train_loader, val_loader, num_classes, CONFIG["pretrained"])
        for key in ("learning_rate", "weight_decay", "momentum"):
            if key in best_params: CONFIG[key] = best_params[key]
        print(f"\nHyperparametres optimises appliques a l'entrainement final.")
        _train_single(CONFIG, aug_coeffs, "optimize", "simple", args.cbam_reduction, args.cbam_kernel_size)
        return

    training_mode = "attention" if mode == "attention" else "simple"
    if mode == "attention":
        print(f"\nArchitecture: Faster R-CNN + CBAM(reduction={args.cbam_reduction}, kernel={args.cbam_kernel_size})")

    _train_single(CONFIG, aug_coeffs, mode, training_mode, args.cbam_reduction, args.cbam_kernel_size)


if __name__ == "__main__":
    main()
