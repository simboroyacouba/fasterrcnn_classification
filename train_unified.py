"""
Entrainement Faster R-CNN unifie — toutes classes, un seul modele

Entraine un unique modele sur l'ensemble du dataset (toutes images, toutes
classes) sans separation nadir / oblique.

Usage :
  python train_unified.py
  python train_unified.py --freeze-epochs 5
  python train_unified.py --images-dir /path/to/images
"""

import os
import json
import yaml
import shutil
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
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
# CONSTANTES
# =============================================================================

# Poids d'oversampling pour les classes sous-representees (1.0 = pas d'effet)
RARE_CLASS_WEIGHTS = {
    "panneau_solaire":       1.0,
    "batiment_peint":        4.0,
    "batiment_enduit":       1.0,
    "batiment_non_enduit":   2.0,
    "menuiserie_metallique": 1.0,
}


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
    print(f"Classes depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config(args):
    return {
        "images_dir":       args.images_dir or os.getenv(
                                "DETECTION_DATASET_IMAGES_DIR",
                                "../dataset1/images/default",
                            ),
        "annotations_file": args.annotations_file or os.getenv(
                                "DETECTION_DATASET_ANNOTATIONS_FILE",
                                "../dataset1/annotations/instances_default.json",
                            ),
        "output_dir":       args.output_dir or os.getenv("OUTPUT_DIR", "./runs/detect/train"),
        "classes_file":     args.classes_file or os.getenv("CLASSES_FILE", "classes.yaml"),
        "classes":          None,
        "num_epochs":       int(os.getenv("NUM_EPOCHS",   "50")),
        "batch_size":       int(os.getenv("BATCH_SIZE",   "2")),
        "learning_rate":    float(os.getenv("LEARNING_RATE", "0.005")),
        "momentum":         float(os.getenv("MOMENTUM",   "0.9")),
        "weight_decay":     float(os.getenv("WEIGHT_DECAY", "0.0005")),
        "image_size":       int(os.getenv("IMAGE_SIZE",   "640")),
        "train_split":      float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":        float(os.getenv("VAL_SPLIT",  "0.20")),
        "test_split":       float(os.getenv("TEST_SPLIT", "0.10")),
        "save_every":       int(os.getenv("SAVE_EVERY",   "5")),
        "score_threshold":  float(os.getenv("SCORE_THRESHOLD", "0.5")),
        "pretrained":       os.getenv("PRETRAINED", "true").lower() == "true",
        "freeze_epochs":    args.freeze_epochs,
        "rare_class_weights": RARE_CLASS_WEIGHTS,
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
    all_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val

    if n_test < 1 and n_total > 2:
        n_test  = max(1, int(n_total * 0.10))
        n_train = n_total - n_val - n_test

    print(f"\n   Split des images (total : {n_total}) :")
    print(f"      Train : {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"      Val   : {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"      Test  : {n_test}  ({n_test/n_total*100:.1f}%)")

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]

    stats = {'train': {}, 'val': {}, 'test': {}}
    for cat_id in coco.getCatIds():
        stats['train'][cat_id] = 0
        stats['val'][cat_id]   = 0
        stats['test'][cat_id]  = 0

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
    print(f"\n   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*62}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = "!" if val == 0 or test == 0 else ""
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8}  {ok}")
    print(f"   {'-'*62}")


# =============================================================================
# DATASET PYTORCH
# =============================================================================

def compute_sample_weights(coco, image_ids, cat_mapping, classes, rare_class_weights):
    class_idx_to_weight = {
        classes.index(name): weight
        for name, weight in rare_class_weights.items()
        if name in classes
    }
    weights = []
    for img_id in image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        img_weight = 1.0
        for ann in anns:
            cls_idx = cat_mapping.get(ann['category_id'])
            if cls_idx in class_idx_to_weight:
                img_weight = max(img_weight, class_idx_to_weight[cls_idx])
        weights.append(img_weight)
    return weights


def augment_sample(image, boxes, labels, image_size):
    W = H = image_size

    if random.random() < 0.5:
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        image  = jitter(image)

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

    kept = [
        (i, [max(0, x1), max(0, y1), min(W, x2), min(H, y2)])
        for i, (x1, y1, x2, y2) in enumerate(boxes)
        if x2 > x1 and y2 > y1
    ]
    if kept:
        idxs, boxes = zip(*kept)
        boxes  = list(boxes)
        labels = [labels[i] for i in idxs]
    else:
        boxes, labels = [], []

    return image, boxes, labels


class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping, image_size=640, augment=False):
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

        image  = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image  = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

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

        if self.augment and len(boxes) > 0:
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
# MODELE
# =============================================================================

def build_model(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# =============================================================================
# METRIQUES
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
    all_class_indices = set(range(1, len(classes)))
    for gt in ground_truths:
        all_class_indices.update(gt['labels'].tolist())
    all_class_indices.discard(0)

    aps = []
    per_class_ap = {}

    for cls in all_class_indices:
        tps, fps, scores_list = [], [], []
        n_gt = sum((gt['labels'] == cls).sum().item() for gt in ground_truths)
        if n_gt == 0:
            continue

        for pred, gt in zip(predictions, ground_truths):
            mask_p  = pred['labels'] == cls
            mask_g  = gt['labels']   == cls
            p_boxes = pred['boxes'][mask_p].cpu().numpy()
            p_scores= pred['scores'][mask_p].cpu().numpy()
            g_boxes = gt['boxes'][mask_g].cpu().numpy()

            matched = set()
            for i in np.argsort(-p_scores):
                scores_list.append(p_scores[i])
                if len(g_boxes) == 0:
                    tps.append(0); fps.append(1)
                    continue
                ious   = [calculate_iou(p_boxes[i], g) for g in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j); tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)

        if not scores_list:
            aps.append(0.0); per_class_ap[cls] = 0.0; continue

        order    = np.argsort(-np.array(scores_list))
        tp_cum   = np.cumsum(np.array(tps)[order])
        fp_cum   = np.cumsum(np.array(fps)[order])
        precision = tp_cum / (tp_cum + fp_cum + 1e-10)
        recall    = tp_cum / (n_gt + 1e-10)

        ap = 0
        for r_thresh in np.arange(0, 1.1, 0.1):
            mask = recall >= r_thresh
            ap  += np.max(precision[mask]) if mask.any() else 0
        ap /= 11
        aps.append(ap); per_class_ap[cls] = ap

    return float(np.mean(aps)) if aps else 0.0, per_class_ap


# =============================================================================
# ENTRAINEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss  = 0
    losses_dict = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
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
            print(f"   Avertissement batch : {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    for k in losses_dict:
        losses_dict[k] /= max(num_batches, 1)

    return avg_loss, losses_dict


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, classes, score_threshold=0.5):
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

    map50, per_class_ap = compute_map(all_preds, all_gts, classes, iou_threshold=0.5)
    per_class_named = {classes[cls]: ap for cls, ap in per_class_ap.items() if cls < len(classes)}
    return map50, per_class_named


# =============================================================================
# MAIN
# =============================================================================

def train_fasterrcnn():
    parser = argparse.ArgumentParser(
        description="Entrainement Faster R-CNN unifie — toutes classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir",       default=None, help="Dossier images")
    parser.add_argument("--annotations-file", default=None, help="Fichier annotations COCO (.json)")
    parser.add_argument("--classes-file",     default=None, help="Fichier classes (.yaml)")
    parser.add_argument("--output-dir",       default=None, help="Dossier de sortie")
    parser.add_argument(
        "--freeze-epochs",
        type=int, default=0,
        help="Nombre d'epochs avec backbone gele (0 = desactive).",
    )
    args = parser.parse_args()

    config = build_config(args)
    config["classes"] = load_classes(config["classes_file"])
    num_classes = len(config["classes"])

    print("=" * 70)
    print(f"   Faster R-CNN (ResNet-50 FPN) — Modele UNIFIE (toutes classes)")
    print("=" * 70)
    print(f"\n   Images      : {config['images_dir']}")
    print(f"   Annotations : {config['annotations_file']}")
    print(f"   Classes     : {num_classes} (avec __background__)")
    print(f"   Epochs      : {config['num_epochs']} | Batch : {config['batch_size']} | LR : {config['learning_rate']}")
    print(f"   Freeze      : {config['freeze_epochs']} epochs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device      : {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join(config["output_dir"], f"fasterrcnn_unified_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Split
    coco     = COCO(config["annotations_file"])
    cat_ids  = coco.getCatIds()
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
    classes   = config["classes"]

    cat_mapping = {}
    for cat_id, cat_name in coco_cats.items():
        if cat_name in classes:
            cat_mapping[cat_id] = classes.index(cat_name)
        else:
            print(f"   Categorie ignoree (absente du yaml): '{cat_name}' (id={cat_id})")

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    # Sauvegarder les IDs du test set
    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       config["images_dir"],
        'annotations_file': config["annotations_file"],
        'num_test_images':  len(test_ids),
        'classes':          classes,
    }
    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    # Datasets & DataLoaders
    train_dataset = CocoDetectionDataset(
        config["images_dir"], config["annotations_file"],
        train_ids, cat_mapping, config["image_size"], augment=True,
    )
    val_dataset = CocoDetectionDataset(
        config["images_dir"], config["annotations_file"],
        val_ids, cat_mapping, config["image_size"],
    )

    sample_weights = compute_sample_weights(
        coco, train_ids, cat_mapping, classes, config["rare_class_weights"]
    )
    sampler   = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    n_rare    = sum(1 for w in sample_weights if w > 1.0)
    print(f"\n   WeightedSampler : {n_rare}/{len(sample_weights)} images avec classes rares")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"   Train : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_ids)}")

    # Modele & Optimiseur
    print(f"\nChargement Faster R-CNN ResNet-50 FPN (pretrained={config['pretrained']})...")
    model = build_model(num_classes, pretrained=config["pretrained"])
    model.to(device)

    freeze_epochs = config["freeze_epochs"]
    if freeze_epochs > 0:
        print(f"   Staged training : backbone gele pour les {freeze_epochs} premieres epochs")
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=config["learning_rate"],
                                momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-5
    )

    # Boucle d'entrainement
    print("\n" + "=" * 70)
    print(f"   ENTRAINEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history = {
        'train_loss': [], 'val_map50': [],
        'loss_classifier': [], 'loss_box_reg': [],
        'loss_objectness': [], 'loss_rpn_box_reg': [],
        'lr': [],
    }

    best_map50 = 0.0
    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch}/{config['num_epochs']}]")

        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print(f"   Degel du backbone (epoch {epoch}), LR divise par 5")
            for param in model.parameters():
                param.requires_grad = True
            new_lr    = config["learning_rate"] / 5.0
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=new_lr, momentum=config["momentum"], weight_decay=config["weight_decay"],
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["num_epochs"] - freeze_epochs, eta_min=1e-5,
            )

        current_lr = optimizer.param_groups[0]['lr']
        avg_loss, losses_dict = train_one_epoch(model, optimizer, train_loader, device)
        val_map50, per_class_ap = evaluate_epoch(model, val_loader, device, classes, config["score_threshold"])
        lr_scheduler.step()

        history['train_loss'].append(avg_loss)
        history['val_map50'].append(val_map50)
        history['loss_classifier'].append(losses_dict['loss_classifier'])
        history['loss_box_reg'].append(losses_dict['loss_box_reg'])
        history['loss_objectness'].append(losses_dict['loss_objectness'])
        history['loss_rpn_box_reg'].append(losses_dict['loss_rpn_box_reg'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"   Loss : {avg_loss:.4f} | mAP@50 : {val_map50:.4f} | LR : {current_lr:.6f} | {format_time(epoch_time)}")
        for cls_name, ap in per_class_ap.items():
            if cls_name == '__background__':
                continue
            print(f"      {cls_name:<30} AP@50 : {ap:.4f}")

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':              best_map50,
                'num_classes':        num_classes,
                'classes':            classes,
                'cat_mapping':        cat_mapping,
            }, os.path.join(weights_dir, "best.pth"))
            print(f"   Meilleur modele sauvegarde (mAP@50 : {best_map50:.4f})")

        if epoch % config["save_every"] == 0 or epoch == config["num_epochs"]:
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':              val_map50,
                'num_classes':        num_classes,
                'classes':            classes,
                'cat_mapping':        cat_mapping,
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    # Copier les modeles
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_p = os.path.join(weights_dir, src)
        dst_p = os.path.join(train_dir, dst)
        if os.path.exists(src_p):
            shutil.copy2(src_p, dst_p)
            print(f"   {dst} ({os.path.getsize(dst_p) / 1024 / 1024:.1f} MB)")

    # Historique
    history['time_stats'] = {
        'total_time':               total_time,
        'total_time_formatted':     format_time(total_time),
        'avg_epoch_time_formatted': format_time(total_time / config["num_epochs"]),
    }
    history['config']    = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict))}
    history['best_map50'] = best_map50

    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Graphiques
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Courbes d'entrainement Faster R-CNN — Modele Unifie", fontsize=13)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].set_title('Loss totale'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['val_map50'], 'g-', label='mAP@50')
    axes[0, 1].set_title('mAP@50 (Val)'); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(epochs, history['loss_classifier'], label='Classifier')
    axes[1, 0].plot(epochs, history['loss_box_reg'],    label='Box Reg')
    axes[1, 0].set_title('Losses cls & box'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['loss_objectness'],  label='Objectness')
    axes[1, 1].plot(epochs, history['loss_rpn_box_reg'], label='RPN Box Reg')
    axes[1, 1].set_title('Losses RPN'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Rapport texte
    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write("Faster R-CNN (ResNet-50 FPN) — Modele Unifie\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset  : {config['images_dir']}\n")
        f.write(f"Classes  : {classes}\n")
        f.write(f"Epochs   : {config['num_epochs']} | Batch : {config['batch_size']}\n\n")
        f.write(f"mAP@50   : {best_map50:.4f}\n")
        f.write(f"Temps    : {format_time(total_time)}\n")
        f.write(f"Chemin   : {train_dir}\n")

    # model_info_unified.json pour evaluate_unified.py et inference_unified.py
    model_info = {
        'train_dir':        train_dir,
        'best_model':       os.path.join(train_dir, "best_model.pth"),
        'final_model':      os.path.join(train_dir, "final_model.pth"),
        'test_info':        os.path.join(train_dir, "test_info.json"),
        'classes':          classes,
        'image_size':       config["image_size"],
        'trained_at':       datetime.now().isoformat(),
    }
    info_path = os.path.join(config["output_dir"], "model_info_unified.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\n   model_info_unified.json : {info_path}")

    print("\n" + "=" * 70)
    print("   TERMINE — Modele unifie")
    print("=" * 70)
    print(f"   mAP@50  : {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   Temps   : {format_time(total_time)}")
    print(f"   Modeles : {train_dir}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    train_fasterrcnn()
