"""
evaluate_dual.py — FasterRCNN
Evalue les modeles nadir et oblique separement puis genere un rapport combine.
Metriques et structure JSON conformes au script YOLO de reference.

Usage :
    python evaluate_dual.py
    python evaluate_dual.py --nadir-model path/to/nadir.pth --oblique-model path/to/oblique.pth
    python evaluate_dual.py --only nadir
    python evaluate_dual.py --only oblique
"""

import os
import json
import yaml
import time
import argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

IOu_THRESHOLDS  = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
IMAGE_SIZE      = int(os.getenv("IMAGE_SIZE",      "640"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
OUTPUT_BASE     = os.getenv("EVALUATION_DIR", "./evaluation")

NADIR_CLASSES_FILE   = os.getenv("NADIR_CLASSES_FILE",   "classes_nadir.yaml")
OBLIQUE_CLASSES_FILE = os.getenv("OBLIQUE_CLASSES_FILE", "classes_oblique.yaml")

NADIR_CLASSES   = ["panneau_solaire"]
OBLIQUE_CLASSES = ["batiment_peint", "batiment_non_enduit", "batiment_enduit"]
ALL_CLASSES     = NADIR_CLASSES + OBLIQUE_CLASSES


# =============================================================================
# UTILITAIRES
# =============================================================================

def load_classes(yaml_path):
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f).get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    return classes


def _list_output_dirs(mode):
    candidates = []
    for base in [os.getenv("OUTPUT_DIR", "./output"), "./runs/detect/train"]:
        if not os.path.exists(base):
            continue
        if mode != "all":
            prefix = f"fasterrcnn_{mode}_"
            dirs = [d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)) and d.startswith(prefix)]
        else:
            dirs = [d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d))
                    and d.startswith("fasterrcnn_")
                    and not d.startswith("fasterrcnn_nadir_")
                    and not d.startswith("fasterrcnn_oblique_")]
        for d in sorted(dirs, reverse=True):
            candidates.append(os.path.join(base, d))
    return candidates


def find_best_model(mode):
    for train_dir in _list_output_dirs(mode):
        for fname in ["best_model.pth", "best.pth"]:
            candidate = os.path.join(train_dir, fname)
            if os.path.exists(candidate):
                return candidate
    return None


def find_test_info(mode):
    for train_dir in _list_output_dirs(mode):
        candidate = os.path.join(train_dir, "test_info.json")
        if os.path.exists(candidate):
            return candidate
    return None


# =============================================================================
# MODELE
# =============================================================================

def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path, device):
    print(f"   Modele : {model_path}")
    checkpoint  = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 5)
    classes     = checkpoint.get('classes', [])
    cat_mapping = checkpoint.get('cat_mapping', {})

    model = build_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    map50 = checkpoint.get('map50', 0)
    print(f"   Epoch {epoch} | mAP@50 val = {map50:.4f} | Classes : {classes}")
    return model, classes, cat_mapping


# =============================================================================
# DATASET
# =============================================================================

class TestDataset(Dataset):
    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping, image_size=640):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
        self.image_size  = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        image_tensor = TF.to_tensor(image)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            class_id = self.cat_mapping.get(ann['category_id'])
            if class_id is None:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x1 = max(0, x * scale_x);          y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x+w)*scale_x)
            y2 = min(self.image_size, (y+h)*scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        target = {
            'boxes':  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0,4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),  dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =============================================================================
# METRIQUES
# =============================================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0


class MetricsCalculator:
    """
    Calcule AP (aire sous courbe P/R) par classe et par seuil IoU.
    Stocke toutes les detections avec leurs scores pour un AP correct.
    Precision/Recall/F1 sont calcules au seuil de score fixe.
    """

    def __init__(self, class_names, iou_thresholds):
        self.class_names    = [c for c in class_names if c != '__background__']
        self.iou_thresholds = iou_thresholds
        # Per (class, iou_thresh) : liste de (score, is_tp)
        self.det_records = {
            name: {t: [] for t in iou_thresholds}
            for name in self.class_names
        }
        # Nombre total de GT par classe (denominateur du recall)
        self.n_gt = {name: 0 for name in self.class_names}

    def add_image(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        # Comptage des GT (une fois par image, independant du seuil IoU)
        for class_id, name in enumerate(self.class_names, start=1):
            self.n_gt[name] += int((gt_labels == class_id).sum())

        # Appariement predictions / GT a chaque seuil IoU
        for iou_thresh in self.iou_thresholds:
            for class_id, name in enumerate(self.class_names, start=1):
                p_mask = pred_labels == class_id
                g_mask = gt_labels   == class_id
                p_b = pred_boxes[p_mask]
                p_s = pred_scores[p_mask]
                g_b = gt_boxes[g_mask]

                if len(p_b) == 0:
                    continue

                if len(g_b) == 0:
                    for s in p_s:
                        self.det_records[name][iou_thresh].append((float(s), False))
                    continue

                iou_mat = np.array([[calculate_iou(p, g) for g in g_b] for p in p_b])
                matched_gt = set()
                for i in np.argsort(-p_s):
                    best_j = -1; best_iou = -1.0
                    for j in range(len(g_b)):
                        if j not in matched_gt and iou_mat[i, j] >= iou_thresh:
                            if iou_mat[i, j] > best_iou:
                                best_iou = iou_mat[i, j]; best_j = j
                    if best_j >= 0:
                        matched_gt.add(best_j)
                        self.det_records[name][iou_thresh].append((float(p_s[i]), True))
                    else:
                        self.det_records[name][iou_thresh].append((float(p_s[i]), False))

    def _compute_ap(self, records, n_gt):
        """AP via interpolation all-points (aire sous courbe P/R)."""
        if n_gt == 0 or not records:
            return 0.0
        records_sorted = sorted(records, key=lambda x: -x[0])
        tp_cum = 0; fp_cum = 0
        ap = 0.0; prev_r = 0.0
        for _, is_tp in records_sorted:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            p = tp_cum / (tp_cum + fp_cum)
            r = tp_cum / n_gt
            if r > prev_r:
                ap += (r - prev_r) * p
                prev_r = r
        return float(ap)

    def merge(self, other):
        """Fusionne les compteurs d'un autre MetricsCalculator dans celui-ci."""
        for name in other.class_names:
            if name not in self.class_names:
                self.class_names.append(name)
                self.n_gt[name] = 0
                self.det_records[name] = {t: [] for t in self.iou_thresholds}
            self.n_gt[name] += other.n_gt[name]
            for t in self.iou_thresholds:
                self.det_records[name][t].extend(other.det_records[name].get(t, []))

    def compute(self, score_threshold=0.5):
        """
        Retourne un dict avec :
          mAP50, mAP50_95, precision, recall, f1_score,
          per_class_AP50, per_class_precision, per_class_recall, per_class_f1
        """
        per_class_AP50      = {}
        per_class_ap50_95   = {}
        per_class_precision = {}
        per_class_recall    = {}
        per_class_f1        = {}

        for name in self.class_names:
            n_gt = self.n_gt[name]

            # AP par seuil IoU -> AP50 et AP50:95
            aps = [self._compute_ap(self.det_records[name][t], n_gt)
                   for t in self.iou_thresholds]
            per_class_AP50[name]    = aps[0]   # IoU = 0.5
            per_class_ap50_95[name] = float(np.mean(aps))

            # Precision / Recall / F1 au seuil de score fixe (IoU = 0.5)
            records_50 = self.det_records[name][0.5]
            tp = sum(1 for s, is_tp in records_50 if s >= score_threshold and is_tp)
            fp = sum(1 for s, is_tp in records_50 if s >= score_threshold and not is_tp)
            fn = max(0, n_gt - tp)

            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            per_class_precision[name] = p
            per_class_recall[name]    = r
            per_class_f1[name]        = f1

        # Metriques globales (moyenne macro sur les classes)
        mAP50    = float(np.mean(list(per_class_AP50.values())))    if per_class_AP50    else 0.0
        mAP50_95 = float(np.mean(list(per_class_ap50_95.values()))) if per_class_ap50_95 else 0.0
        precision = float(np.mean(list(per_class_precision.values()))) if per_class_precision else 0.0
        recall    = float(np.mean(list(per_class_recall.values())))    if per_class_recall    else 0.0
        f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'mAP50':               mAP50,
            'mAP50_95':            mAP50_95,
            'precision':           precision,
            'recall':              recall,
            'f1_score':            f1_score,
            'per_class_AP50':      per_class_AP50,
            'per_class_precision': per_class_precision,
            'per_class_recall':    per_class_recall,
            'per_class_f1':        per_class_f1,
        }


# =============================================================================
# EVALUATION D'UN MODELE
# =============================================================================

def run_evaluation(model, test_info_path, classes, device, score_threshold, image_size):
    """
    Lance l'evaluation sur le test set decrit par test_info.json.
    Retourne (MetricsCalculator, n_images, inference_ms_moyen).
    Toutes les predictions (sans filtrage par score) sont passees au calculateur
    pour un calcul correct de l'AP. Le seuil de score est applique dans compute().
    """
    with open(test_info_path, 'r') as f:
        test_info = json.load(f)

    images_dir       = test_info['images_dir']
    annotations_file = test_info['annotations_file']
    test_image_ids   = test_info['test_image_ids']
    cat_mapping      = {int(k): v for k, v in test_info['cat_mapping'].items()}

    dataset = TestDataset(images_dir, annotations_file, test_image_ids, cat_mapping, image_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    calc            = MetricsCalculator(classes, IOu_THRESHOLDS)
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="   Evaluation", leave=False):
            images_dev = [img.to(device) for img in images]

            # Chronometrage image par image avec synchronisation CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(images_dev)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            inference_times.append((t1 - t0) * 1000.0)  # ms

            for output, target in zip(outputs, targets):
                # Pas de filtrage par score ici : le calculateur stocke tout
                pred_boxes  = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_boxes    = target['boxes'].numpy()
                gt_labels   = target['labels'].numpy()
                calc.add_image(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

    inference_ms = float(np.mean(inference_times)) if inference_times else None
    return calc, len(test_image_ids), inference_ms


def evaluate_model(model_path, test_info_path, classes, mode, output_dir, device,
                   score_threshold, image_size):
    """
    Charge le modele, evalue sur le test set et retourne un dict de metriques
    conforme au format YOLO. Sauvegarde metrics_{mode}.json dans output_dir.
    """
    print(f"\n{'='*65}")
    print(f"   Evaluation : {mode.upper()}")
    print(f"{'='*65}")

    if not model_path or not os.path.exists(model_path):
        print(f"   Modele non trouve : {model_path}")
        print(f"   Conseil : lancez  python train.py --mode {mode}")
        return None

    if not test_info_path or not os.path.exists(test_info_path):
        print(f"   test_info.json non trouve : {test_info_path}")
        print(f"   Conseil : lancez  python train.py --mode {mode}")
        return None

    model, _, _ = load_model(model_path, device)

    # Nombre de parametres en millions
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    print("\n   Evaluation sur le test set...")
    calc, n_images, inference_ms = run_evaluation(
        model, test_info_path, classes, device, score_threshold, image_size
    )
    fps_gpu  = (1000.0 / inference_ms) if inference_ms else None
    computed = calc.compute(score_threshold=score_threshold)

    metrics = {
        "mode":                mode,
        "model":               model_path,
        "mAP50":               computed['mAP50'],
        "mAP50_95":            computed['mAP50_95'],
        "precision":           computed['precision'],
        "recall":              computed['recall'],
        "f1_score":            computed['f1_score'],
        "inference_ms":        inference_ms,
        "fps_gpu":             fps_gpu,
        "params_M":            params_M,
        "per_class_AP50":      computed['per_class_AP50'],
        "per_class_precision": computed['per_class_precision'],
        "per_class_recall":    computed['per_class_recall'],
        "per_class_f1":        computed['per_class_f1'],
        "evaluated_at":        datetime.now().isoformat(),
    }

    # --- Tableau global ---
    print(f"\n   {'Metrique':<32} {'Valeur':>10}")
    print(f"   {'-'*44}")
    print(f"   {'mAP@50':<32} {metrics['mAP50']:>10.4f}")
    print(f"   {'mAP@50:95':<32} {metrics['mAP50_95']:>10.4f}")
    print(f"   {'Precision':<32} {metrics['precision']:>10.4f}")
    print(f"   {'Recall':<32} {metrics['recall']:>10.4f}")
    print(f"   {'F1 Score':<32} {metrics['f1_score']:>10.4f}")
    if inference_ms is not None:
        print(f"   {'Vitesse Inference (ms) ↓':<32} {inference_ms:>10.2f}")
    if fps_gpu is not None:
        print(f"   {'FPS GPU':<32} {fps_gpu:>10.1f}")
    print(f"   {'Parametres (M)':<32} {params_M:>10.2f}")

    # --- Tableau par classe ---
    per_class_AP50      = computed['per_class_AP50']
    per_class_precision = computed['per_class_precision']
    per_class_recall    = computed['per_class_recall']
    per_class_f1        = computed['per_class_f1']

    if per_class_AP50:
        print(f"\n   {'Classe':<25} {'AP@50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  Statut")
        print(f"   {'-'*62}")
        for cls in sorted(per_class_AP50.keys()):
            ap  = per_class_AP50.get(cls, float('nan'))
            pc  = per_class_precision.get(cls, float('nan'))
            rc  = per_class_recall.get(cls, float('nan'))
            f1c = per_class_f1.get(cls, float('nan'))
            status = "OK" if ap >= 0.5 else ("~" if ap >= 0.3 else "!!")
            print(f"      {cls:<25} {ap:>7.4f} {pc:>7.4f} {rc:>7.4f} {f1c:>7.4f}  [{status}]")

    # --- Sauvegarde metrics_{mode}.json ---
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"metrics_{mode}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\n   Metriques sauvegardees : {metrics_path}")

    return metrics


# =============================================================================
# RAPPORT COMBINE
# =============================================================================

def print_combined_report(nadir_metrics, oblique_metrics, output_dir):
    """Affiche et sauvegarde le rapport combine (metrics_combined.json)."""

    print("\n" + "=" * 65)
    print("   RAPPORT COMBINE — Evaluation Duale FasterRCNN")
    print("=" * 65)

    # --- Tableau par classe ---
    print(f"\n   {'Classe':<25} {'Modele':<10} {'AP@50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  Statut")
    print(f"   {'-'*72}")

    all_results = {}
    for m, model_name in [(nadir_metrics, "nadir"), (oblique_metrics, "oblique")]:
        if m:
            for cls, ap in m.get("per_class_AP50", {}).items():
                all_results[cls] = {
                    "ap":    ap,
                    "prec":  m.get("per_class_precision", {}).get(cls, float('nan')),
                    "rec":   m.get("per_class_recall",    {}).get(cls, float('nan')),
                    "f1":    m.get("per_class_f1",        {}).get(cls, float('nan')),
                    "model": model_name,
                }

    for cls in ALL_CLASSES:
        if cls in all_results:
            d      = all_results[cls]
            status = "[OK]" if d["ap"] >= 0.5 else ("[~]" if d["ap"] >= 0.3 else "[!!]")
            print(f"   {cls:<25} {d['model']:<10} {d['ap']:>7.4f} {d['prec']:>7.4f}"
                  f" {d['rec']:>7.4f} {d['f1']:>7.4f}  {status}")
        else:
            print(f"   {cls:<25} {'N/A':<10} {'—':>7} {'—':>7} {'—':>7} {'—':>7}")

    print(f"   {'-'*55}")

    # --- Metriques globales par modele ---
    for label, m in [("NADIR", nadir_metrics), ("OBLIQUE", oblique_metrics)]:
        if m:
            print(f"\n   [{label}]")
            print(f"   mAP@50              : {m['mAP50']:.4f} ({m['mAP50'] * 100:.2f}%)")
            print(f"   mAP@50:95           : {m['mAP50_95']:.4f}")
            print(f"   Precision           : {m['precision']:.4f}")
            print(f"   Recall              : {m['recall']:.4f}")
            if m.get('f1_score') is not None:
                print(f"   F1 Score            : {m['f1_score']:.4f}")
            if m.get('inference_ms') is not None:
                print(f"   Vitesse Inference   : {m['inference_ms']:.2f} ms ↓")
            if m.get('fps_gpu') is not None:
                print(f"   FPS GPU             : {m['fps_gpu']:.1f}")
            if m.get('params_M') is not None:
                print(f"   Parametres          : {m['params_M']:.2f} M")

    # --- Moyennes globales combinees ---
    available = [m for m in [nadir_metrics, oblique_metrics] if m]

    def _mean(key):
        vals = [m[key] for m in available if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    mean_map    = _mean("mAP50")
    mean_map95  = _mean("mAP50_95")
    mean_prec   = _mean("precision")
    mean_rec    = _mean("recall")
    mean_f1     = _mean("f1_score")
    mean_inf_ms = _mean("inference_ms")
    mean_fps    = _mean("fps_gpu")
    mean_params = _mean("params_M")

    if mean_map is not None:
        print(f"\n   {'Metrique combinee':<32} {'Valeur':>10}")
        print(f"   {'-'*44}")
        print(f"   {'mAP@50':<32} {mean_map:>10.4f}  ({mean_map*100:.2f}%)")
        if mean_map95  is not None: print(f"   {'mAP@50:95':<32} {mean_map95:>10.4f}")
        if mean_prec   is not None: print(f"   {'Precision':<32} {mean_prec:>10.4f}")
        if mean_rec    is not None: print(f"   {'Recall':<32} {mean_rec:>10.4f}")
        if mean_f1     is not None: print(f"   {'F1 Score':<32} {mean_f1:>10.4f}")
        if mean_inf_ms is not None: print(f"   {'Vitesse Inference (ms) ↓':<32} {mean_inf_ms:>10.2f}")
        if mean_fps    is not None: print(f"   {'FPS GPU':<32} {mean_fps:>10.1f}")
        if mean_params is not None: print(f"   {'Parametres (M)':<32} {mean_params:>10.2f}")

    print("=" * 65)

    # --- Sauvegarde metrics_combined.json ---
    combined = {
        "timestamp": datetime.now().isoformat(),
        "nadir":     nadir_metrics,
        "oblique":   oblique_metrics,
        "combined": {
            "mAP50":        mean_map,
            "mAP50_95":     mean_map95,
            "precision":    mean_prec,
            "recall":       mean_rec,
            "f1_score":     mean_f1,
            "inference_ms": mean_inf_ms,
            "fps_gpu":      mean_fps,
            "params_M":     mean_params,
        },
        "per_class": {
            cls: all_results[cls] for cls in ALL_CLASSES if cls in all_results
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "metrics_combined.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, default=float)
    print(f"\n   Rapport combine : {report_path}")

    return combined


# =============================================================================
# GRAPHIQUES
# =============================================================================

def plot_dual_results(nadir_metrics, oblique_metrics, output_dir):
    """AP@50 par classe pour les deux modeles."""
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    for cls in ALL_CLASSES:
        for metrics, model_name, color in [
            (nadir_metrics,   "nadir",   "steelblue"),
            (oblique_metrics, "oblique", "darkorange"),
        ]:
            if metrics and cls in metrics.get("per_class_AP50", {}):
                entries.append({
                    "label": f"{cls}\n({model_name})",
                    "ap":    metrics["per_class_AP50"][cls],
                    "color": color,
                })

    if not entries:
        return

    labels = [e["label"] for e in entries]
    ap50s  = [e["ap"]    for e in entries]
    colors = [e["color"] for e in entries]

    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 1.5), 6))
    bars = ax.bar(range(len(labels)), ap50s, color=colors, edgecolor="white", linewidth=0.5)
    for bar, ap in zip(bars, ap50s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ap:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AP@50")
    ax.set_title("AP@50 par classe — Evaluation Duale FasterRCNN")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red",    linestyle="--", alpha=0.5, label="Seuil 0.5")
    ax.axhline(y=0.3, color="orange", linestyle=":",  alpha=0.4, label="Seuil 0.3")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_dual.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Graphique : {out_path}")


def plot_global_comparison(nadir_metrics, oblique_metrics, output_dir):
    """Metriques globales (mAP50, Precision, Recall) nadir vs oblique."""
    available = {
        k: v for k, v in [("nadir", nadir_metrics), ("oblique", oblique_metrics)]
        if v is not None
    }
    if not available:
        return

    metric_keys   = ["mAP50", "mAP50_95", "precision", "recall"]
    metric_labels = ["mAP@50", "mAP@50:95", "Precision", "Recall"]
    colors        = {"nadir": "#2196F3", "oblique": "#FF9800"}

    x     = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, metrics) in enumerate(available.items()):
        vals   = [metrics[k] for k in metric_keys]
        offset = (i - (len(available) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=f"Modele {name}",
                        color=colors.get(name, "gray"), alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Metriques globales — Nadir vs Oblique")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison_dual.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Comparaison : {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation duale FasterRCNN — nadir + oblique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nadir-model",   default=None, help="Modele nadir (.pth)")
    parser.add_argument("--oblique-model", default=None, help="Modele oblique (.pth)")
    parser.add_argument("--only", choices=["nadir", "oblique"], default=None,
                        help="Evaluer un seul modele (sans rapport combine)")
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_BASE, "dual"),
                        help="Dossier de sortie")
    parser.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD,
                        help="Seuil de score pour Precision/Recall/F1")
    parser.add_argument("--image-size", type=int,   default=IMAGE_SIZE,
                        help="Taille d'image pour le redimensionnement")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   Device : {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    nadir_classes   = load_classes(NADIR_CLASSES_FILE)
    oblique_classes = load_classes(OBLIQUE_CLASSES_FILE)

    nadir_model_path   = args.nadir_model   or find_best_model("nadir")
    oblique_model_path = args.oblique_model or find_best_model("oblique")

    print("=" * 65)
    print("   EVALUATION DUALE FasterRCNN — Toitures Cadastrales")
    print("=" * 65)

    nadir_metrics   = None
    oblique_metrics = None

    # ------------------------------------------------------------------ NADIR
    if args.only in (None, "nadir"):
        nadir_test_info = find_test_info("nadir")
        nadir_metrics = evaluate_model(
            nadir_model_path, nadir_test_info, nadir_classes, "nadir",
            args.output_dir, device, args.threshold, args.image_size,
        )

    # --------------------------------------------------------------- OBLIQUE
    if args.only in (None, "oblique"):
        oblique_test_info = find_test_info("oblique")
        oblique_metrics = evaluate_model(
            oblique_model_path, oblique_test_info, oblique_classes, "oblique",
            args.output_dir, device, args.threshold, args.image_size,
        )

    # --------------------------------------------------------------- COMBINE
    if nadir_metrics and oblique_metrics:
        print_combined_report(nadir_metrics, oblique_metrics, args.output_dir)
        plot_dual_results(nadir_metrics, oblique_metrics, args.output_dir)
        plot_global_comparison(nadir_metrics, oblique_metrics, args.output_dir)
        print(f"\n   Resultats sauvegardes : {args.output_dir}")

    elif nadir_metrics or oblique_metrics:
        plot_dual_results(nadir_metrics, oblique_metrics, args.output_dir)
        print(f"\n   Resultats sauvegardes : {args.output_dir}")

    else:
        print("\n   Aucune evaluation effectuee.")
        print("   Lancez d'abord :")
        print("      python train.py --mode nadir")
        print("      python train.py --mode oblique")


if __name__ == "__main__":
    main()
