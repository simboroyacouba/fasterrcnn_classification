"""
Evaluation Faster R-CNN unifie — un seul modele, toutes classes

Evalue le modele entraine par train_unified.py sur son test set.
Produit : metrics_unified.json, graphiques, rapport texte.

Usage :
  python evaluate_unified.py
  python evaluate_unified.py --model runs/detect/train/fasterrcnn_unified_.../best_model.pth
  python evaluate_unified.py --model best.pth --test-info test_info.json
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
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

IOU_THRESHOLDS  = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
IMAGE_SIZE      = int(os.getenv("IMAGE_SIZE",      "640"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
OUTPUT_BASE     = os.getenv("EVALUATION_DIR", "./evaluation_unified")


# =============================================================================
# DECOUVERTE AUTOMATIQUE DU MODELE
# =============================================================================

def find_unified_model_and_test_info(output_base=None):
    """Trouve (model_path, test_info_path) du modele unifie le plus recent."""

    if output_base is None:
        output_base = os.getenv("OUTPUT_DIR", "./runs/detect/train")

    # 1. model_info_unified.json
    info_path = os.path.join(output_base, "model_info_unified.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        model_path = info.get("best_model")
        test_info  = info.get("test_info")
        if model_path and os.path.exists(model_path):
            print(f"   model_info_unified.json -> {model_path}")
            return model_path, test_info

    # 2. Chercher le dossier fasterrcnn_unified_* le plus recent
    for base in [output_base, "./runs/detect/train"]:
        if not os.path.exists(base):
            continue
        dirs = sorted(
            [d for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d)) and d.startswith("fasterrcnn_unified_")],
            reverse=True,
        )
        for d in dirs:
            for fname in ["best_model.pth", "best.pth"]:
                candidate = os.path.join(base, d, fname)
                if os.path.exists(candidate):
                    test_info = os.path.join(base, d, "test_info.json")
                    print(f"   Trouve -> {candidate}")
                    return candidate, test_info if os.path.exists(test_info) else None

    raise FileNotFoundError(
        "\n[ERREUR] Modele unifie Faster R-CNN introuvable.\n"
        "Lancez d'abord : python train_unified.py\n"
        "Ou passez le chemin manuellement : --model chemin/vers/best_model.pth"
    )


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
    num_classes = checkpoint.get('num_classes', 6)
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
# DATASET TEST
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

        image  = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image  = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        image_tensor = TF.to_tensor(image)

        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
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
            x1 = max(0, x * scale_x)
            y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2]); labels.append(class_id)

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64),
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
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0


class MetricsCalculator:
    """AP (aire sous courbe P/R) par classe et par seuil IoU."""

    def __init__(self, class_names, iou_thresholds):
        self.class_names    = [c for c in class_names if c != '__background__']
        self.iou_thresholds = iou_thresholds
        self.det_records    = {
            name: {t: [] for t in iou_thresholds}
            for name in self.class_names
        }
        self.n_gt = {name: 0 for name in self.class_names}

    def add_image(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        for class_id, name in enumerate(self.class_names, start=1):
            self.n_gt[name] += int((gt_labels == class_id).sum())

        for iou_thresh in self.iou_thresholds:
            for class_id, name in enumerate(self.class_names, start=1):
                p_mask = pred_labels == class_id
                g_mask = gt_labels   == class_id
                p_b = pred_boxes[p_mask]; p_s = pred_scores[p_mask]; g_b = gt_boxes[g_mask]

                if len(p_b) == 0:
                    continue
                if len(g_b) == 0:
                    for s in p_s:
                        self.det_records[name][iou_thresh].append((float(s), False))
                    continue

                iou_mat    = np.array([[calculate_iou(p, g) for g in g_b] for p in p_b])
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
        if n_gt == 0 or not records:
            return 0.0
        records_sorted = sorted(records, key=lambda x: -x[0])
        tp_cum = 0; fp_cum = 0; ap = 0.0; prev_r = 0.0
        for _, is_tp in records_sorted:
            if is_tp: tp_cum += 1
            else:     fp_cum += 1
            p = tp_cum / (tp_cum + fp_cum)
            r = tp_cum / n_gt
            if r > prev_r:
                ap += (r - prev_r) * p; prev_r = r
        return float(ap)

    def compute(self, score_threshold=0.5):
        per_class_AP50      = {}
        per_class_ap50_95   = {}
        per_class_precision = {}
        per_class_recall    = {}
        per_class_f1        = {}

        for name in self.class_names:
            n_gt = self.n_gt[name]
            aps  = [self._compute_ap(self.det_records[name][t], n_gt) for t in self.iou_thresholds]
            per_class_AP50[name]    = aps[0]
            per_class_ap50_95[name] = float(np.mean(aps))

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
# EVALUATION
# =============================================================================

def run_evaluation(model, test_info_path, classes, device, score_threshold, image_size):
    with open(test_info_path, 'r') as f:
        test_info = json.load(f)

    images_dir       = test_info['images_dir']
    annotations_file = test_info['annotations_file']
    test_image_ids   = test_info['test_image_ids']
    cat_mapping      = {int(k): v for k, v in test_info['cat_mapping'].items()}

    dataset = TestDataset(images_dir, annotations_file, test_image_ids, cat_mapping, image_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    calc            = MetricsCalculator(classes, IOU_THRESHOLDS)
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="   Evaluation", leave=False):
            images_dev = [img.to(device) for img in images]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(images_dev)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_times.append((time.perf_counter() - t0) * 1000.0)

            for output, target in zip(outputs, targets):
                calc.add_image(
                    output['boxes'].cpu().numpy(),
                    output['labels'].cpu().numpy(),
                    output['scores'].cpu().numpy(),
                    target['boxes'].numpy(),
                    target['labels'].numpy(),
                )

    inference_ms = float(np.mean(inference_times)) if inference_times else None
    return calc, len(test_image_ids), inference_ms


def evaluate_unified(model_path, test_info_path, classes, output_dir, device,
                     score_threshold, image_size):

    print(f"\n{'='*65}")
    print(f"   Evaluation — Modele Unifie Faster R-CNN")
    print(f"{'='*65}")

    if not model_path or not os.path.exists(model_path):
        print(f"   Modele non trouve : {model_path}")
        return None
    if not test_info_path or not os.path.exists(test_info_path):
        print(f"   test_info.json non trouve : {test_info_path}")
        return None

    model, _, _ = load_model(model_path, device)
    params_M    = sum(p.numel() for p in model.parameters()) / 1e6

    print("\n   Evaluation sur le test set...")
    calc, n_images, inference_ms = run_evaluation(
        model, test_info_path, classes, device, score_threshold, image_size
    )
    fps_gpu  = (1000.0 / inference_ms) if inference_ms else None
    computed = calc.compute(score_threshold=score_threshold)

    metrics = {
        "model":                model_path,
        "mAP50":                computed['mAP50'],
        "mAP50_95":             computed['mAP50_95'],
        "precision":            computed['precision'],
        "recall":               computed['recall'],
        "f1_score":             computed['f1_score'],
        "inference_ms":         inference_ms,
        "fps_gpu":              fps_gpu,
        "params_M":             params_M,
        "per_class_AP50":       computed['per_class_AP50'],
        "per_class_precision":  computed['per_class_precision'],
        "per_class_recall":     computed['per_class_recall'],
        "per_class_f1":         computed['per_class_f1'],
        "evaluated_at":         datetime.now().isoformat(),
    }

    print(f"\n   {'Metrique':<32} {'Valeur':>10}")
    print(f"   {'-'*44}")
    print(f"   {'mAP@50':<32} {metrics['mAP50']:>10.4f}")
    print(f"   {'mAP@50:95':<32} {metrics['mAP50_95']:>10.4f}")
    print(f"   {'Precision':<32} {metrics['precision']:>10.4f}")
    print(f"   {'Recall':<32} {metrics['recall']:>10.4f}")
    print(f"   {'F1 Score':<32} {metrics['f1_score']:>10.4f}")
    if inference_ms is not None:
        print(f"   {'Vitesse Inference (ms)':<32} {inference_ms:>10.2f}")
    if fps_gpu is not None:
        print(f"   {'FPS GPU':<32} {fps_gpu:>10.1f}")
    print(f"   {'Parametres (M)':<32} {params_M:>10.2f}")

    per_class_AP50 = computed['per_class_AP50']
    if per_class_AP50:
        print(f"\n   {'Classe':<25} {'AP@50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  Statut")
        print(f"   {'-'*62}")
        for cls in sorted(per_class_AP50.keys()):
            ap  = per_class_AP50.get(cls, float('nan'))
            pc  = computed['per_class_precision'].get(cls, float('nan'))
            rc  = computed['per_class_recall'].get(cls, float('nan'))
            f1c = computed['per_class_f1'].get(cls, float('nan'))
            status = "OK" if ap >= 0.5 else ("~" if ap >= 0.3 else "!!")
            print(f"      {cls:<25} {ap:>7.4f} {pc:>7.4f} {rc:>7.4f} {f1c:>7.4f}  [{status}]")

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics_unified.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\n   Metriques : {metrics_path}")

    return metrics


# =============================================================================
# GRAPHIQUES
# =============================================================================

def plot_per_class_ap(metrics, output_dir):
    per_class = metrics.get("per_class_AP50", {})
    if not per_class:
        return

    classes = sorted(per_class.keys())
    ap50s   = [per_class[c] for c in classes]
    colors  = ["#4CAF50" if ap >= 0.5 else ("#FF9800" if ap >= 0.3 else "#F44336") for ap in ap50s]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.6), 5))
    bars = ax.bar(classes, ap50s, color=colors, edgecolor="white", linewidth=0.5)

    for bar, ap in zip(bars, ap50s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ap:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("AP@50")
    ax.set_title("AP@50 par classe — Modele Unifie Faster R-CNN")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red",    linestyle="--", alpha=0.5, label="Seuil 0.5")
    ax.axhline(y=0.3, color="orange", linestyle=":",  alpha=0.4, label="Seuil 0.3")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(output_dir, "metrics_unified.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"   Graphique : {out}")


def plot_global_metrics(metrics, output_dir):
    keys   = ["mAP50", "mAP50_95", "precision", "recall", "f1_score"]
    labels = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1"]
    vals   = [metrics.get(k, 0) for k in keys]
    colors = ["#2196F3", "#1565C0", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("Metriques globales — Modele Unifie Faster R-CNN")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="Seuil 0.5")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(output_dir, "metrics_unified_global.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"   Metriques globales : {out}")


def write_report(metrics, output_dir):
    report_path = os.path.join(output_dir, "evaluation_report_unified.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Faster R-CNN — Modele Unifie — Rapport d'Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date          : {metrics['evaluated_at']}\n")
        f.write(f"Modele        : {metrics['model']}\n\n")
        f.write(f"mAP@50        : {metrics['mAP50']:.4f}\n")
        f.write(f"mAP@50:95     : {metrics['mAP50_95']:.4f}\n")
        f.write(f"Precision     : {metrics['precision']:.4f}\n")
        f.write(f"Recall        : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score      : {metrics['f1_score']:.4f}\n")
        if metrics.get("inference_ms"):
            f.write(f"Inference (ms): {metrics['inference_ms']:.2f}\n")
        if metrics.get("fps_gpu"):
            f.write(f"FPS GPU       : {metrics['fps_gpu']:.1f}\n")
        if metrics.get("params_M"):
            f.write(f"Parametres    : {metrics['params_M']:.2f} M\n")
        f.write("\nAP@50 par classe :\n" + "-" * 40 + "\n")
        for cls in sorted(metrics.get("per_class_AP50", {})):
            ap  = metrics["per_class_AP50"][cls]
            f1c = metrics.get("per_class_f1", {}).get(cls, float("nan"))
            f.write(f"  {cls:<28} AP@50={ap:.4f}  F1={f1c:.4f}\n")
    print(f"   Rapport texte : {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Faster R-CNN unifie — toutes classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default=None, help="Chemin modele (.pth)")
    parser.add_argument("--test-info",  default=None, help="Chemin test_info.json")
    parser.add_argument("--output-dir", default=OUTPUT_BASE, help="Dossier de sortie")
    parser.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD, help="Seuil de score")
    parser.add_argument("--image-size", type=int,   default=IMAGE_SIZE,      help="Taille d'image")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = os.getenv("OUTPUT_DIR", "./runs/detect/train")

    print("=" * 65)
    print("   EVALUATION FASTER R-CNN — Modele Unifie")
    print("=" * 65)
    print(f"   Device : {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model:
        model_path    = args.model
        test_info_path = args.test_info
    else:
        try:
            model_path, test_info_path = find_unified_model_and_test_info(output_base)
        except FileNotFoundError as e:
            print(e); return
        if args.test_info:
            test_info_path = args.test_info

    if not test_info_path or not os.path.exists(test_info_path):
        print(f"   [ERREUR] test_info.json introuvable : {test_info_path}")
        return

    with open(test_info_path) as f:
        test_info = json.load(f)
    classes = test_info.get('classes', [])
    if not classes:
        print("   [ERREUR] Impossible de charger les classes depuis test_info.json")
        return

    metrics = evaluate_unified(
        model_path, test_info_path, classes, args.output_dir,
        device, args.threshold, args.image_size,
    )

    if metrics:
        plot_per_class_ap(metrics, args.output_dir)
        plot_global_metrics(metrics, args.output_dir)
        write_report(metrics, args.output_dir)

        print("\n" + "=" * 65)
        print("   RESUME")
        print("=" * 65)
        print(f"   mAP@50    : {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
        print(f"   mAP@50:95 : {metrics['mAP50_95']:.4f}")
        print(f"   Precision : {metrics['precision']:.4f}")
        print(f"   Recall    : {metrics['recall']:.4f}")
        print(f"   F1        : {metrics['f1_score']:.4f}")
        print(f"   Resultats : {args.output_dir}")
        print("=" * 65)


if __name__ == "__main__":
    main()
