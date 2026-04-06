"""
evaluate_dual.py
Evalue les modeles nadir et oblique separement puis genere un rapport combine.

Logique :
  - Modele NADIR   : evalue sur le test set nadir  (panneau_solaire)
  - Modele OBLIQUE : evalue sur le test set oblique (4 classes)
  - COMBINE        : fusionne les TP/FP/FN des deux evaluations par classe

Usage :
    python evaluate_dual.py
    python evaluate_dual.py --nadir-model path/to/nadir.pth --oblique-model path/to/oblique.pth
    python evaluate_dual.py --only nadir
    python evaluate_dual.py --only oblique
"""

import os
import json
import yaml
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


def find_best_model(mode):
    """Chercher dans output/fasterrcnn_{mode}_* le modele le plus recent."""
    output_base = "output"
    if not os.path.exists(output_base):
        return None
    prefix = f"fasterrcnn_{mode}_"
    subdirs = sorted(
        [d for d in os.listdir(output_base)
         if os.path.isdir(os.path.join(output_base, d)) and d.startswith(prefix)],
        reverse=True
    )
    for subdir in subdirs:
        for fname in ["best_model.pth", "best.pth"]:
            candidate = os.path.join(output_base, subdir, fname)
            if os.path.exists(candidate):
                return candidate
    return None


def find_test_info(mode):
    """Chercher test_info.json dans output/fasterrcnn_{mode}_*."""
    output_base = "output"
    if not os.path.exists(output_base):
        return None
    prefix = f"fasterrcnn_{mode}_"
    subdirs = sorted(
        [d for d in os.listdir(output_base)
         if os.path.isdir(os.path.join(output_base, d)) and d.startswith(prefix)],
        reverse=True
    )
    for subdir in subdirs:
        candidate = os.path.join(output_base, subdir, "test_info.json")
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
    """Calcule TP/FP/FN par classe et par seuil IoU."""

    def __init__(self, class_names, iou_thresholds):
        self.class_names    = [c for c in class_names if c != '__background__']
        self.iou_thresholds = iou_thresholds
        self.tp  = defaultdict(lambda: defaultdict(int))
        self.fp  = defaultdict(lambda: defaultdict(int))
        self.fn  = defaultdict(lambda: defaultdict(int))

    def add_image(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        for iou_thresh in self.iou_thresholds:
            for class_id, name in enumerate(self.class_names, start=1):
                p_mask = pred_labels == class_id
                g_mask = gt_labels   == class_id
                p_b = pred_boxes[p_mask]; p_s = pred_scores[p_mask]
                g_b = gt_boxes[g_mask]

                if len(g_b) == 0 and len(p_b) == 0:
                    continue
                if len(g_b) == 0:
                    self.fp[name][iou_thresh] += len(p_b); continue
                if len(p_b) == 0:
                    self.fn[name][iou_thresh] += len(g_b); continue

                iou_mat = np.array([[calculate_iou(p, g) for g in g_b] for p in p_b])
                matched = set()
                for i in np.argsort(-p_s):
                    best_j = -1
                    for j in range(len(g_b)):
                        if j not in matched and iou_mat[i, j] >= iou_thresh:
                            if best_j < 0 or iou_mat[i, j] > iou_mat[i, best_j]:
                                best_j = j
                    if best_j >= 0:
                        matched.add(best_j)
                        self.tp[name][iou_thresh] += 1
                    else:
                        self.fp[name][iou_thresh] += 1
                self.fn[name][iou_thresh] += len(g_b) - len(matched)

    def merge(self, other):
        """Fusionne les compteurs d'un autre MetricsCalculator dans celui-ci."""
        for name in other.class_names:
            if name not in self.class_names:
                self.class_names.append(name)
            for t in other.iou_thresholds:
                self.tp[name][t] += other.tp[name][t]
                self.fp[name][t] += other.fp[name][t]
                self.fn[name][t] += other.fn[name][t]

    def compute(self):
        results = {'per_class': {}, 'overall': {}}

        for name in self.class_names:
            results['per_class'][name] = {}
            for t in self.iou_thresholds:
                tp = self.tp[name][t]; fp = self.fp[name][t]; fn = self.fn[name][t]
                p  = tp / (tp + fp) if tp + fp > 0 else 0
                r  = tp / (tp + fn) if tp + fn > 0 else 0
                results['per_class'][name][f'iou_{t}'] = {
                    'TP': tp, 'FP': fp, 'FN': fn,
                    'Precision': p, 'Recall': r,
                    'F1': 2*p*r / (p+r) if p+r > 0 else 0
                }

        for t in self.iou_thresholds:
            tp = sum(self.tp[n][t] for n in self.class_names)
            fp = sum(self.fp[n][t] for n in self.class_names)
            fn = sum(self.fn[n][t] for n in self.class_names)
            p  = tp / (tp + fp) if tp + fp > 0 else 0
            r  = tp / (tp + fn) if tp + fn > 0 else 0
            results['overall'][f'iou_{t}'] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': p, 'Recall': r,
                'F1': 2*p*r / (p+r) if p+r > 0 else 0
            }

        results['mAP50']    = np.mean([
            results['per_class'][n]['iou_0.5']['Precision'] for n in self.class_names
        ])
        results['mAP50_95'] = np.mean([
            np.mean([results['per_class'][n][f'iou_{t}']['Precision'] for t in self.iou_thresholds])
            for n in self.class_names
        ])
        results['mAP_per_class'] = {
            name: {
                'AP50':    results['per_class'][name]['iou_0.5']['Precision'],
                'AP50_95': float(np.mean([
                    results['per_class'][name][f'iou_{t}']['Precision']
                    for t in self.iou_thresholds
                ]))
            }
            for name in self.class_names
        }
        return results


# =============================================================================
# EVALUATION D'UN MODELE
# =============================================================================

def run_evaluation(model, test_info_path, classes, device, score_threshold, image_size):
    """
    Lance l'evaluation sur le test set decrit par test_info.json.
    Retourne un MetricsCalculator rempli et le nombre d'images testees.
    """
    with open(test_info_path, 'r') as f:
        test_info = json.load(f)

    images_dir       = test_info['images_dir']
    annotations_file = test_info['annotations_file']
    test_image_ids   = test_info['test_image_ids']
    cat_mapping      = {int(k): v for k, v in test_info['cat_mapping'].items()}

    dataset = TestDataset(images_dir, annotations_file, test_image_ids, cat_mapping, image_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    calc = MetricsCalculator(classes, IOu_THRESHOLDS)

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="   Evaluation", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                keep = output['scores'].cpu() >= score_threshold
                pred_boxes  = output['boxes'].cpu().numpy()[keep.numpy()]
                pred_labels = output['labels'].cpu().numpy()[keep.numpy()]
                pred_scores = output['scores'].cpu().numpy()[keep.numpy()]
                gt_boxes    = target['boxes'].numpy()
                gt_labels   = target['labels'].numpy()
                calc.add_image(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

    return calc, len(test_image_ids)


# =============================================================================
# AFFICHAGE ET GRAPHIQUES
# =============================================================================

def print_results(label, results, n_images):
    w = 70
    print("\n" + "=" * w)
    print(f"   {label}")
    print("=" * w)
    print(f"   Images testees : {n_images}")
    print(f"   mAP@50         : {results['mAP50']:.4f}  ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95      : {results['mAP50_95']:.4f}")
    print(f"   Precision      : {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall         : {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score       : {results['overall']['iou_0.5']['F1']:.4f}")
    print("-" * w)
    print(f"   {'Classe':<28} {'P':>7} {'R':>7} {'F1':>7} {'AP@50':>8} {'AP@50:95':>10}")
    print("-" * w)
    for name in results['mAP_per_class']:
        m   = results['per_class'][name]['iou_0.5']
        ap  = results['mAP_per_class'][name]
        print(f"   {name:<28} {m['Precision']:>7.3f} {m['Recall']:>7.3f} {m['F1']:>7.3f}"
              f" {ap['AP50']:>8.3f} {ap['AP50_95']:>10.3f}")
    print("=" * w)


def plot_dual_results(nadir_res, oblique_res, combined_res, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # --- Graphique 1 : AP@50 par classe pour chaque modele ---
    all_classes = list(combined_res['mAP_per_class'].keys())
    x = np.arange(len(all_classes))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AP@50 par classe
    nadir_ap   = [nadir_res['mAP_per_class'].get(c, {}).get('AP50', 0)    for c in all_classes]
    oblique_ap = [oblique_res['mAP_per_class'].get(c, {}).get('AP50', 0)  for c in all_classes]
    combined_ap= [combined_res['mAP_per_class'][c]['AP50']                 for c in all_classes]

    axes[0].bar(x - w, nadir_ap,    w, label='Nadir',    color='steelblue',  alpha=0.8)
    axes[0].bar(x,     oblique_ap,  w, label='Oblique',  color='darkorange', alpha=0.8)
    axes[0].bar(x + w, combined_ap, w, label='Combine',  color='seagreen',   alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_classes, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('AP@50 par classe')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # P/R/F1 combine
    prec = [combined_res['per_class'][c]['iou_0.5']['Precision'] for c in all_classes]
    rec  = [combined_res['per_class'][c]['iou_0.5']['Recall']    for c in all_classes]
    f1   = [combined_res['per_class'][c]['iou_0.5']['F1']        for c in all_classes]

    axes[1].bar(x - w, prec, w, label='Precision', color='steelblue',  alpha=0.8)
    axes[1].bar(x,     rec,  w, label='Recall',    color='darkorange', alpha=0.8)
    axes[1].bar(x + w, f1,   w, label='F1',        color='seagreen',   alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_classes, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('P / R / F1 combine (IoU=0.5)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_dual.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Graphique : {os.path.join(output_dir, 'metrics_dual.png')}")

    # --- Graphique 2 : mAP@50 global comparatif ---
    models  = ['Nadir', 'Oblique', 'Combine']
    map50s  = [nadir_res['mAP50'], oblique_res['mAP50'], combined_res['mAP50']]
    colors  = ['steelblue', 'darkorange', 'seagreen']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, map50s, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, map50s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('mAP@50')
    ax.set_title('mAP@50 — Comparaison Nadir / Oblique / Combine')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map50_comparaison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_report(nadir_res, oblique_res, combined_res,
                n_nadir, n_oblique, model_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    full = {
        'timestamp':  datetime.now().isoformat(),
        'models':     model_paths,
        'nadir':      {'n_images': n_nadir,   **nadir_res},
        'oblique':    {'n_images': n_oblique, **oblique_res},
        'combined':   {'n_images': n_nadir + n_oblique, **combined_res},
    }
    json_path = os.path.join(output_dir, 'metrics_dual.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2, default=float)

    txt_path = os.path.join(output_dir, 'report_dual.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"EVALUATION DUALE Faster R-CNN - {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Modele nadir   : {model_paths.get('nadir', 'N/A')}\n")
        f.write(f"Modele oblique : {model_paths.get('oblique', 'N/A')}\n\n")

        for label, res, n in [
            ("NADIR",    nadir_res,    n_nadir),
            ("OBLIQUE",  oblique_res,  n_oblique),
            ("COMBINE",  combined_res, n_nadir + n_oblique),
        ]:
            f.write(f"\n--- {label} ({n} images) ---\n")
            f.write(f"mAP@50    : {res['mAP50']:.4f}\n")
            f.write(f"mAP@50:95 : {res['mAP50_95']:.4f}\n")
            f.write(f"Precision : {res['overall']['iou_0.5']['Precision']:.4f}\n")
            f.write(f"Recall    : {res['overall']['iou_0.5']['Recall']:.4f}\n")
            f.write(f"F1        : {res['overall']['iou_0.5']['F1']:.4f}\n")
            f.write("Par classe :\n")
            for name in res['mAP_per_class']:
                m  = res['per_class'][name]['iou_0.5']
                ap = res['mAP_per_class'][name]
                f.write(f"  {name:<28} P={m['Precision']:.3f} R={m['Recall']:.3f}"
                        f" F1={m['F1']:.3f} AP50={ap['AP50']:.3f}\n")

    print(f"   JSON   : {json_path}")
    print(f"   Rapport: {txt_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluation duale nadir + oblique")
    parser.add_argument("--nadir-model",   default=None)
    parser.add_argument("--oblique-model", default=None)
    parser.add_argument("--only", choices=["nadir", "oblique"], default=None,
                        help="Evaluer un seul modele (sans rapport combine)")
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_BASE, "dual"))
    parser.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--image-size", type=int,   default=IMAGE_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   Device : {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    nadir_classes   = load_classes(NADIR_CLASSES_FILE)
    oblique_classes = load_classes(OBLIQUE_CLASSES_FILE)

    nadir_model_path   = args.nadir_model   or find_best_model("nadir")
    oblique_model_path = args.oblique_model or find_best_model("oblique")

    # ------------------------------------------------------------------ NADIR
    nadir_res = nadir_calc = None
    n_nadir   = 0

    if args.only in (None, "nadir"):
        print("\n" + "=" * 70)
        print("   EVALUATION NADIR")
        print("=" * 70)
        if nadir_model_path and os.path.exists(nadir_model_path):
            nadir_model, _, _ = load_model(nadir_model_path, device)
            nadir_test_info   = find_test_info("nadir")
            if nadir_test_info:
                nadir_calc, n_nadir = run_evaluation(
                    nadir_model, nadir_test_info, nadir_classes,
                    device, args.threshold, args.image_size
                )
                nadir_res = nadir_calc.compute()
                print_results("NADIR - TEST SET", nadir_res, n_nadir)
            else:
                print("   test_info.json nadir introuvable. Lancez : python train.py --mode nadir")
        else:
            print("   Modele nadir introuvable. Lancez : python train.py --mode nadir")

    # --------------------------------------------------------------- OBLIQUE
    oblique_res = oblique_calc = None
    n_oblique   = 0

    if args.only in (None, "oblique"):
        print("\n" + "=" * 70)
        print("   EVALUATION OBLIQUE")
        print("=" * 70)
        if oblique_model_path and os.path.exists(oblique_model_path):
            oblique_model, _, _ = load_model(oblique_model_path, device)
            oblique_test_info   = find_test_info("oblique")
            if oblique_test_info:
                oblique_calc, n_oblique = run_evaluation(
                    oblique_model, oblique_test_info, oblique_classes,
                    device, args.threshold, args.image_size
                )
                oblique_res = oblique_calc.compute()
                print_results("OBLIQUE - TEST SET", oblique_res, n_oblique)
            else:
                print("   test_info.json oblique introuvable. Lancez : python train.py --mode oblique")
        else:
            print("   Modele oblique introuvable. Lancez : python train.py --mode oblique")

    # --------------------------------------------------------------- COMBINE
    if nadir_calc is not None and oblique_calc is not None:
        print("\n" + "=" * 70)
        print("   RAPPORT COMBINE")
        print("=" * 70)

        # Toutes les classes uniques dans l'ordre : nadir d'abord, puis oblique
        all_class_names = list(nadir_classes)
        for c in oblique_classes:
            if c not in all_class_names:
                all_class_names.append(c)

        combined_calc = MetricsCalculator(all_class_names, IOu_THRESHOLDS)
        combined_calc.merge(nadir_calc)
        combined_calc.merge(oblique_calc)
        combined_res = combined_calc.compute()

        print_results("COMBINE (nadir + oblique)", combined_res, n_nadir + n_oblique)

        plot_dual_results(nadir_res, oblique_res, combined_res, args.output_dir)
        save_report(
            nadir_res, oblique_res, combined_res,
            n_nadir, n_oblique,
            {'nadir': nadir_model_path, 'oblique': oblique_model_path},
            args.output_dir
        )
        print(f"\n   Resultats sauvegardes : {args.output_dir}")

    elif nadir_res or oblique_res:
        # Un seul modele : sauvegarder quand meme
        res    = nadir_res   or oblique_res
        calc   = nadir_calc  or oblique_calc
        n      = n_nadir     or n_oblique
        mode   = "nadir" if nadir_res else "oblique"
        out    = os.path.join(args.output_dir, mode)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f'metrics_{mode}.json'), 'w') as f:
            json.dump({'n_images': n, **res}, f, indent=2, default=float)
        print(f"\n   Resultats sauvegardes : {out}")

    else:
        print("\n   Aucune evaluation effectuee.")
        print("   Lancez d'abord : python train.py --mode nadir && python train.py --mode oblique")


if __name__ == "__main__":
    main()
