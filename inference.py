"""
Inférence Faster R-CNN - Détection des toitures cadastrales
Configuration: .env + classes.yaml
"""

import os
import json
import yaml
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import time
import argparse
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

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        return ['__background__', 'panneau_solaire', 'menuiserie_metallique', 'menuiserie_aluminium', 'cloture_enduit', 'cloture_non_enduit', 'cloture_peinte']
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f).get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    return classes


def load_colors(yaml_path="classes.yaml"):
    default = {
        'panneau_solaire':       (255, 0,   0),
        'menuiserie_metallique': (128, 0,   128),
        'menuiserie_aluminium':  (0,   200, 200),
        'cloture_enduit':        (255, 100, 0),
        'cloture_non_enduit':    (150, 75,  0),
        'cloture_peinte':        (255, 0,   150),
    }
    if not os.path.exists(yaml_path):
        return default
    with open(yaml_path, 'r', encoding='utf-8') as f:
        colors = yaml.safe_load(f).get('colors', {})
    return {k: tuple(v) for k, v in colors.items()} if colors else default


CLASSES_FILE = os.getenv("CLASSES_FILE", "classes.yaml")
CLASSES      = load_classes(CLASSES_FILE)
COLORS       = load_colors(CLASSES_FILE)


NADIR_PREFIX   = 'Production_'
OBLIQUE_PREFIX = 'Snapshot_'


def detect_image_mode(filename):
    """Retourne 'nadir', 'oblique' ou 'both' selon le prefixe du nom de fichier."""
    name = Path(filename).name
    if name.startswith(NADIR_PREFIX):   return 'nadir'
    if name.startswith(OBLIQUE_PREFIX): return 'oblique'
    return 'both'


def format_time(seconds):
    return f"{seconds*1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


# =============================================================================
# MODÈLE
# =============================================================================

def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _list_output_dirs(mode):
    """Retourne les repertoires d'entrainement tries du plus recent, filtres par mode."""
    candidates = []
    base_output = os.getenv("OUTPUT_DIR", "./output")
    runs_base   = os.getenv("RUNS_DIR",   "./runs/detect/train")

    if mode in ("nadir", "oblique"):
        prefix = f"fasterrcnn_{mode}_"
        for base in (os.path.join(base_output, mode), os.path.join(runs_base, mode)):
            if os.path.isdir(base):
                dirs = sorted(
                    [d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d)) and d.startswith(prefix)],
                    reverse=True,
                )
                for d in dirs:
                    candidates.append(os.path.join(base, d))
    else:
        for base in (base_output, runs_base):
            if os.path.isdir(base):
                dirs = sorted(
                    [d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d))
                     and d.startswith("fasterrcnn_")
                     and not d.startswith("fasterrcnn_nadir_")
                     and not d.startswith("fasterrcnn_oblique_")],
                    reverse=True,
                )
                for d in dirs:
                    candidates.append(os.path.join(base, d))

    return candidates


def find_best_model(mode="all"):
    """Trouve le meilleur modele disponible."""
    path = os.getenv("MODEL_PATH", None)
    if path and os.path.exists(path):
        return path

    subs = ("nadir", "oblique") if mode == "all" else (mode,)

    # Mode specifique ou unified
    for train_dir in _list_output_dirs(mode):
        for fname in ("best_model.pth", "weights/best.pth", "best.pth"):
            candidate = os.path.join(train_dir, fname)
            if os.path.exists(candidate):
                print(f"   Modele trouve: {candidate}")
                return candidate

    # Fallback dual si mode == all
    if mode == "all":
        found = {}
        for sub in ("nadir", "oblique"):
            for train_dir in _list_output_dirs(sub):
                for fname in ("best_model.pth", "weights/best.pth", "best.pth"):
                    candidate = os.path.join(train_dir, fname)
                    if os.path.exists(candidate) and sub not in found:
                        found[sub] = candidate
        if found:
            print(f"   Aucun modele unifie trouve. Modeles partiels detectes: {list(found.keys())}")
            print(f"   Utilisez --mode nadir ou --mode oblique pour choisir le bon modele.")
            # Preferer oblique (plus de classes) si disponible
            for sub in ("oblique", "nadir"):
                if sub in found:
                    print(f"   Selection automatique: {sub} → {found[sub]}")
                    return found[sub]

    return None


def find_both_models():
    """Trouve les modeles nadir et oblique disponibles (pour --mode all dual)."""
    found = {}
    for sub in ("nadir", "oblique"):
        for train_dir in _list_output_dirs(sub):
            for fname in ("best_model.pth", "weights/best.pth", "best.pth"):
                candidate = os.path.join(train_dir, fname)
                if os.path.exists(candidate):
                    found[sub] = candidate
                    break
            if sub in found:
                break
    return found


def load_model(model_path, device):
    checkpoint  = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', len(CLASSES))
    classes     = checkpoint.get('classes', CLASSES)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    print(f"   Classes ({len(classes)}): {classes}")
    return model, classes


# =============================================================================
# INFÉRENCE
# =============================================================================

@torch.no_grad()
def predict(model, image_path, classes, device, threshold=0.5, image_size=640):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((image_size, image_size))
    tensor  = TF.to_tensor(resized).unsqueeze(0).to(device)

    scale_x = orig_w / image_size
    scale_y = orig_h / image_size

    start   = time.time()
    outputs = model(tensor)
    inf_time = time.time() - start

    output = outputs[0]
    keep   = output['scores'] >= threshold

    boxes  = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()

    # Rescaler les boxes vers les dimensions originales
    boxes[:, 0] *= scale_x; boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y; boxes[:, 3] *= scale_y

    class_names = [classes[int(l)] if int(l) < len(classes) else 'unknown' for l in labels]

    preds = {
        'boxes':          boxes,
        'labels':         labels,
        'scores':         scores,
        'class_names':    class_names,
        'inference_time': inf_time,
    }
    return image, preds


def visualize(image, preds, output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(image)

    for box, class_name, score in zip(preds['boxes'], preds['class_names'], preds['scores']):
        color = [c / 255 for c in COLORS.get(class_name, (128, 128, 128))]
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        ))
        axes[1].text(x1, y1 - 5, f"{class_name}\n{score:.2f}", fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    axes[1].set_title(f"Faster R-CNN: {len(preds['boxes'])} objets | ⏱️ {format_time(preds['inference_time'])}")
    axes[1].axis('off')

    legend = [patches.Patch(facecolor=[c / 255 for c in col], label=name) for name, col in COLORS.items()]
    fig.legend(handles=legend, loc='lower center', ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def generate_report(preds, image_name, classes):
    class_names_no_bg = [c for c in classes if c != '__background__']
    report = {
        'image':            image_name,
        'model':            'Faster R-CNN',
        'timestamp':        datetime.now().isoformat(),
        'inference_time_ms': preds['inference_time'] * 1000,
        'total_objects':    len(preds['boxes']),
        'by_class':         {c: {'count': 0} for c in class_names_no_bg},
        'detections':       [],
    }
    for i, (box, class_name, score) in enumerate(zip(preds['boxes'], preds['class_names'], preds['scores'])):
        if class_name in report['by_class']:
            report['by_class'][class_name]['count'] += 1
        report['detections'].append({
            'id': i, 'class': class_name,
            'confidence': float(score),
            'bbox': box.tolist()
        })
    return report


def generate_summary(reports, output_dir, total_time, classes):
    class_names_no_bg = [c for c in classes if c != '__background__']
    summary = {
        'model':            'Faster R-CNN',
        'timestamp':        datetime.now().isoformat(),
        'total_images':     len(reports),
        'total_time_s':     total_time,
        'avg_inference_ms': sum(r['inference_time_ms'] for r in reports) / len(reports) if reports else 0,
        'total_objects':    sum(r['total_objects'] for r in reports),
        'by_class':         {c: sum(r['by_class'].get(c, {}).get('count', 0) for r in reports) for c in class_names_no_bg},
        'per_image':        [{'image': r['image'], 'objects': r['total_objects'], 'time_ms': r['inference_time_ms']} for r in reports],
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"RÉSUMÉ Faster R-CNN\n{'='*50}\n")
        f.write(f"Images: {summary['total_images']} | Temps: {total_time:.1f}s\n")
        f.write(f"Temps moyen: {summary['avg_inference_ms']:.1f} ms | Objets: {summary['total_objects']}\n")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def _auto_find_input():
    for candidate in (
        os.getenv("DETECTION_INFERENCE_IMAGES_DIR", ""),
        "./test_images",
        "./images",
        "../test",
    ):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Inference Faster R-CNN")
    parser.add_argument("--model",       default=None)
    parser.add_argument("--mode",        choices=["nadir", "oblique", "all"], default="all",
                        help="Sous-mode pour auto-detection du modele")
    parser.add_argument("--input",       default=None,
                        help="Dossier ou image (defaut: auto-detection)")
    parser.add_argument("--output",      default=os.getenv("PREDICTIONS_DIR", "./predictions"))
    parser.add_argument("--threshold",   type=float, default=float(os.getenv("SCORE_THRESHOLD", "0.5")))
    parser.add_argument("--image-size",  type=int,   default=int(os.getenv("IMAGE_SIZE", "640")))
    parser.add_argument("--display",     action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    if args.input is None:
        args.input = _auto_find_input()
    if args.input is None:
        print("   Aucun dossier d'images trouve.")
        print("   Utilisez : python inference.py --input /chemin/vers/images")
        print("   Ou definir DETECTION_INFERENCE_IMAGES_DIR dans .env")
        return

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir()
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
    else:
        images = [input_path]

    os.makedirs(args.output, exist_ok=True)

    # ── Chargement du ou des modeles ─────────────────────────────────────────
    # Mode all sans modele explicite → routing dual nadir/oblique
    dual_mode = False
    models_map   = {}   # {'nadir': model, 'oblique': model}
    classes_map  = {}   # {'nadir': classes, 'oblique': classes}

    if args.model is not None:
        if not os.path.exists(args.model):
            print(f"   Modele introuvable : {args.model}"); return
        model, classes = load_model(args.model, device)
        print(f"   Modele: {args.model} | Classes: {classes}")
    elif args.mode == "all":
        both = find_both_models()
        if not both:
            # Tentative modele unifie
            unified = find_best_model("all")
            if unified:
                model, classes = load_model(unified, device)
                print(f"   Modele unifie: {unified} | Classes: {classes}")
            else:
                print("   Modele non trouve. Lancez : python train.py --mode dual"); return
        else:
            dual_mode = True
            for sub, path in both.items():
                m, cls = load_model(path, device)
                models_map[sub]  = m
                classes_map[sub] = cls
                print(f"   {sub}: {path}")
            if len(both) == 1:
                print(f"   ⚠  Seul le modele {list(both.keys())[0]} disponible.")
    else:
        path = find_best_model(args.mode)
        if path is None or not os.path.exists(str(path or "")):
            print("   Modele non trouve. Lancez : python train.py --mode simple|attention|optimize|dual")
            return
        model, classes = load_model(path, device)
        print(f"   Modele: {path} | Classes: {classes}")

    print(f"🖼️  {len(images)} image(s)\n")

    reports     = []
    start_total = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] 🔍 {img_path.name}")

        if dual_mode:
            img_mode = detect_image_mode(img_path.name)
            if img_mode == 'nadir' and 'nadir' in models_map:
                cur_model, cur_classes = models_map['nadir'],   classes_map['nadir']
            elif img_mode == 'oblique' and 'oblique' in models_map:
                cur_model, cur_classes = models_map['oblique'], classes_map['oblique']
            elif models_map:
                # Prefixe inconnu : prendre le premier disponible
                key = list(models_map.keys())[0]
                cur_model, cur_classes = models_map[key], classes_map[key]
            else:
                print(f"   ⚠  Aucun modele disponible pour {img_path.name}"); continue
        else:
            cur_model, cur_classes = model, classes

        image, preds = predict(cur_model, str(img_path), cur_classes, device,
                               args.threshold, args.image_size)
        out_img = os.path.join(args.output, f"{img_path.stem}_fasterrcnn.png")
        visualize(image, preds, out_img, show=args.display)
        report = generate_report(preds, img_path.name, cur_classes)
        reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")

    with open(os.path.join(args.output, 'reports.json'), 'w') as f:
        json.dump(reports, f, indent=2)

    if len(images) > 1:
        # Pour le résumé, utiliser classes unifiées ou celles du premier modele
        summary_classes = (list(classes_map.values())[0] if dual_mode else classes)
        summary = generate_summary(reports, args.output, time.time() - start_total, summary_classes)
        print(f"\n📊 Résumé: {summary['total_objects']} objets | {summary['avg_inference_ms']:.1f} ms/image")

    print(f"\n📁 Résultats: {args.output}")


if __name__ == "__main__":
    main()