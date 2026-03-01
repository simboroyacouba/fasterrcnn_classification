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
        return ['__background__', 'panneau_solaire', 'batiment_peint', 'batiment_non_enduit', 'batiment_enduit']
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f).get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    return classes


def load_colors(yaml_path="classes.yaml"):
    default = {
        'panneau_solaire':    (255, 0,   0),
        'batiment_peint':     (0,   255, 0),
        'batiment_non_enduit':(0,   0,   255),
        'batiment_enduit':    (255, 165, 0),
    }
    if not os.path.exists(yaml_path):
        return default
    with open(yaml_path, 'r', encoding='utf-8') as f:
        colors = yaml.safe_load(f).get('colors', {})
    return {k: tuple(v) for k, v in colors.items()} if colors else default


CLASSES_FILE = os.getenv("CLASSES_FILE", "classes.yaml")
CLASSES      = load_classes(CLASSES_FILE)
COLORS       = load_colors(CLASSES_FILE)


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


def find_best_model():
    """Trouver automatiquement le meilleur modèle"""
    path = os.getenv("MODEL_PATH", None)
    if path and os.path.exists(path):
        return path

    # Chercher dans runs/detect/train/ (dossier le plus récent en premier)
    runs_base = os.path.join("runs", "detect", "train")
    if os.path.exists(runs_base):
        subdirs = sorted(
            [d for d in os.listdir(runs_base) if os.path.isdir(os.path.join(runs_base, d))],
            reverse=True
        )
        for subdir in subdirs:
            for fname in ["best_model.pth", "best.pth"]:
                candidate = os.path.join(runs_base, subdir, fname)
                if os.path.exists(candidate):
                    print(f"📁 Modèle trouvé: {candidate}")
                    return candidate

    # Chercher dans output/
    for root, dirs, files in os.walk("output"):
        for fname in ["best_model.pth", "best.pth"]:
            if fname in files:
                found = os.path.join(root, fname)
                print(f"📁 Modèle trouvé: {found}")
                return found

    return None


def load_model(model_path, device):
    checkpoint  = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', len(CLASSES))
    classes     = checkpoint.get('classes', CLASSES)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
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

def main():
    parser = argparse.ArgumentParser(description="Inférence Faster R-CNN")
    parser.add_argument("--model",       default=None,    help="Chemin du modèle .pth (auto-détecté si absent)")
    parser.add_argument("--input",       default=os.getenv("DETECTION_INFERENCE_IMAGES_DIR", "../test"))
    parser.add_argument("--output",      default=os.getenv("PREDICTIONS_DIR", "./predictions"))
    parser.add_argument("--threshold",   type=float, default=float(os.getenv("SCORE_THRESHOLD", "0.5")))
    parser.add_argument("--image-size",  type=int,   default=int(os.getenv("IMAGE_SIZE", "640")))
    parser.add_argument("--no-display",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Trouver le modèle
    if args.model is None:
        args.model = find_best_model()
    if args.model is None or not os.path.exists(args.model):
        print(f"❌ Modèle non trouvé: {args.model}")
        return

    model, classes = load_model(args.model, device)
    print(f"🧠 Modèle: {args.model} | Classes: {classes}")

    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted([p for p in input_path.iterdir()
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}])
    else:
        images = [input_path]

    print(f"🖼️  {len(images)} image(s)\n")

    reports    = []
    start_total = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] 🔍 {img_path.name}")
        image, preds = predict(model, str(img_path), classes, device, args.threshold, args.image_size)
        out_img = os.path.join(args.output, f"{img_path.stem}_fasterrcnn.png")
        visualize(image, preds, out_img, show=not args.no_display)
        report = generate_report(preds, img_path.name, classes)
        reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")

    with open(os.path.join(args.output, 'reports.json'), 'w') as f:
        json.dump(reports, f, indent=2)

    if len(images) > 1:
        summary = generate_summary(reports, args.output, time.time() - start_total, classes)
        print(f"\n📊 Résumé: {summary['total_objects']} objets | {summary['avg_inference_ms']:.1f} ms/image")

    print(f"\n📁 Résultats: {args.output}")


if __name__ == "__main__":
    main()
