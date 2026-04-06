"""
inference_dual.py
Applique les modeles nadir et oblique selon le type d'image detecte.

Regles de routage :
  Production_*.png  ->  modele nadir uniquement  (panneau_solaire)
  Snapshot_*.jpg    ->  modele oblique uniquement (4 classes)
  Autres            ->  les deux modeles + fusion NMS

Usage :
    python inference_dual.py --input ./images/
    python inference_dual.py --input image.png --nadir-model ./output/fasterrcnn_nadir_.../best_model.pth
    python inference_dual.py --input image.jpg --force-mode oblique
"""

import os
import json
import yaml
import argparse
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms as torchvision_nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
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

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
IMAGE_SIZE      = int(os.getenv("IMAGE_SIZE", "640"))
NMS_THRESHOLD   = float(os.getenv("NMS_THRESHOLD", "0.5"))

COLORS = {
    "panneau_solaire":    (255, 0,   0),
    "batiment_peint":     (0,   255, 0),
    "batiment_non_enduit":(0,   0,   255),
    "batiment_enduit":    (255, 165, 0),
}

MODEL_LABEL = {
    "nadir":   "[N]",
    "oblique": "[O]",
    "both":    "[N+O]",
}


# =============================================================================
# DETECTION DU TYPE D'IMAGE
# =============================================================================

def detect_image_type(file_name):
    """
    Retourne 'nadir', 'oblique' ou 'unknown' selon le nom de fichier.
    Production_*.png -> nadir
    Snapshot_*.jpg   -> oblique
    """
    name = os.path.basename(file_name)
    if name.startswith("Production"):
        return "nadir"
    if name.startswith("Snapshot"):
        return "oblique"
    return "unknown"


# =============================================================================
# MODELE
# =============================================================================

def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path, device):
    print(f"   Chargement : {model_path}")
    checkpoint  = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 5)
    classes     = checkpoint.get("classes", [])

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    map50 = checkpoint.get("map50", 0)
    print(f"   Epoch {epoch} | mAP@50 (val) = {map50:.4f} | Classes : {classes}")
    return model, classes


def find_best_model(mode):
    """Chercher automatiquement le modele le plus recent pour le mode donne."""
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
        for subdir in sorted(dirs, reverse=True):
            for fname in ["best_model.pth", "best.pth"]:
                candidate = os.path.join(base, subdir, fname)
                if os.path.exists(candidate):
                    return candidate
    return None


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def predict(model, image_path, classes, device, threshold=0.5, image_size=640):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((image_size, image_size))
    tensor  = TF.to_tensor(resized).unsqueeze(0).to(device)

    scale_x = orig_w / image_size
    scale_y = orig_h / image_size

    t0      = time.time()
    outputs = model(tensor)
    inf_time = time.time() - t0

    output = outputs[0]
    keep   = output["scores"] >= threshold

    boxes  = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    # Rescaler vers dimensions originales
    if len(boxes) > 0:
        boxes[:, 0] *= scale_x; boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y; boxes[:, 3] *= scale_y

    class_names = [
        classes[int(l)] if int(l) < len(classes) else "unknown"
        for l in labels
    ]

    return image, {
        "boxes":          boxes,
        "labels":         labels,
        "scores":         scores,
        "class_names":    class_names,
        "inference_time": inf_time,
    }


# =============================================================================
# FUSION NMS (quand les deux modeles sont appliques)
# =============================================================================

def merge_predictions(preds_list, nms_iou=0.5):
    """
    Fusionne les predictions de plusieurs modeles.
    Applique NMS par classe pour eliminer les doublons.
    """
    all_boxes  = [p["boxes"]  for p in preds_list if len(p["boxes"]) > 0]
    all_labels = [p["labels"] for p in preds_list if len(p["boxes"]) > 0]
    all_scores = [p["scores"] for p in preds_list if len(p["boxes"]) > 0]
    all_names  = [p["class_names"] for p in preds_list if len(p["boxes"]) > 0]
    total_time = sum(p["inference_time"] for p in preds_list)

    if not all_boxes:
        return {
            "boxes": np.zeros((0, 4)), "labels": np.array([]),
            "scores": np.array([]), "class_names": [], "inference_time": total_time,
        }

    boxes  = np.concatenate(all_boxes,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    names  = [n for sublist in all_names for n in sublist]

    # NMS par classe
    kept = []
    for lbl in np.unique(labels):
        mask    = labels == lbl
        idxs    = np.where(mask)[0]
        b_t     = torch.tensor(boxes[mask],  dtype=torch.float32)
        s_t     = torch.tensor(scores[mask], dtype=torch.float32)
        keep_nms = torchvision_nms(b_t, s_t, nms_iou).numpy()
        kept.extend(idxs[keep_nms])

    kept = sorted(kept, key=lambda i: -scores[i])

    return {
        "boxes":          boxes[kept],
        "labels":         labels[kept],
        "scores":         scores[kept],
        "class_names":    [names[i] for i in kept],
        "inference_time": total_time,
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize(image, preds, model_label, output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(image)

    for box, class_name, score in zip(preds["boxes"], preds["class_names"], preds["scores"]):
        color = [c / 255 for c in COLORS.get(class_name, (128, 128, 128))]
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        ))
        axes[1].text(
            x1, y1 - 5,
            f"{class_name} {score:.2f}",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.8)
        )

    n   = len(preds["boxes"])
    ms  = preds["inference_time"] * 1000
    axes[1].set_title(f"Faster R-CNN {model_label} | {n} objets | {ms:.0f} ms")
    axes[1].axis("off")

    legend = [
        patches.Patch(facecolor=[c / 255 for c in col], label=name)
        for name, col in COLORS.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# RAPPORT
# =============================================================================

def generate_report(preds, image_name, model_used):
    by_class = {}
    for class_name in set(preds["class_names"]):
        by_class[class_name] = sum(1 for n in preds["class_names"] if n == class_name)

    return {
        "image":             image_name,
        "model_used":        model_used,
        "timestamp":         datetime.now().isoformat(),
        "inference_time_ms": preds["inference_time"] * 1000,
        "total_objects":     len(preds["boxes"]),
        "by_class":          by_class,
        "detections": [
            {
                "id":         i,
                "class":      class_name,
                "confidence": float(score),
                "bbox":       box.tolist(),
            }
            for i, (box, class_name, score) in enumerate(
                zip(preds["boxes"], preds["class_names"], preds["scores"])
            )
        ],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference duale Faster R-CNN nadir+oblique")
    parser.add_argument("--input",        default=os.getenv("DETECTION_INFERENCE_IMAGES_DIR", "./test"),
                        help="Image ou dossier d'images")
    parser.add_argument("--output",       default=os.getenv("PREDICTIONS_DIR", "./predictions_dual"))
    parser.add_argument("--nadir-model",  default=None, help="Chemin du modele nadir .pth (auto si absent)")
    parser.add_argument("--oblique-model",default=None, help="Chemin du modele oblique .pth (auto si absent)")
    parser.add_argument("--threshold",    type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--image-size",   type=int,   default=IMAGE_SIZE)
    parser.add_argument("--force-mode",   choices=["nadir", "oblique", "both"], default=None,
                        help="Forcer un mode pour toutes les images (ignore le nom de fichier)")
    parser.add_argument("--no-display",   action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device : {device}")

    # --- Charger les modeles disponibles ---
    nadir_model_path   = args.nadir_model   or find_best_model("nadir")
    oblique_model_path = args.oblique_model or find_best_model("oblique")

    nadir_model   = oblique_model   = None
    nadir_classes = oblique_classes = []

    if nadir_model_path and os.path.exists(nadir_model_path):
        print("\n[NADIR]")
        nadir_model, nadir_classes = load_model(nadir_model_path, device)
    else:
        print("   Modele nadir non trouve — images nadir ignorees")

    if oblique_model_path and os.path.exists(oblique_model_path):
        print("\n[OBLIQUE]")
        oblique_model, oblique_classes = load_model(oblique_model_path, device)
    else:
        print("   Modele oblique non trouve — images obliques ignorees")

    if nadir_model is None and oblique_model is None:
        print("\n   Aucun modele disponible. Lancez train.py --mode nadir et/ou --mode oblique.")
        return

    # --- Lister les images ---
    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted([
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ])
    else:
        images = [input_path]

    print(f"\n   {len(images)} image(s) a traiter\n")
    os.makedirs(args.output, exist_ok=True)

    reports = []

    for idx, img_path in enumerate(images, 1):
        # Determiner le mode
        if args.force_mode:
            mode = args.force_mode
        else:
            mode = detect_image_type(img_path.name)
            if mode == "unknown":
                mode = "both" if (nadir_model and oblique_model) else (
                    "nadir" if nadir_model else "oblique"
                )

        print(f"[{idx}/{len(images)}] {img_path.name}  ->  mode={mode}")

        # --- Inference ---
        if mode == "nadir":
            if nadir_model is None:
                print("   Modele nadir absent, image ignoree.")
                continue
            image, preds = predict(
                nadir_model, str(img_path), nadir_classes,
                device, args.threshold, args.image_size
            )
            model_label = MODEL_LABEL["nadir"]

        elif mode == "oblique":
            if oblique_model is None:
                print("   Modele oblique absent, image ignoree.")
                continue
            image, preds = predict(
                oblique_model, str(img_path), oblique_classes,
                device, args.threshold, args.image_size
            )
            model_label = MODEL_LABEL["oblique"]

        else:  # both
            preds_list = []
            if nadir_model:
                image, p_nadir = predict(
                    nadir_model, str(img_path), nadir_classes,
                    device, args.threshold, args.image_size
                )
                preds_list.append(p_nadir)
            if oblique_model:
                image, p_oblique = predict(
                    oblique_model, str(img_path), oblique_classes,
                    device, args.threshold, args.image_size
                )
                preds_list.append(p_oblique)
            preds = merge_predictions(preds_list, nms_iou=NMS_THRESHOLD)
            model_label = MODEL_LABEL["both"]

        # --- Visualisation ---
        out_img = os.path.join(args.output, f"{img_path.stem}_dual.png")
        visualize(image, preds, model_label, out_img, show=not args.no_display)

        # --- Rapport ---
        report = generate_report(preds, img_path.name, mode)
        reports.append(report)

        ms = preds["inference_time"] * 1000
        print(f"   {len(preds['boxes'])} objet(s) | {ms:.0f} ms | {report['by_class']}")

    # --- Sauvegarder les rapports ---
    with open(os.path.join(args.output, "reports_dual.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    if len(reports) > 1:
        total_objs = sum(r["total_objects"] for r in reports)
        avg_ms     = sum(r["inference_time_ms"] for r in reports) / len(reports)
        by_class   = {}
        for r in reports:
            for cls, cnt in r["by_class"].items():
                by_class[cls] = by_class.get(cls, 0) + cnt
        print(f"\n   Total : {total_objs} objets | {avg_ms:.0f} ms/image")
        print(f"   Par classe : {by_class}")

    print(f"\n   Resultats sauvegardes : {args.output}")


if __name__ == "__main__":
    main()
