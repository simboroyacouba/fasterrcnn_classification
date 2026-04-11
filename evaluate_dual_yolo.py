"""
Evaluation duale YOLO — nadir et oblique separement puis resultats combines

Evaluation :
  - Modele nadir   : evalue sur le test set nadir   (panneau_solaire)
  - Modele oblique : evalue sur le test set oblique (batiments)
  - Rapport combine : AP@50 par classe + metriques globales

Usage :
  python evaluate_dual.py
  python evaluate_dual.py --nadir-model  output/nadir/.../best_model.pt \\
                           --oblique-model output/oblique/.../best_model.pt
  python evaluate_dual.py --skip-oblique
"""

import os
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONSTANTES
# =============================================================================

NADIR_CLASSES   = ["panneau_solaire"]
OBLIQUE_CLASSES = ["batiment_peint", "batiment_non_enduit", "batiment_enduit"]
ALL_CLASSES     = NADIR_CLASSES + OBLIQUE_CLASSES

COLORS_MAP = {
    "nadir":   "#2196F3",   # Bleu
    "oblique": "#FF9800",   # Orange
}


# =============================================================================
# DECOUVERTE AUTOMATIQUE
# =============================================================================

def find_model_and_data(mode, output_base=None):
    """Trouver le chemin du modele et du dataset.yaml pour un mode donne."""

    script_dir = Path(__file__).parent
    if output_base is None:
        output_base = str(script_dir / "output")

    model_path = None
    data_yaml  = None

    # 1. model_info_{mode}.json genere par train.py
    info_path = os.path.join(output_base, f"model_info_{mode}.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        candidate  = info.get("best_model")
        data_yaml  = info.get("dataset_yaml")
        if candidate and os.path.exists(candidate):
            model_path = candidate
            print(f"   [{mode}] model_info.json -> {model_path}")

    # 2. Parcourir output/{mode}/
    if not model_path:
        base_dir = os.path.join(output_base, mode)
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if "weights" in dirs:
                    pt = os.path.join(root, "weights", "best.pt")
                    if os.path.exists(pt):
                        model_path = pt
                        print(f"   [{mode}] scan -> {model_path}")
                        break

    # 3. Parcourir runs/detect/ puis data/*/runs/detect/ (chemins absolus)
    if not model_path:
        script_dir = Path(__file__).parent
        for glob_pattern in ["runs/detect/*/weights/best.pt",
                             "data/*/runs/detect/*/weights/best.pt"]:
            candidates = sorted(script_dir.glob(glob_pattern))
            if candidates:
                model_path = str(candidates[-1])  # train la plus recente
                print(f"   [{mode}] runs -> {model_path}")
                break

    # 4. dataset.yaml — chercher dans output/{mode}/ puis output/ puis data/*/output/
    if not data_yaml:
        yaml_candidate = os.path.join(output_base, mode, "dataset", "dataset.yaml")
        if os.path.exists(yaml_candidate):
            data_yaml = yaml_candidate

    if not data_yaml:
        script_dir = Path(__file__).parent
        for glob_pattern in ["output/dataset/dataset.yaml",
                             "data/*/output/dataset/dataset.yaml"]:
            candidates = sorted(script_dir.glob(glob_pattern))
            if candidates:
                data_yaml = str(candidates[-1])
                break

    return model_path, data_yaml


# =============================================================================
# BENCHMARK VITESSE
# =============================================================================

def benchmark_inference_speed(model, data_yaml, n_warmup=5, n_runs=30):
    """
    Mesure la vitesse d'inference image par image.
    Utilise torch.cuda.synchronize() si GPU disponible,
    coherent avec la methode utilisee pour Faster R-CNN.
    Retourne la moyenne en ms par image.
    """
    # Recuperer les images du test set depuis le yaml
    try:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        data_root = Path(data_yaml).parent
        test_rel  = cfg.get("test", "images/test")
        test_dir  = (data_root / test_rel) if not Path(test_rel).is_absolute() else Path(test_rel)
        exts      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images    = [p for p in sorted(test_dir.rglob("*")) if p.suffix.lower() in exts]
    except Exception:
        images = []

    if not images:
        return None

    # Detecter le device du modele
    try:
        device    = next(model.model.parameters()).device
        use_cuda  = device.type == "cuda"
    except Exception:
        use_cuda  = torch.cuda.is_available()

    sample = images[: n_warmup + n_runs]

    # Chauffe
    for img in sample[:n_warmup]:
        model.predict(str(img), verbose=False)

    # Chronometrage
    times = []
    for img in sample[n_warmup:]:
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(str(img), verbose=False)
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)) if times else None


# =============================================================================
# EVALUATION D'UN MODELE
# =============================================================================

def evaluate_model(model_path, data_yaml, mode, output_dir):
    """
    Evaluer un modele YOLO sur son test set.
    Retourne un dict de metriques ou None si echec.
    """

    print(f"\n{'='*65}")
    print(f"   Evaluation : {mode.upper()}")
    print(f"{'='*65}")

    if not model_path or not os.path.exists(model_path):
        print(f"   Modele non trouve : {model_path}")
        print(f"   Conseil : lancez  python train.py --mode {mode}")
        return None

    if not data_yaml or not os.path.exists(data_yaml):
        print(f"   dataset.yaml non trouve : {data_yaml}")
        print(f"   Conseil : lancez  python train.py --mode {mode}")
        return None

    print(f"   Modele  : {model_path}")
    print(f"   Dataset : {data_yaml}")

    model = YOLO(model_path)

    # Paramètres du modèle (en millions) — cohérent avec Faster R-CNN
    try:
        n_params_M = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        n_params_M = None

    # Evaluation sur le test set
    print("\n   Evaluation sur le test set...")
    results = model.val(data=data_yaml, split="test", verbose=False)

    box   = results.box
    names = results.names

    # Vitesse d'inférence image par image avec cuda.synchronize() — cohérent avec Faster R-CNN
    print("   Benchmark vitesse inference...")
    inference_ms = benchmark_inference_speed(model, data_yaml)
    fps_gpu      = (1000.0 / inference_ms) if inference_ms else None

    # Metriques par classe (AP50 + precision + recall + F1)
    ap_class_index = getattr(box, "ap_class_index", None)
    ap_matrix      = getattr(box, "ap",             None)
    p_per_cls      = getattr(box, "p",              None)   # precision par classe
    r_per_cls      = getattr(box, "r",              None)   # recall par classe

    per_class_ap50      = {}
    per_class_precision = {}
    per_class_recall    = {}
    per_class_f1        = {}

    if ap_matrix is not None and ap_class_index is not None:
        ap50 = ap_matrix[:, 0] if ap_matrix.ndim == 2 else ap_matrix
        for i, (idx, ap) in enumerate(zip(ap_class_index, ap50)):
            cls_name = names.get(int(idx), f"class_{idx}")
            per_class_ap50[cls_name] = float(ap)

            pc = float(p_per_cls[i]) if p_per_cls is not None and i < len(p_per_cls) else None
            rc = float(r_per_cls[i]) if r_per_cls is not None and i < len(r_per_cls) else None

            if pc is not None:
                per_class_precision[cls_name] = pc
            if rc is not None:
                per_class_recall[cls_name] = rc
            if pc is not None and rc is not None and (pc + rc) > 0:
                per_class_f1[cls_name] = 2 * pc * rc / (pc + rc)

    # F1 score moyen global
    p, r = float(box.mp), float(box.mr)
    f1_score = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    metrics = {
        "mode":                 mode,
        "model":                model_path,
        "dataset_yaml":         data_yaml,
        "mAP50":                float(box.map50),
        "mAP50_95":             float(box.map),
        "precision":            p,
        "recall":               r,
        "f1_score":             f1_score,
        "inference_ms":         inference_ms,
        "fps_gpu":              fps_gpu,
        "params_M":             n_params_M,
        "per_class_AP50":       per_class_ap50,
        "per_class_precision":  per_class_precision,
        "per_class_recall":     per_class_recall,
        "per_class_f1":         per_class_f1,
        "evaluated_at":         datetime.now().isoformat(),
    }

    # Afficher le tableau global
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
    if n_params_M is not None:
        print(f"   {'Parametres (M)':<32} {n_params_M:>10.2f}")

    # Afficher le tableau par classe
    if per_class_ap50:
        print(f"\n   {'Classe':<25} {'AP@50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  Statut")
        print(f"   {'-'*62}")
        for cls in sorted(per_class_ap50.keys()):
            ap  = per_class_ap50.get(cls, float("nan"))
            pc  = per_class_precision.get(cls, float("nan"))
            rc  = per_class_recall.get(cls, float("nan"))
            f1c = per_class_f1.get(cls, float("nan"))
            status = "OK" if ap >= 0.5 else ("~" if ap >= 0.3 else "!!")
            print(f"      {cls:<25} {ap:>7.4f} {pc:>7.4f} {rc:>7.4f} {f1c:>7.4f}  [{status}]")

    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"metrics_{mode}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   Metriques sauvegardees : {metrics_path}")

    return metrics


# =============================================================================
# RAPPORT COMBINE
# =============================================================================

def print_combined_report(nadir_metrics, oblique_metrics, output_dir):
    """Afficher et sauvegarder le rapport combine."""

    print("\n" + "=" * 65)
    print("   RAPPORT COMBINE — Evaluation Duale YOLO")
    print("=" * 65)

    # Tableau par classe
    print(f"\n   {'Classe':<25} {'Modele':<10} {'AP@50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  Statut")
    print(f"   {'-'*72}")

    all_results = {}

    for m, model_name in [(nadir_metrics, "nadir"), (oblique_metrics, "oblique")]:
        if m:
            for cls, ap in m.get("per_class_AP50", {}).items():
                all_results[cls] = {
                    "ap":    ap,
                    "prec":  m.get("per_class_precision", {}).get(cls, float("nan")),
                    "rec":   m.get("per_class_recall",    {}).get(cls, float("nan")),
                    "f1":    m.get("per_class_f1",        {}).get(cls, float("nan")),
                    "model": model_name,
                }

    for cls in ALL_CLASSES:
        if cls in all_results:
            d      = all_results[cls]
            status = "[OK]" if d["ap"] >= 0.5 else ("[~]" if d["ap"] >= 0.3 else "[!!]")
            print(f"   {cls:<25} {d['model']:<10} {d['ap']:>7.4f} {d['prec']:>7.4f} {d['rec']:>7.4f} {d['f1']:>7.4f}  {status}")
        else:
            print(f"   {cls:<25} {'N/A':<10} {'—':>7} {'—':>7} {'—':>7} {'—':>7}")

    print(f"   {'-'*55}")

    # Metriques globales par modele
    for label, m in [("NADIR", nadir_metrics), ("OBLIQUE", oblique_metrics)]:
        if m:
            print(f"\n   [{label}]")
            print(f"   mAP@50              : {m['mAP50']:.4f} ({m['mAP50'] * 100:.2f}%)")
            print(f"   mAP@50:95           : {m['mAP50_95']:.4f}")
            print(f"   Precision           : {m['precision']:.4f}")
            print(f"   Recall              : {m['recall']:.4f}")
            if m.get("f1_score") is not None:
                print(f"   F1 Score            : {m['f1_score']:.4f}")
            if m.get("inference_ms") is not None:
                print(f"   Vitesse Inference   : {m['inference_ms']:.2f} ms ↓")
            if m.get("fps_gpu") is not None:
                print(f"   FPS GPU             : {m['fps_gpu']:.1f}")
            if m.get("params_M") is not None:
                print(f"   Parametres          : {m['params_M']:.2f} M")

    # Moyennes globales combinées
    available = [m for m in [nadir_metrics, oblique_metrics] if m]

    def _mean(key):
        vals = [m[key] for m in available if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _sum_inv(key):
        """Pour les vitesses : moyenne harmonique via somme des inverses -> FPS moyen."""
        vals = [m[key] for m in available if m.get(key) is not None and m[key] > 0]
        return float(np.mean(vals)) if vals else None

    mean_map      = _mean("mAP50")
    mean_map95    = _mean("mAP50_95")
    mean_prec     = _mean("precision")
    mean_rec      = _mean("recall")
    mean_f1       = _mean("f1_score")
    mean_inf_ms   = _sum_inv("inference_ms")
    mean_fps      = _sum_inv("fps_gpu")
    mean_params   = _mean("params_M")

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

    # Sauvegarder le rapport combine
    combined = {
        "timestamp":      datetime.now().isoformat(),
        "nadir":          nadir_metrics,
        "oblique":        oblique_metrics,
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
        "per_class":      {
            cls: all_results[cls] for cls in ALL_CLASSES if cls in all_results
        },
    }

    report_path = os.path.join(output_dir, "metrics_combined.json")
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n   Rapport combine : {report_path}")

    return combined


# =============================================================================
# GRAPHIQUE
# =============================================================================

def plot_combined_metrics(nadir_metrics, oblique_metrics, output_dir):
    """Tracer l'AP@50 par classe pour les deux modeles."""

    entries = []

    for cls in ALL_CLASSES:
        for metrics, model in [(nadir_metrics, "nadir"), (oblique_metrics, "oblique")]:
            if metrics and cls in metrics.get("per_class_AP50", {}):
                entries.append({
                    "label": f"{cls}\n({model})",
                    "ap":    metrics["per_class_AP50"][cls],
                    "color": COLORS_MAP[model],
                })

    if not entries:
        return

    labels = [e["label"] for e in entries]
    ap50s  = [e["ap"]    for e in entries]
    colors = [e["color"] for e in entries]

    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 1.5), 6))
    bars = ax.bar(range(len(labels)), ap50s, color=colors, edgecolor="white", linewidth=0.5)

    for bar, ap in zip(bars, ap50s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{ap:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AP@50")
    ax.set_title("AP@50 par classe — Evaluation Duale YOLO")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red",    linestyle="--", alpha=0.5, label="Seuil 0.5")
    ax.axhline(y=0.3, color="orange", linestyle=":",  alpha=0.4, label="Seuil 0.3")
    ax.grid(True, alpha=0.3, axis="y")

    legend_handles = [
        mpatches.Patch(facecolor=COLORS_MAP[m], label=f"Modele {m}")
        for m in COLORS_MAP
    ] + [
        plt.Line2D([0], [0], color="red",    linestyle="--", label="Seuil 0.5"),
        plt.Line2D([0], [0], color="orange", linestyle=":",  label="Seuil 0.3"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_dual.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Graphique : {out_path}")


def plot_global_comparison(nadir_metrics, oblique_metrics, output_dir):
    """Graphique en barres des metriques globales (mAP50, Precision, Recall)."""

    available = {
        k: v for k, v in [("nadir", nadir_metrics), ("oblique", oblique_metrics)]
        if v is not None
    }

    if len(available) < 1:
        return

    metric_keys   = ["mAP50", "mAP50_95", "precision", "recall"]
    metric_labels = ["mAP@50", "mAP@50:95", "Precision", "Recall"]

    x     = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, metrics) in enumerate(available.items()):
        vals = [metrics[k] for k in metric_keys]
        offset = (i - (len(available) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"Modele {name}",
                      color=COLORS_MAP[name], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            )

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
        description="Evaluation duale YOLO — nadir + oblique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nadir-model",   default=None, help="Modele nadir (.pt)")
    parser.add_argument("--oblique-model", default=None, help="Modele oblique (.pt)")
    parser.add_argument("--nadir-data",    default=None, help="dataset.yaml nadir")
    parser.add_argument("--oblique-data",  default=None, help="dataset.yaml oblique")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EVALUATION_DIR", "./evaluation"),
        help="Dossier de sortie",
    )
    parser.add_argument("--skip-nadir",   action="store_true", help="Ignorer evaluation nadir")
    parser.add_argument("--skip-oblique", action="store_true", help="Ignorer evaluation oblique")
    args = parser.parse_args()

    output_base = os.getenv("OUTPUT_DIR", "./output")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("   EVALUATION DUALE YOLO — Toitures Cadastrales")
    print("=" * 65)

    nadir_metrics   = None
    oblique_metrics = None

    # -------------------------------------------------------------------
    # Evaluation nadir
    # -------------------------------------------------------------------

    if not args.skip_nadir:
        nadir_model_path, nadir_yaml = find_model_and_data("nadir", output_base)

        if args.nadir_model:
            nadir_model_path = args.nadir_model
        if args.nadir_data:
            nadir_yaml = args.nadir_data

        nadir_metrics = evaluate_model(
            nadir_model_path, nadir_yaml, "nadir", args.output_dir
        )

    # -------------------------------------------------------------------
    # Evaluation oblique
    # -------------------------------------------------------------------

    if not args.skip_oblique:
        oblique_model_path, oblique_yaml = find_model_and_data("oblique", output_base)

        if args.oblique_model:
            oblique_model_path = args.oblique_model
        if args.oblique_data:
            oblique_yaml = args.oblique_data

        oblique_metrics = evaluate_model(
            oblique_model_path, oblique_yaml, "oblique", args.output_dir
        )

    # -------------------------------------------------------------------
    # Rapport combine et graphiques
    # -------------------------------------------------------------------

    if nadir_metrics or oblique_metrics:
        print_combined_report(nadir_metrics, oblique_metrics, args.output_dir)
        plot_combined_metrics(nadir_metrics, oblique_metrics, args.output_dir)
        plot_global_comparison(nadir_metrics, oblique_metrics, args.output_dir)
    else:
        print("\n   Aucune metrique disponible.")
        print("   Lancez d'abord :")
        print("      python train.py --mode nadir")
        print("      python train.py --mode oblique")


if __name__ == "__main__":
    main()
