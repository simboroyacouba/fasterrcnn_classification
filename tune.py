"""
Optimisation des hyperparametres Faster R-CNN avec Optuna.

Analogue de yolo_classification/tune.py pour Faster R-CNN (PyTorch natif).
Chaque essai entraine le modele pendant --tune-epochs epochs et optimise
le mAP@50 sur le split de validation.

Algorithme : TPE (Tree-structured Parzen Estimator) + MedianPruner.
Les etudes sont persistees dans une base SQLite (resumable).

Usage :
  python tune.py --mode oblique --n-trials 20
  python tune.py --mode nadir   --n-trials 20
  python tune.py --mode oblique --n-trials 30 --tune-epochs 15
  python tune.py --mode oblique --resume          # reprendre une etude existante
"""

import os
import gc
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from torch.utils.data import DataLoader, WeightedRandomSampler
from pycocotools.coco import COCO
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from train import (
    CONFIG,
    load_classes,
    stratified_split,
    CocoDetectionDataset,
    collate_fn,
    build_model,
    compute_sample_weights,
    train_one_epoch,
    evaluate_epoch,
)


# =============================================================================
# ESPACE DE RECHERCHE
# =============================================================================

def _suggest_hparams(trial):
    """Definit l'espace de recherche des hyperparametres."""
    return {
        "lr0":           trial.suggest_float("lr0",           1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True),
        "momentum":      trial.suggest_float("momentum",      0.85, 0.98),
        "freeze_epochs": trial.suggest_categorical("freeze_epochs", [0, 3, 5, 10]),
    }


# =============================================================================
# OBJECTIF OPTUNA
# =============================================================================

def make_objective(base_config, coco, train_ids, val_ids, cat_mapping):
    """
    Retourne la fonction objectif Optuna.
    Le split dataset est prepare une seule fois (train_ids/val_ids partages entre trials).
    """
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(base_config["classes"])

    def objective(trial):
        hp = _suggest_hparams(trial)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Datasets & DataLoaders ---
        train_dataset = CocoDetectionDataset(
            base_config["images_dir"], base_config["annotations_file"],
            train_ids, cat_mapping, base_config["image_size"], augment=True,
        )
        val_dataset = CocoDetectionDataset(
            base_config["images_dir"], base_config["annotations_file"],
            val_ids, cat_mapping, base_config["image_size"], augment=False,
        )

        sample_weights = compute_sample_weights(
            coco, train_ids, cat_mapping,
            base_config["classes"], base_config["rare_class_weights"],
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset, batch_size=base_config["batch_size"],
            sampler=sampler, collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # --- Modele ---
        model = build_model(num_classes, pretrained=True)
        model.to(device)

        freeze_n = hp["freeze_epochs"]
        if freeze_n > 0:
            for name, param in model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False

        params    = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=hp["lr0"],
            momentum=hp["momentum"],
            weight_decay=hp["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=base_config["tune_epochs"], eta_min=1e-5,
        )

        # --- Boucle d'entrainement ---
        best_map50 = 0.0
        try:
            for epoch in range(1, base_config["tune_epochs"] + 1):
                # Degel du backbone apres freeze_epochs
                if freeze_n > 0 and epoch == freeze_n + 1:
                    for param in model.parameters():
                        param.requires_grad = True
                    optimizer = torch.optim.SGD(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=hp["lr0"] / 5.0,
                        momentum=hp["momentum"],
                        weight_decay=hp["weight_decay"],
                    )
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=base_config["tune_epochs"] - freeze_n, eta_min=1e-5,
                    )

                train_one_epoch(model, optimizer, train_loader, device, epoch, base_config["tune_epochs"])
                val_map50, _ = evaluate_epoch(model, val_loader, device, base_config["score_threshold"])
                lr_scheduler.step()

                best_map50 = max(best_map50, val_map50)
                trial.report(val_map50, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"\n   [Trial {trial.number}] Erreur : {e}")
            best_map50 = 0.0
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                del model, optimizer, lr_scheduler
            except Exception:
                pass

        return best_map50

    return objective


# =============================================================================
# RAPPORT FINAL
# =============================================================================

def _print_and_save_report(study, base_config):
    """Affiche et sauvegarde le rapport de l'etude."""
    best = study.best_trial

    print("\n" + "=" * 70)
    print("   RESULTATS DE L'OPTIMISATION")
    print("=" * 70)
    print(f"   Nombre d'essais termines : {len(study.trials)}")
    print(f"   Meilleur mAP@50          : {best.value:.4f}  (trial #{best.number})")
    print(f"\n   Meilleurs hyperparametres :")
    for k, v in best.params.items():
        print(f"      {k:<20} {v}")

    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\n   Top-5 trials :")
    print(f"   {'Trial':>6}  {'mAP@50':>8}  {'lr0':>10}  {'wd':>10}  {'momentum':>8}")
    print(f"   {'-'*55}")
    for t in completed[:5]:
        p = t.params
        print(
            f"   {t.number:>6}  {t.value:>8.4f}  "
            f"{p.get('lr0', 0):>10.5f}  {p.get('weight_decay', 0):>10.6f}  "
            f"{p.get('momentum', 0):>8.4f}"
        )

    report = {
        "study_name":   study.study_name,
        "n_trials":     len(study.trials),
        "best_trial":   best.number,
        "best_map50":   best.value,
        "best_params":  best.params,
        "top5": [
            {"trial": t.number, "map50": t.value, "params": t.params}
            for t in completed[:5]
        ],
        "optimized_at": datetime.now().isoformat(),
    }

    report_path = os.path.join(base_config["tune_dir"], "best_hparams.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n   Meilleurs hparams sauvegardes : {report_path}")
    print(f"\n   Lancer l'entrainement final :")
    p = best.params
    print(
        f"   python train.py --mode {base_config['mode']}"
        f"  # puis editer .env : LR={p.get('lr0', ''):.5f}"
        + (f"  FREEZE_EPOCHS={p.get('freeze_epochs', 0)}" if p.get("freeze_epochs", 0) > 0 else "")
    )
    print("=" * 70)

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimisation hyperparametres Faster R-CNN — Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["nadir", "oblique", "all"], default="oblique",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Nombre d'essais Optuna.",
    )
    parser.add_argument(
        "--tune-epochs", type=int, default=10, help="Epochs par essai (moins que l'entrainement final).",
    )
    parser.add_argument(
        "--study-name", default=None, help="Nom de l'etude Optuna (default : frcnn_<mode>_<date>).",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reprendre une etude existante (meme --study-name).",
    )
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--output-dir",       default=None)
    args = parser.parse_args()

    mode       = args.mode
    output_dir = args.output_dir or os.path.join(os.getenv("OUTPUT_DIR", "./output"), mode)
    tune_dir   = os.path.join(output_dir, "tuning")
    os.makedirs(tune_dir, exist_ok=True)

    # --- Annotations & classes selon le mode ---
    base_annotations = os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    ann_dir = os.path.dirname(os.path.abspath(base_annotations))

    if args.annotations_file:
        annotations_file = args.annotations_file
    elif mode == "nadir":
        annotations_file = os.path.join(ann_dir, "instances_nadir.json")
    elif mode == "oblique":
        annotations_file = os.path.join(ann_dir, "instances_oblique.json")
    else:
        annotations_file = base_annotations

    if args.classes_file:
        classes_file = args.classes_file
    elif mode == "nadir":
        classes_file = "classes_nadir.yaml"
    elif mode == "oblique":
        classes_file = "classes_oblique.yaml"
    else:
        classes_file = os.getenv("CLASSES_FILE", "classes.yaml")

    classes = load_classes(classes_file)
    if not classes:
        print(f"Erreur : aucune classe pour le mode '{mode}'")
        return

    # Mettre a jour le CONFIG global utilise par evaluate_epoch / compute_map
    CONFIG["classes"] = classes

    rare_class_weights = {} if mode == "nadir" else CONFIG["rare_class_weights"]

    base_config = {
        "mode":              mode,
        "images_dir":        args.images_dir or os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
        "annotations_file":  annotations_file,
        "output_dir":        output_dir,
        "tune_dir":          tune_dir,
        "tune_epochs":       args.tune_epochs,
        "batch_size":        int(os.getenv("BATCH_SIZE", "2")),
        "image_size":        int(os.getenv("IMAGE_SIZE", "640")),
        "train_split":       float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":         float(os.getenv("VAL_SPLIT", "0.20")),
        "test_split":        float(os.getenv("TEST_SPLIT", "0.10")),
        "score_threshold":   float(os.getenv("SCORE_THRESHOLD", "0.5")),
        "classes":           classes,
        "rare_class_weights": rare_class_weights,
    }

    print("=" * 70)
    print(f"   OPTUNA — Optimisation hyperparametres Faster R-CNN")
    print("=" * 70)
    print(f"   Mode         : {mode}")
    print(f"   Classes      : {len(classes)} (avec __background__)")
    print(f"   Essais       : {args.n_trials}  x  {args.tune_epochs} epochs")
    print(f"   Dossier      : {tune_dir}")

    # --- Preparer le split une seule fois pour tous les essais ---
    print("\n   Preparation du dataset (une seule fois pour tous les essais)...")
    coco      = COCO(annotations_file)
    cat_ids   = coco.getCatIds()
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
    cat_mapping = {}
    for cat_id, cat_name in coco_cats.items():
        if cat_name in classes:
            cat_mapping[cat_id] = classes.index(cat_name)
        else:
            print(f"   Categorie COCO ignoree (absente du yaml) : '{cat_name}' (id={cat_id})")

    train_ids, val_ids, _, _ = stratified_split(
        coco,
        base_config["train_split"],
        base_config["val_split"],
        base_config["test_split"],
        seed=42,
    )

    # --- Creer ou reprendre l'etude ---
    study_name   = args.study_name or f"frcnn_{mode}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    storage_path = os.path.join(tune_dir, "optuna_study.db")
    storage_url  = f"sqlite:///{storage_path}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
    )

    already_done = len([t for t in study.trials if t.value is not None])
    n_remaining  = max(0, args.n_trials - already_done)

    if n_remaining == 0:
        print(f"\n   Etude deja complete ({already_done} essais). Utilisez --resume pour continuer.")
    else:
        if already_done > 0:
            print(f"\n   Reprise : {already_done} essais deja faits, {n_remaining} restants.")

        print(f"\n   Lancement de l'optimisation ({n_remaining} essais)...\n")
        study.optimize(
            make_objective(base_config, coco, train_ids, val_ids, cat_mapping),
            n_trials=n_remaining,
            show_progress_bar=True,
            catch=(Exception,),
        )

    _print_and_save_report(study, base_config)


if __name__ == "__main__":
    main()
