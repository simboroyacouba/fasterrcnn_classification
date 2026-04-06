"""
split_dataset.py
Sépare instances_default.json en deux sous-datasets COCO :
  - instances_nadir.json   : Production_*.png  → panneau_solaire uniquement
  - instances_oblique.json : Snapshot_*.jpg    → toutes les classes (4)

Génère aussi classes_nadir.yaml et classes_oblique.yaml.

Usage :
    python split_dataset.py
    python split_dataset.py --input path/to/instances_default.json --output-dir ./datasets
"""

import os
import json
import yaml
import argparse
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_INPUT  = os.getenv(
    "DETECTION_DATASET_ANNOTATIONS_FILE",
    r"C:\Users\NEBRATA\Desktop\Memoire\Donnees\classification\dataset_all\instances_default.json"
)
DEFAULT_OUTPUT = os.getenv(
    "SPLIT_OUTPUT_DIR",
    r"C:\Users\NEBRATA\Desktop\Memoire\Donnees\classification\dataset_all"
)

NADIR_CLASSES   = {"panneau_solaire"}
OBLIQUE_CLASSES = {"panneau_solaire", "batiment_peint", "batiment_non_enduit", "batiment_enduit"}

# Règle de séparation par préfixe de nom de fichier
def get_view_type(file_name):
    if file_name.startswith("Production"):
        return "nadir"
    if file_name.startswith("Snapshot"):
        return "oblique"
    return None


# =============================================================================
# SPLIT
# =============================================================================

def split_coco(src_path, output_dir):
    with open(src_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_by_id   = {c["id"]: c["name"] for c in coco["categories"]}
    cat_by_name = {c["name"]: c for c in coco["categories"]}

    # --- Séparer les images ---
    nadir_images   = []
    oblique_images = []
    skipped        = []

    for img in coco["images"]:
        vtype = get_view_type(img["file_name"])
        if vtype == "nadir":
            nadir_images.append(img)
        elif vtype == "oblique":
            oblique_images.append(img)
        else:
            skipped.append(img["file_name"])

    nadir_img_ids   = {i["id"] for i in nadir_images}
    oblique_img_ids = {i["id"] for i in oblique_images}

    # --- Séparer les annotations ---
    nadir_anns   = []
    oblique_anns = []

    for ann in coco["annotations"]:
        cat_name = cat_by_id.get(ann["category_id"], "")
        if ann["image_id"] in nadir_img_ids and cat_name in NADIR_CLASSES:
            nadir_anns.append(ann)
        elif ann["image_id"] in oblique_img_ids and cat_name in OBLIQUE_CLASSES:
            oblique_anns.append(ann)

    # --- Construire les catégories filtrées ---
    nadir_cat_names   = NADIR_CLASSES
    oblique_cat_names = OBLIQUE_CLASSES

    nadir_cats   = [c for c in coco["categories"] if c["name"] in nadir_cat_names]
    oblique_cats = [c for c in coco["categories"] if c["name"] in oblique_cat_names]

    # --- Assembler les JSON COCO ---
    base = {"licenses": coco.get("licenses", []), "info": coco.get("info", {})}

    nadir_coco = {
        **base,
        "categories":   nadir_cats,
        "images":       nadir_images,
        "annotations":  nadir_anns,
    }
    oblique_coco = {
        **base,
        "categories":   oblique_cats,
        "images":       oblique_images,
        "annotations":  oblique_anns,
    }

    # --- Sauvegarder ---
    os.makedirs(output_dir, exist_ok=True)

    nadir_path   = os.path.join(output_dir, "instances_nadir.json")
    oblique_path = os.path.join(output_dir, "instances_oblique.json")

    with open(nadir_path, "w", encoding="utf-8") as f:
        json.dump(nadir_coco, f, indent=2, ensure_ascii=False)

    with open(oblique_path, "w", encoding="utf-8") as f:
        json.dump(oblique_coco, f, indent=2, ensure_ascii=False)

    return nadir_coco, oblique_coco, skipped


# =============================================================================
# YAML
# =============================================================================

COLORS = {
    "panneau_solaire":    [255, 0,   0],
    "batiment_peint":     [0,   255, 0],
    "batiment_non_enduit":[0,   0,   255],
    "batiment_enduit":    [255, 165, 0],
}

DESCRIPTIONS = {
    "panneau_solaire":    "Tôle ondulée métallique",
    "batiment_peint":     "Tôle bac acier",
    "batiment_non_enduit":"Tuiles en terre cuite ou béton",
    "batiment_enduit":    "Dalle béton (toit plat)",
}

def write_yaml(classes_list, path, comment):
    data = {
        "classes":      classes_list,
        "colors":       {c: COLORS[c] for c in classes_list if c != "__background__"},
        "descriptions": {c: DESCRIPTIONS[c] for c in classes_list if c != "__background__"},
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {comment}\n\n")
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"   OK: {path}")


# =============================================================================
# STATISTIQUES
# =============================================================================

def print_stats(label, coco_data):
    cat_names = {c["id"]: c["name"] for c in coco_data["categories"]}
    ann_per_cat = defaultdict(int)
    for ann in coco_data["annotations"]:
        ann_per_cat[cat_names.get(ann["category_id"], "?")] += 1

    # Images avec au moins 1 annotation
    img_with_ann = len({ann["image_id"] for ann in coco_data["annotations"]})

    print(f"\n{'-'*55}")
    print(f"  {label}")
    print(f"{'-'*55}")
    print(f"  Images totales    : {len(coco_data['images'])}")
    print(f"  Images annotees   : {img_with_ann}")
    print(f"  Annotations total : {len(coco_data['annotations'])}")
    print(f"  Classes           : {len(coco_data['categories'])}")
    print(f"\n  Distribution par classe :")
    for name, count in sorted(ann_per_cat.items(), key=lambda x: -x[1]):
        bar = "#" * (count // 20)
        print(f"    {name:<28} {count:>5}  {bar}")

    # Images sans annotation
    all_img_ids = {i["id"] for i in coco_data["images"]}
    ann_img_ids = {a["image_id"] for a in coco_data["annotations"]}
    no_ann = all_img_ids - ann_img_ids
    if no_ann:
        print(f"\n  /!\\ {len(no_ann)} image(s) sans annotation (seront ignorees a l'entrainement)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sépare le dataset COCO en nadir/oblique")
    parser.add_argument("--input",      default=DEFAULT_INPUT,  help="instances_default.json source")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Dossier de sortie")
    parser.add_argument("--yaml-dir",   default=os.path.dirname(os.path.abspath(__file__)),
                        help="Dossier pour les fichiers YAML (défaut: dossier du script)")
    args = parser.parse_args()

    print("=" * 55)
    print("  SPLIT DATASET  nadir / oblique")
    print("=" * 55)
    print(f"  Source : {args.input}")
    print(f"  Sortie : {args.output_dir}")

    # Split
    nadir_coco, oblique_coco, skipped = split_coco(args.input, args.output_dir)

    if skipped:
        print(f"\n  /!\\ {len(skipped)} image(s) ignorees (prefixe inconnu) :")
        for f in skipped[:5]:
            print(f"     {f}")

    # Stats
    print_stats("NADIR   — Production_*.png", nadir_coco)
    print_stats("OBLIQUE — Snapshot_*.jpg",   oblique_coco)

    # YAML
    print(f"\n{'-'*55}")
    print("  Generation des fichiers YAML")
    print(f"{'-'*55}")

    nadir_classes   = ["__background__"] + [c["name"] for c in nadir_coco["categories"]]
    oblique_classes = ["__background__"] + [c["name"] for c in oblique_coco["categories"]]

    write_yaml(
        nadir_classes,
        os.path.join(args.yaml_dir, "classes_nadir.yaml"),
        "Classes modèle NADIR (orthophotos Production_*.png)"
    )
    write_yaml(
        oblique_classes,
        os.path.join(args.yaml_dir, "classes_oblique.yaml"),
        "Classes modèle OBLIQUE (snapshots drone Snapshot_*.jpg)"
    )

    # Chemins à mettre dans .env
    print(f"\n{'-'*55}")
    print("  Variables .env a configurer")
    print(f"{'-'*55}")
    nadir_json   = os.path.join(args.output_dir, "instances_nadir.json").replace("\\", "/")
    oblique_json = os.path.join(args.output_dir, "instances_oblique.json").replace("\\", "/")
    print(f"\n  # Modèle NADIR")
    print(f"  NADIR_ANNOTATIONS_FILE={nadir_json}")
    print(f"  NADIR_CLASSES_FILE=classes_nadir.yaml")
    print(f"\n  # Modèle OBLIQUE")
    print(f"  OBLIQUE_ANNOTATIONS_FILE={oblique_json}")
    print(f"  OBLIQUE_CLASSES_FILE=classes_oblique.yaml")

    print(f"\n{'='*55}")
    print("  TERMINE")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
