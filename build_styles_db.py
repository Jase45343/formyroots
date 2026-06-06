"""
build_styles_db.py — Auto-generate a tagged styles database from data/styles/

Now includes the cultural layer: origin, meaning, and a Basotho / Southern
African cultural note for each style. This is the foundation of ForMyRoots'
cultural-intelligence identity and the first seed of the DeepRooted vision.

Run:  python build_styles_db.py
"""

import os
import json

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.jfif')
STYLES_DIR = "data/styles"
OUTPUT     = "styles_db.json"

# ── Default metadata per style family ──────────────────────────────────────────
# Cultural notes are written through a Southern Africa / Lesotho lens.
# Note: these are concise starting points — verify and deepen with community
# knowledge and Basotho elders/stylists as the project grows.
STYLE_DEFAULTS = {
    "afro puff natural hair African": {
        "name": "Afro Puff",
        "gender_presentation": "feminine",
        "hair_types": ["3C", "4A", "4B", "4C"],
        "face_shapes": ["oval", "round", "square", "heart", "oblong"],
        "length": "medium",
        "maintenance": "low",
        "occasion": "daily",
        "protective": False,
        "origin": "Pan-African; everyday natural style across the continent.",
        "meaning": "Celebration of natural texture worn in its unaltered form.",
        "cultural_note": (
            "In Southern Africa the puff is a common everyday way to wear "
            "natural hair without heat or chemicals. For many young Basotho "
            "women it has become a quiet statement of pride in natural texture."
        ),
    },
    "bantu knots natural hair African": {
        "name": "Bantu Knots",
        "gender_presentation": "unisex",
        "hair_types": ["4A", "4B", "4C"],
        "face_shapes": ["oval", "round", "heart", "oblong"],
        "length": "short",
        "maintenance": "medium",
        "occasion": "daily",
        "protective": True,
        "origin": "Named for the Bantu peoples of Southern and Central Africa.",
        "meaning": "Heritage style tied directly to Bantu cultural identity.",
        "cultural_note": (
            "Bantu knots carry the name of the Bantu peoples, whose languages "
            "and migrations shape much of Southern Africa, including the "
            "Sesotho-speaking world. The style doubles as a way to stretch and "
            "protect coily hair, and as a curl-setting method when taken down."
        ),
    },
    "box braids African hairstyle": {
        "name": "Box Braids",
        "gender_presentation": "unisex",
        "hair_types": ["3C", "4A", "4B", "4C"],
        "face_shapes": ["oval", "round", "square", "heart", "oblong"],
        "length": "long",
        "maintenance": "low",
        "occasion": "daily",
        "protective": True,
        "origin": "Ancient African braiding traditions, worn across the continent.",
        "meaning": "Protection, versatility, and personal expression.",
        "cultural_note": (
            "Braiding is one of the oldest African grooming traditions and "
            "remains central in Lesotho and across Southern Africa. Beyond "
            "style, box braids protect the hair for weeks and are often a "
            "shared, social act — done by family, friends, or a trusted braider."
        ),
    },
    "cornrows African man hairstyle": {
        "name": "Cornrows",
        "gender_presentation": "unisex",
        "hair_types": ["3C", "4A", "4B", "4C"],
        "face_shapes": ["oval", "square", "oblong", "round"],
        "length": "short",
        "maintenance": "low",
        "occasion": "daily",
        "protective": True,
        "origin": "Among the oldest documented African hairstyles.",
        "meaning": "Order, artistry, and community; patterns can carry meaning.",
        "cultural_note": (
            "Cornrows are an ancient African art form where the pattern itself "
            "can carry meaning. In Southern Africa they are worn by men, women, "
            "and children alike, valued as a neat, durable protective style and "
            "as a canvas for individual creativity."
        ),
    },
    "dreadlocks locs African hairstyle": {
        "name": "Locs",
        "gender_presentation": "unisex",
        "hair_types": ["4A", "4B", "4C"],
        "face_shapes": ["oval", "square", "oblong", "heart"],
        "length": "long",
        "maintenance": "low",
        "occasion": "daily",
        "protective": True,
        "origin": "Worn across Africa; deep spiritual and cultural roots.",
        "meaning": "Identity, spirituality, and a long-term commitment to natural hair.",
        "cultural_note": (
            "Locs carry spiritual and cultural weight across Africa and the "
            "diaspora. In Southern Africa they are worn both as a personal and "
            "spiritual statement and as a lasting protective style, growing and "
            "maturing with the person over years."
        ),
    },
    "high top fade African man": {
        "name": "High Top Fade",
        "gender_presentation": "masculine",
        "hair_types": ["3C", "4A", "4B", "4C"],
        "face_shapes": ["oval", "round", "square"],
        "length": "short",
        "maintenance": "medium",
        "occasion": "daily",
        "protective": False,
        "origin": "Rose to prominence through Black popular culture.",
        "meaning": "Bold self-expression and confidence.",
        "cultural_note": (
            "The high top fade is a bold, sculptural cut popular among young "
            "men across Southern Africa. It celebrates the upward growth and "
            "density of African hair, shaping it into clean, confident lines."
        ),
    },
    "TWA teeny weeny afro natural": {
        "name": "TWA",
        "gender_presentation": "unisex",
        "hair_types": ["3C", "4A", "4B", "4C"],
        "face_shapes": ["oval", "square", "heart"],
        "length": "short",
        "maintenance": "low",
        "occasion": "daily",
        "protective": False,
        "origin": "Often the starting point of a natural-hair journey.",
        "meaning": "A fresh start; embracing natural texture from the root.",
        "cultural_note": (
            "The teeny-weeny afro is often the first step when someone returns "
            "to their natural hair after relaxers or a big chop. In Southern "
            "Africa it is an increasingly visible symbol of embracing one's "
            "roots — low-maintenance, honest, and entirely natural."
        ),
    },
}


def build():
    if not os.path.exists(STYLES_DIR):
        print(f"ERROR: {STYLES_DIR} not found. Run from your project root.")
        return

    db = []
    style_id = 1

    for folder_name in sorted(os.listdir(STYLES_DIR)):
        folder = os.path.join(STYLES_DIR, folder_name)
        if not os.path.isdir(folder):
            continue

        d = STYLE_DEFAULTS.get(folder_name)
        if d is None:
            print(f"  ⚠ No defaults for '{folder_name}' — skipping.")
            continue

        images = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
        if not images:
            print(f"  ⚠ '{folder_name}' has no images — skipping.")
            continue

        for img in sorted(images):
            db.append({
                "style_id":   style_id,
                "name":       d["name"],
                "image_path": os.path.join(STYLES_DIR, folder_name, img).replace("\\", "/"),
                "gender_presentation": d["gender_presentation"],
                "hair_types":  d["hair_types"],
                "face_shapes": d["face_shapes"],
                "length":      d["length"],
                "maintenance": d["maintenance"],
                "occasion":    d["occasion"],
                "protective":  d["protective"],
                "origin":      d["origin"],
                "meaning":     d["meaning"],
                "cultural_note": d["cultural_note"],
            })
            style_id += 1

        print(f"  ✓ {d['name']:<16} {len(images)} images tagged (with cultural note)")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote {len(db)} style entries to {OUTPUT}")
    print(f"  Unique styles: {len(set(e['name'] for e in db))}")
    print("\nEach style now carries origin, meaning, and a Southern Africa /")
    print("Lesotho cultural note. Review and deepen these with community knowledge.")


if __name__ == "__main__":
    print("Building styles database (with cultural layer)...\n")
    build()
