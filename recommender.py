"""
recommender.py — Ori recommendation engine (database-driven)

Queries styles_db.json to recommend hairstyles, filtered by face shape,
hair type, and gender presentation. Also returns a hair care routine and
budget-friendly product suggestions.

This replaces the old hard-coded dictionary. The hairstyle recommendations
now come from the styles database, so adding new tagged styles to
styles_db.json improves recommendations with no code changes.
"""

import json
import os
import random

# ── Load styles database ────────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles_db.json")

def _load_styles():
    if not os.path.exists(_DB_PATH):
        return []
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

STYLES = _load_styles()


# ── Hairstyle recommendation (query the database) ────────────────────────────────
def get_hairstyles(face_shape, hair_type, gender="unisex", max_results=3):
    """
    Return recommended hairstyle names by querying styles_db.json.

    gender: "masculine" | "feminine" | "unisex" / "either"
      - masculine  -> masculine + unisex styles
      - feminine   -> feminine + unisex styles
      - unisex/either -> all styles
    """
    face_shape = face_shape.lower()
    hair_type  = hair_type.upper()
    gender     = (gender or "unisex").lower()

    if gender in ("masculine", "male", "man"):
        allowed_gender = {"masculine", "unisex"}
    elif gender in ("feminine", "female", "woman"):
        allowed_gender = {"feminine", "unisex"}
    else:
        allowed_gender = {"masculine", "feminine", "unisex"}

    # Score each candidate: +2 if face shape matches, +2 if hair type matches
    scored = {}
    for s in STYLES:
        if s["gender_presentation"] not in allowed_gender:
            continue
        score = 0
        if hair_type in s.get("hair_types", []):
            score += 2
        if face_shape in s.get("face_shapes", []):
            score += 2
        if score == 0:
            continue
        name = s["name"]
        # Keep the best score per unique style name
        if name not in scored or score > scored[name]:
            scored[name] = score

    if not scored:
        # Fallback: any style matching the gender filter
        names = list({s["name"] for s in STYLES
                      if s["gender_presentation"] in allowed_gender})
        return names[:max_results] if names else ["Afro", "Box Braids", "Cornrows"]

    # Sort by score desc, then name for stability
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:max_results]]


def get_reference_image(style_name, gender="unisex"):
    """Return one reference image path for a recommended style name, or None."""
    candidates = [s for s in STYLES if s["name"] == style_name]
    if not candidates:
        return None
    # Prefer an image matching the requested gender presentation
    g = (gender or "unisex").lower()
    if g in ("masculine", "male", "man"):
        pref = [c for c in candidates if c["gender_presentation"] in ("masculine", "unisex")]
    elif g in ("feminine", "female", "woman"):
        pref = [c for c in candidates if c["gender_presentation"] in ("feminine", "unisex")]
    else:
        pref = candidates
    pool = pref or candidates
    idx = abs(hash(style_name)) % len(pool)
    return pool[idx]["image_path"]


def get_style_culture(style_name):
    """Return {origin, meaning, cultural_note} for a style name, or empty dict."""
    for s in STYLES:
        if s["name"] == style_name:
            return {
                "origin":        s.get("origin", ""),
                "meaning":       s.get("meaning", ""),
                "cultural_note": s.get("cultural_note", ""),
            }
    return {"origin": "", "meaning": "", "cultural_note": ""}


# ── Hair care routine ─────────────────────────────────────────────────────────
HAIRCARE_RULES = {
    "3C": {
        "cleanse":   "Sulfate-free moisturising shampoo (wash every 5-7 days)",
        "condition": "Rinse-out conditioner + weekly deep conditioner (30 min)",
        "style":     "Leave-in conditioner + light curl cream or gel",
        "extras":    "Scrunch out moisture with a microfibre towel. Diffuse or air dry.",
    },
    "4A": {
        "cleanse":   "Co-wash weekly, clarifying shampoo every 2-3 weeks",
        "condition": "Deep conditioner weekly (45 min with heat cap)",
        "style":     "Leave-in conditioner + curl defining cream",
        "extras":    "Shingling or finger coiling helps define the curl pattern.",
    },
    "4B": {
        "cleanse":   "Moisturising shampoo every 1-2 weeks",
        "condition": "Deep conditioner weekly (1 hour). Protein treatment monthly.",
        "style":     "LOC method - Liquid (water), Oil (castor/jojoba), Cream (shea butter)",
        "extras":    "Stretched styles (braid-outs, twist-outs) reduce shrinkage and breakage.",
    },
    "4C": {
        "cleanse":   "Co-wash or moisturising shampoo every 1-2 weeks",
        "condition": "Deep conditioner weekly (1 hour with steam or heat cap)",
        "style":     "LOC or LCO method - seal with heavy butter or oil (shea, castor)",
        "extras":    "Protective styles recommended. Moisturise and seal every 2-3 days.",
    },
}

def get_haircare_routine(hair_type):
    return HAIRCARE_RULES.get(hair_type.upper(), HAIRCARE_RULES["4C"])


# ── Product database ─────────────────────────────────────────────────────────
PRODUCTS = [
    {"name": "Sofn'Free GroHealthy Moisture Retain Shampoo", "type": "shampoo",
     "hair_types": ["4A","4B","4C"], "price_ZAR": 45, "where": "Clicks, Pick n Pay",
     "why": "Sulfate-free, adds moisture to dry coily hair"},
    {"name": "ORS Olive Oil Creamy Aloe Shampoo", "type": "shampoo",
     "hair_types": ["3C","4A","4B","4C"], "price_ZAR": 60, "where": "Clicks, Dis-Chem",
     "why": "Gentle cleanse, olive oil prevents moisture loss"},
    {"name": "Dark & Lovely Au Naturale Moisture LOC Shampoo", "type": "shampoo",
     "hair_types": ["4B","4C"], "price_ZAR": 55, "where": "Shoprite, Pick n Pay",
     "why": "Formulated for natural African hair textures"},
    {"name": "ORS Olive Oil Replenishing Conditioner", "type": "conditioner",
     "hair_types": ["3C","4A","4B","4C"], "price_ZAR": 65, "where": "Clicks, Dis-Chem",
     "why": "Deep moisture, detangles 4C hair effectively"},
    {"name": "Sofn'Free Curl Activator & Moisturiser", "type": "conditioner",
     "hair_types": ["3C","4A"], "price_ZAR": 40, "where": "Clicks, Pick n Pay",
     "why": "Activates curl definition for 3C and 4A patterns"},
    {"name": "Dark & Lovely Au Naturale Curl Defining Creme", "type": "conditioner",
     "hair_types": ["4B","4C"], "price_ZAR": 70, "where": "Shoprite, Pick n Pay",
     "why": "Reduces shrinkage, defines coils"},
    {"name": "Eco Styler Olive Oil Gel", "type": "styling",
     "hair_types": ["3C","4A"], "price_ZAR": 80, "where": "Clicks, Dis-Chem",
     "why": "Hold without crunch, enhances curl definition"},
    {"name": "African Pride Moisture Miracle Curl Pudding", "type": "styling",
     "hair_types": ["4A","4B","4C"], "price_ZAR": 75, "where": "Clicks, Pick n Pay",
     "why": "Moisture-rich pudding for coily and kinky textures"},
    {"name": "Sofn'Free Shea Butter Wrap Lotion", "type": "styling",
     "hair_types": ["4B","4C"], "price_ZAR": 35, "where": "Shoprite, Clicks",
     "why": "Budget-friendly sealant, reduces frizz"},
    {"name": "Castor Oil (100% pure)", "type": "oil",
     "hair_types": ["3C","4A","4B","4C"], "price_ZAR": 30, "where": "Clicks, pharmacies",
     "why": "Seals moisture, promotes growth, suits all African hair types"},
    {"name": "African Pride Shea Butter & Coconut Oil", "type": "oil",
     "hair_types": ["4B","4C"], "price_ZAR": 55, "where": "Clicks, Pick n Pay",
     "why": "Heavy sealant for very dry 4B/4C hair"},
    {"name": "ORS Hair Mayonnaise Treatment", "type": "treatment",
     "hair_types": ["4A","4B","4C"], "price_ZAR": 70, "where": "Clicks, Dis-Chem",
     "why": "Protein treatment to strengthen brittle natural hair"},
]

def get_products(hair_type, max_results=5):
    matched = [p for p in PRODUCTS if hair_type.upper() in p["hair_types"]]
    matched.sort(key=lambda x: x["price_ZAR"])
    return matched[:max_results]


# ── Master function ─────────────────────────────────────────────────────────
def get_full_recommendation(face_shape, hair_type, gender="unisex"):
    hairstyles = get_hairstyles(face_shape, hair_type, gender)
    return {
        "face_shape": face_shape,
        "hair_type":  hair_type,
        "gender":     gender,
        "hairstyles": hairstyles,
        "routine":    get_haircare_routine(hair_type),
        "products":   get_products(hair_type),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loaded {len(STYLES)} style entries "
          f"({len(set(s['name'] for s in STYLES))} unique styles)\n")

    for gender in ["masculine", "feminine"]:
        print(f"\n{'='*50}\n  {gender.upper()} — square face, 4C hair\n{'='*50}")
        rec = get_full_recommendation("square", "4C", gender)
        for i, style in enumerate(rec["hairstyles"], 1):
            img = get_reference_image(style, gender)
            print(f"  {i}. {style}")
            print(f"     {img}")
