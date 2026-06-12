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


# ══════════════════════════════════════════════════════════════════════════════
# HAIR STATE HIERARCHY
#
#   Natural  -> sub-classified by type (3C / 4A / 4B / 4C)  [fully implemented]
#   Relaxed  -> chemically straightened; coil-type classifier does NOT apply
#   Fluffy   -> texture/state; sub-classification not yet known
#
# The face-shape pipeline is the same for all three states. Only the hair-type
# classifier and the type-specific routines/products differ.
#
# IMPORTANT: relaxed and fluffy routines/products below are PLACEHOLDERS.
# Replace them with knowledge from your salon contacts before relying on them.
# ══════════════════════════════════════════════════════════════════════════════

VALID_STATES = ("natural", "relaxed", "fluffy")


# ── Hair care routines, by state ────────────────────────────────────────────────
# Natural is keyed by hair type (3C/4A/4B/4C). Relaxed and fluffy are single
# routines for now (no sub-type).
NATURAL_HAIRCARE = {
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

# PLACEHOLDER — confirm with salon contacts and replace.
RELAXED_HAIRCARE = {
    "cleanse":   "Neutralising shampoo after relaxing; gentle moisturising shampoo otherwise",
    "condition": "Deep conditioner weekly to counter chemical dryness",
    "style":     "Light moisturiser and wrap lotion; avoid heavy product buildup",
    "extras":    "PLACEHOLDER - confirm relaxed-hair routine with salon contacts. "
                 "Note: relaxers are being phased out due to scalp/eczema concerns.",
}

# PLACEHOLDER — confirm with salon contacts and replace.
FLUFFY_HAIRCARE = {
    "cleanse":   "Moisturising shampoo",
    "condition": "Regular conditioning to manage volume and moisture",
    "style":     "Blow-out friendly; light moisturiser",
    "extras":    "PLACEHOLDER - confirm what 'fluffy' means to your stylists and "
                 "how it sub-classifies, then replace this routine.",
}


def get_haircare_routine(hair_state, hair_type=None):
    state = (hair_state or "natural").lower()
    if state == "relaxed":
        return RELAXED_HAIRCARE
    if state == "fluffy":
        return FLUFFY_HAIRCARE
    # natural
    return NATURAL_HAIRCARE.get((hair_type or "4C").upper(), NATURAL_HAIRCARE["4C"])


# ── Product database (local, from salon research) ────────────────────────────────
# "states" = which hair states the product suits.
# Products and notes informed by Lesotho salon research (Black Chic, Soft-n-Free,
# Sta-Sof-Fro, Dark n Lovely, etc.). Prices are approximate ZAR — update as needed.
PRODUCTS = [
    # Natural-hair focused
    {"name": "Black Chic Hair Food", "type": "treatment",
     "states": ["natural", "fluffy"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 25, "where": "Local supermarkets, salons",
     "why": "Affordable, reliable hair food to manage dryness and fragility"},
    {"name": "Soft-n-Free Moisturising Spray", "type": "styling",
     "states": ["natural", "relaxed", "fluffy"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 40, "where": "Clicks, Shoprite, salons",
     "why": "Everyday moisture spray; light and widely available"},
    {"name": "Dark n Lovely (moisturiser range)", "type": "treatment",
     "states": ["natural", "relaxed"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 55, "where": "Clicks, Shoprite, Pick n Pay",
     "why": "Common, trusted base for general hair maintenance"},
    {"name": "Castor Oil (100% pure)", "type": "oil",
     "states": ["natural", "fluffy"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 30, "where": "Clicks, pharmacies",
     "why": "Seals moisture, promotes growth, suits natural African hair"},
    {"name": "Sta-Sof-Fro", "type": "styling",
     "states": ["natural"], "hair_types": ["4A","4B","4C"],
     "price_ZAR": 45, "where": "Local supermarkets, salons",
     "why": "Softens and moisturises; used to maintain natural styles and perms"},
    {"name": "Special Feeling Gel", "type": "styling",
     "states": ["natural"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 35, "where": "Local supermarkets, salons",
     "why": "Styling hold; commonly used to maintain perms and set styles"},
    # Maintenance / shared
    {"name": "Restore", "type": "treatment",
     "states": ["natural", "relaxed"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 50, "where": "Clicks, salons",
     "why": "Maintenance for short styles (pixie, spiral) and general care"},
    {"name": "Revlon (maintenance range)", "type": "styling",
     "states": ["natural", "relaxed"], "hair_types": ["3C","4A","4B","4C"],
     "price_ZAR": 60, "where": "Clicks, Dis-Chem",
     "why": "Maintenance for short and styled hair"},
    # Relaxed-specific (chemical aftercare)
    {"name": "Neutralizer (post-relaxer)", "type": "treatment",
     "states": ["relaxed"], "hair_types": [],
     "price_ZAR": 40, "where": "Salons, beauty supply",
     "why": "Neutralises relaxer chemicals to protect the scalp after relaxing"},
]


def get_products(hair_state, hair_type=None, max_results=5):
    state = (hair_state or "natural").lower()
    matched = []
    for p in PRODUCTS:
        if state not in p.get("states", []):
            continue
        # For natural hair, also respect hair type when the product specifies types
        if state == "natural" and hair_type and p.get("hair_types"):
            if hair_type.upper() not in p["hair_types"]:
                continue
        matched.append(p)
    matched.sort(key=lambda x: x["price_ZAR"])
    return matched[:max_results]


# ── Master function ─────────────────────────────────────────────────────────
def get_full_recommendation(face_shape, hair_state="natural", hair_type=None, gender="unisex"):
    """
    hair_state: "natural" | "relaxed" | "fluffy"
    hair_type:  only meaningful when hair_state == "natural" (3C/4A/4B/4C)

    For relaxed/fluffy, the coil-type classifier does not apply, so hair_type
    is ignored for those branches.
    """
    state = (hair_state or "natural").lower()

    # Hairstyles: query the styles DB by face shape + gender always.
    # For natural hair we also use hair_type to refine; for relaxed/fluffy we
    # match on face shape + gender only (until those branches are detailed).
    if state == "natural" and hair_type:
        hairstyles = get_hairstyles(face_shape, hair_type, gender)
    else:
        # Relaxed / fluffy: recommend by face shape + gender, ignore coil type.
        # Use a permissive hair_type so the DB query still returns styles.
        hairstyles = get_hairstyles(face_shape, "4C", gender)

    return {
        "face_shape": face_shape,
        "hair_state": state,
        "hair_type":  hair_type if state == "natural" else None,
        "gender":     gender,
        "hairstyles": hairstyles,
        "routine":    get_haircare_routine(state, hair_type),
        "products":   get_products(state, hair_type),
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loaded {len(STYLES)} style entries "
          f"({len(set(s['name'] for s in STYLES))} unique styles)\n")

    # Natural, by type and gender
    print("="*52)
    print("  NATURAL — square face, 4C, masculine")
    print("="*52)
    rec = get_full_recommendation("square", "natural", "4C", "masculine")
    for i, s in enumerate(rec["hairstyles"], 1):
        print(f"  {i}. {s}")
    print("  Products:", ", ".join(p["name"] for p in rec["products"]))

    # Relaxed (type ignored)
    print("\n" + "="*52)
    print("  RELAXED — oval face, feminine (coil type ignored)")
    print("="*52)
    rec = get_full_recommendation("oval", "relaxed", None, "feminine")
    for i, s in enumerate(rec["hairstyles"], 1):
        print(f"  {i}. {s}")
    print("  Routine extras:", rec["routine"]["extras"])
    print("  Products:", ", ".join(p["name"] for p in rec["products"]))

    # Fluffy
    print("\n" + "="*52)
    print("  FLUFFY — round face, either")
    print("="*52)
    rec = get_full_recommendation("round", "fluffy", None, "unisex")
    print("  Routine extras:", rec["routine"]["extras"])
    print("  Products:", ", ".join(p["name"] for p in rec["products"]))
