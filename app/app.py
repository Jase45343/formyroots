"""
ForMyRoots - by DeepRooted Intelligence
A cultural-intelligence hair assistant for Sub-Saharan African hair.
Face shape + hair type analysis -> hairstyle recommendations with cultural
context, a personalised care routine, and budget-friendly local products.

────────────────────────────────────────────────────────────
RUN NORMALLY (your laptop):
    streamlit run app/app.py

PHONE ACCESS (same WiFi or phone hotspot):
    streamlit run app/app.py --server.address 0.0.0.0
    then open the Network URL on your phone.
    Large phone photos are auto-downscaled so uploads don't time out.
────────────────────────────────────────────────────────────
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import os
import sys
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage import img_as_float

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender import get_full_recommendation, get_reference_image, get_style_culture

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForMyRoots - DeepRooted Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# EARTH-TONE PALETTE
#   Deep brown   #3A2A24  (primary / text on light)
#   Warm sand    #F3E7DA  (background)
#   Clay red     #A44A3F  (accent)
#   Soft gold    #C8A24A  (highlight)
#   Bark         #5A4034  (secondary surfaces)
#   Ink          #1A1A1A  (darkest text)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #F3E7DA !important;
    color: #3A2A24 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #F3E7DA !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #EBDCCB !important;
    border-right: 1px solid #DECBB6 !important;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.25rem; }
[data-testid="stSidebar"] label {
    color: #8A6F5C !important; font-size: 11px !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio > div { gap: 6px; }
[data-testid="stSidebar"] .stRadio label {
    background: #F3E7DA !important; border: 1px solid #DECBB6 !important;
    border-radius: 10px !important; padding: 8px 14px !important;
    cursor: pointer !important; transition: all 0.2s !important;
    text-transform: none !important; font-size: 13px !important;
    letter-spacing: 0 !important; color: #5A4034 !important; font-weight: 400 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    border-color: #A44A3F !important; color: #A44A3F !important;
}

.block-container { padding: 2.5rem 3rem !important; max-width: 1080px !important; }

/* Brand */
.fmr-wordmark {
    font-family: 'Playfair Display', serif; font-size: 14px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; color: #A44A3F;
}
.fmr-company {
    font-size: 10px; color: #A8917C; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 2px;
}
.fmr-hero-title {
    font-family: 'Playfair Display', serif; font-size: clamp(30px, 4.5vw, 48px);
    font-weight: 600; line-height: 1.12; color: #3A2A24; margin: 0.6rem 0 0.6rem;
}
.fmr-hero-sub {
    font-size: 16px; color: #7A6354; font-weight: 300; line-height: 1.65; max-width: 520px;
}

/* Cards */
.fmr-card { background: #FBF4EC; border: 1px solid #E4D3C0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(58,42,36,0.04); }
.fmr-card-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: #A8917C; margin-bottom: 1rem;
}

/* Pills */
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px;
    font-weight: 600; letter-spacing: 0.03em; margin: 3px 3px 0 0;
}
.pill-clay { background: rgba(164,74,63,0.10); color: #A44A3F; border: 1px solid rgba(164,74,63,0.22); }
.pill-gold { background: rgba(200,162,74,0.14); color: #97751F; border: 1px solid rgba(200,162,74,0.3); }
.pill-bark { background: rgba(90,64,52,0.08); color: #5A4034; border: 1px solid rgba(90,64,52,0.18); }

.fmr-metric-value {
    font-family: 'Playfair Display', serif; font-size: 30px; font-weight: 600;
    color: #3A2A24; line-height: 1.1;
}

/* Confidence bars */
.conf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.conf-name { font-size: 12px; color: #8A6F5C; width: 52px; flex-shrink: 0; }
.conf-bar-bg { flex: 1; height: 5px; background: #E4D3C0; border-radius: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #A44A3F, #C8A24A); }
.conf-pct { font-size: 11px; color: #A8917C; width: 32px; text-align: right; flex-shrink: 0; }

/* Style cards */
.style-card {
    background: #FBF4EC; border: 1px solid #E4D3C0; border-radius: 16px;
    padding: 1.25rem; height: 100%; transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.style-card:hover { border-color: #A44A3F; transform: translateY(-2px); box-shadow: 0 6px 18px rgba(58,42,36,0.08); }
.style-card-num { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: #A44A3F; text-transform: uppercase; margin-bottom: 8px; }
.style-card-name { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 600; color: #3A2A24; margin-bottom: 8px; line-height: 1.25; }
.style-card-desc { font-size: 12px; color: #8A6F5C; line-height: 1.6; }

/* Cultural note block */
.culture-note {
    background: #F3E7DA; border-left: 3px solid #C8A24A; border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem; margin-top: 12px;
}
.culture-label { font-size: 9px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #97751F; margin-bottom: 4px; }
.culture-text { font-size: 11.5px; color: #6B5444; line-height: 1.6; }
.culture-meta { font-size: 11px; color: #8A6F5C; margin-top: 6px; }
.culture-meta b { color: #5A4034; font-weight: 600; }

/* Routine */
.routine-row { display: grid; grid-template-columns: 90px 1fr; gap: 0 1rem; padding: 0.9rem 0; border-bottom: 1px solid #EADBC9; }
.routine-row:last-child { border-bottom: none; }
.routine-step { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #A8917C; padding-top: 2px; }
.routine-text { font-size: 13px; color: #5A4034; line-height: 1.6; }

/* Products */
.product-card { background: #F3E7DA; border: 1px solid #E4D3C0; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 8px; display: grid; grid-template-columns: 1fr auto; gap: 0.5rem; align-items: start; }
.product-name { font-size: 13px; font-weight: 600; color: #3A2A24; margin-bottom: 3px; }
.product-why { font-size: 11px; color: #8A6F5C; line-height: 1.5; }
.product-where { font-size: 10px; color: #A8917C; margin-top: 4px; }
.product-price { font-family: 'Playfair Display', serif; font-size: 16px; font-weight: 600; color: #A44A3F; white-space: nowrap; }

/* Section header */
.section-header { margin: 2.5rem 0 1.25rem; display: flex; align-items: baseline; gap: 12px; }
.section-title { font-family: 'Playfair Display', serif; font-size: 21px; font-weight: 600; color: #3A2A24; }
.section-line { flex: 1; height: 1px; background: #E4D3C0; }

.fmr-divider { height: 1px; background: #E4D3C0; margin: 2rem 0; }

/* Streamlit element overrides */
[data-testid="stFileUploader"] { background: #FBF4EC !important; border: 1.5px dashed #D8C3AC !important; border-radius: 14px !important; padding: 1.5rem !important; }
[data-testid="stFileUploader"]:hover { border-color: #A44A3F !important; }
[data-testid="stFileUploader"] label { color: #8A6F5C !important; font-size: 13px !important; }
[data-testid="stFileUploader"] section { border: none !important; background: transparent !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }
[data-testid="stCameraInput"] { background: #FBF4EC !important; border: 1.5px dashed #D8C3AC !important; border-radius: 14px !important; padding: 1rem !important; }
button[kind="secondary"], [data-testid="stCameraInput"] button {
    background: #F3E7DA !important; border: 1px solid #D8C3AC !important; color: #5A4034 !important;
    border-radius: 10px !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important;
}
button[kind="secondary"]:hover, [data-testid="stCameraInput"] button:hover {
    border-color: #A44A3F !important; color: #A44A3F !important;
}
[data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid #E4D3C0 !important; }
.stRadio[role="radiogroup"] > label { color: #5A4034 !important; }
[data-testid="stCaptionContainer"] { color: #A8917C !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md   = os.path.join(base, "models")

    face_svm = joblib.load(os.path.join(md, "face_shape_svm.pkl"))
    face_sc  = joblib.load(os.path.join(md, "face_scaler.pkl"))
    face_pca = joblib.load(os.path.join(md, "face_pca.pkl"))
    face_le  = joblib.load(os.path.join(md, "face_label_encoder.pkl"))

    hair_model = joblib.load(os.path.join(md, "hair_type_model.pkl"))
    hair_sc    = joblib.load(os.path.join(md, "hair_scaler.pkl"))
    hair_pca   = joblib.load(os.path.join(md, "hair_pca.pkl"))
    hair_le    = joblib.load(os.path.join(md, "hair_label_encoder.pkl"))

    MODEL_PATH = os.path.join(base, "face_landmarker.task")
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1, min_face_detection_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(opts)
    return (face_svm, face_sc, face_pca, face_le,
            hair_model, hair_sc, hair_pca, hair_le, detector)


# ── Image loading (EXIF-corrected + downscaled) ─────────────────────────────────
def load_image(source, max_dim=1024):
    if source is None:
        return None, None
    pil = ImageOps.exif_transpose(Image.open(source).convert("RGB"))
    if max(pil.size) > max_dim:
        pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, bgr


# ── Prediction helpers ─────────────────────────────────────────────────────────
def predict_face_shape(img_arr, detector, svm, scaler, pca, le):
    FOREHEAD_TOP=10; CHIN_BOTTOM=152; LEFT_CHEEK=234; RIGHT_CHEEK=454
    LEFT_JAW=172;  RIGHT_JAW=397;  LEFT_FOREHEAD=70; RIGHT_FOREHEAD=300
    LEFT_TEMPLE=162; RIGHT_TEMPLE=389

    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_arr.shape
    result  = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not result.face_landmarks:
        return None, None
    lm = result.face_landmarks[0]
    def pt(i): return np.array([lm[i].x * w, lm[i].y * h])

    fh  = np.linalg.norm(pt(FOREHEAD_TOP) - pt(CHIN_BOTTOM))
    fw  = np.linalg.norm(pt(LEFT_CHEEK)   - pt(RIGHT_CHEEK))
    jw  = np.linalg.norm(pt(LEFT_JAW)     - pt(RIGHT_JAW))
    fw2 = np.linalg.norm(pt(LEFT_FOREHEAD)- pt(RIGHT_FOREHEAD))
    tw  = np.linalg.norm(pt(LEFT_TEMPLE)  - pt(RIGHT_TEMPLE))
    if fw == 0:
        return None, None

    X = pca.transform(scaler.transform([[fh/fw, jw/fw, fw2/fw, tw/fw, jw/fw2, jw/fh]]))
    pred = svm.predict(X)
    prob = svm.predict_proba(X)[0]
    return le.inverse_transform(pred)[0], dict(zip(le.classes_, prob))


def predict_hair_type(img_arr, model, scaler, pca, le):
    img  = cv2.resize(img_arr, (128, 128))
    gray = img_as_float(rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    lbp  = local_binary_pattern(gray, 24, 3, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    gf = []
    for freq in [0.1, 0.2, 0.3]:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            r, _ = gabor(gray, frequency=freq, theta=theta)
            gf.extend([r.mean(), r.var()])
    X    = pca.transform(scaler.transform(np.concatenate([hist, gf]).reshape(1, -1)))
    pred = model.predict(X)
    prob = model.predict_proba(X)[0]
    return le.inverse_transform(pred)[0], dict(zip(le.classes_, prob))


# ── Display metadata ─────────────────────────────────────────────────────────
STYLE_META = {
    "Afro Puff":     {"desc": "Quick, elegant gathered volume for any density.",      "tags": ["Casual", "Low Maintenance"]},
    "Bantu Knots":   {"desc": "Defined knots that create a striking silhouette.",      "tags": ["Trendy", "Bold"]},
    "Box Braids":    {"desc": "Protective style with excellent length versatility.",   "tags": ["Low Maintenance", "Trendy"]},
    "Cornrows":      {"desc": "Sleek, structured rows that work for any occasion.",    "tags": ["Professional", "Classic"]},
    "Locs":          {"desc": "Long-term protective style with cultural depth.",       "tags": ["Classic", "Low Maintenance"]},
    "High Top Fade": {"desc": "Sharp silhouette with clean lines at the sides.",       "tags": ["Bold", "Professional"]},
    "TWA":           {"desc": "Teeny-weeny afro - clean, confident, minimal.",         "tags": ["Low Maintenance", "Bold"]},
}
TAG_COLORS = {
    "Bold": "pill-clay", "Classic": "pill-bark", "Trendy": "pill-gold",
    "Professional": "pill-bark", "Casual": "pill-bark", "Low Maintenance": "pill-bark",
}
HAIR_TYPE_DESC = {
    "3C": "Tight corkscrew curls, pencil-sized, springy and well-defined.",
    "4A": "Tightly coiled S-pattern, straw-sized coils with visible definition.",
    "4B": "Z-shaped bends, less curl definition, soft and fluffy texture.",
    "4C": "Very tight coils, minimal visible curl pattern, maximum shrinkage.",
}
FACE_SHAPE_DESC = {
    "oval":   "Balanced proportions - the most versatile face shape.",
    "round":  "Equal width and length with soft, full features.",
    "square": "Strong jaw equal to forehead width, angular definition.",
    "heart":  "Wide forehead tapering to a narrow, pointed chin.",
    "oblong": "Face length clearly greater than width.",
}


def conf_bars_html(conf_dict, top_n=3):
    rows = ""
    for name, prob in sorted(conf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        pct = int(prob * 100)
        rows += f"""
        <div class="conf-row">
            <span class="conf-name">{name}</span>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%"></div></div>
            <span class="conf-pct">{pct}%</span>
        </div>"""
    return rows


def tag_pills(tags):
    return " ".join(
        f'<span class="pill {TAG_COLORS.get(t, "pill-bark")}">{t}</span>' for t in tags
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fmr-wordmark">ForMyRoots</div>', unsafe_allow_html=True)
    st.markdown('<div class="fmr-company">by DeepRooted Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="fmr-card-label">Style Presentation</div>', unsafe_allow_html=True)
    gender_pref = st.radio("Presentation", ["Either", "Masculine", "Feminine"], label_visibility="collapsed")
    gender_map = {"Either": "unisex", "Masculine": "masculine", "Feminine": "feminine"}
    gender = gender_map[gender_pref]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="fmr-card-label">Maintenance</div>', unsafe_allow_html=True)
    maint_pref = st.radio("Maintenance", ["Any", "Low", "Medium", "High"], label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#A8917C;line-height:1.6;">'
        'Recommendations are AI-generated and intended for inspiration. '
        'Cultural notes are starting points, verified with community knowledge.</div>',
        unsafe_allow_html=True
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2.5rem;">
    <div class="fmr-wordmark">ForMyRoots &nbsp;·&nbsp; by DeepRooted Intelligence</div>
    <div class="fmr-hero-title">Let's understand your roots</div>
    <div class="fmr-hero-sub">
        Hairstyles designed for your hair type, your features, and your culture.
        Share two photos and discover styles made for African hair - with the
        stories behind them.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Capture section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-title">Share your photos</span>
    <span class="section-line"></span>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Input method", ["Take photo", "Upload file"], horizontal=True, label_visibility="collapsed")

col1, col2 = st.columns(2, gap="large")
face_src = hair_src = None

with col1:
    st.markdown('<div class="fmr-card-label">Your face</div>', unsafe_allow_html=True)
    st.caption("Front-facing, good lighting, no sunglasses")
    if mode == "Take photo":
        face_src = st.camera_input("Face", key="face_cam", label_visibility="collapsed")
    else:
        face_src = st.file_uploader("Face", type=["jpg","jpeg","png","webp","jfif"],
                                    key="face_up", label_visibility="collapsed")

with col2:
    st.markdown('<div class="fmr-card-label">Your hair</div>', unsafe_allow_html=True)
    st.caption("Close-up of your hair texture")
    if mode == "Take photo":
        hair_src = st.camera_input("Hair", key="hair_cam", label_visibility="collapsed")
    else:
        hair_src = st.file_uploader("Hair", type=["jpg","jpeg","png","webp","jfif"],
                                    key="hair_up", label_visibility="collapsed")

face_pil, face_arr = load_image(face_src)
hair_pil, hair_arr = load_image(hair_src)


# ── Analysis ───────────────────────────────────────────────────────────────────
if face_pil is not None and hair_pil is not None:

    st.markdown("<br>", unsafe_allow_html=True)
    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.image(face_pil, caption="Your face", use_container_width=True)
    with p2:
        st.image(hair_pil, caption="Your hair", use_container_width=True)

    with st.spinner("Reading your roots..."):
        (face_svm, face_sc, face_pca, face_le,
         hair_model, hair_sc, hair_pca, hair_le, detector) = load_models()
        face_shape, face_conf = predict_face_shape(face_arr, detector, face_svm, face_sc, face_pca, face_le)
        hair_type,  hair_conf = predict_hair_type(hair_arr, hair_model, hair_sc, hair_pca, hair_le)

    if face_shape is None:
        st.error("We couldn't find a face. Please share a clear, front-facing photo.")
        st.stop()

    # ── Your profile ──
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Your hair profile</span><span class="section-line"></span>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.markdown(f"""
        <div class="fmr-card">
            <div class="fmr-card-label">Face Shape</div>
            <div class="fmr-metric-value">{face_shape.title()}</div>
            <div style="font-size:12px;color:#8A6F5C;margin:6px 0 1rem;">{FACE_SHAPE_DESC.get(face_shape, "")}</div>
            {conf_bars_html(face_conf)}
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="fmr-card">
            <div class="fmr-card-label">Hair Type</div>
            <div class="fmr-metric-value">{hair_type}</div>
            <div style="font-size:12px;color:#8A6F5C;margin:6px 0 1rem;">{HAIR_TYPE_DESC.get(hair_type, "")}</div>
            {conf_bars_html(hair_conf)}
        </div>
        """, unsafe_allow_html=True)

    result = get_full_recommendation(face_shape, hair_type, gender)
    hairstyles = result["hairstyles"]

    def maint_ok(style_name):
        if maint_pref == "Any":
            return True
        tags = STYLE_META.get(style_name, {}).get("tags", [])
        if maint_pref == "Low":
            return "Low Maintenance" in tags
        return True

    filtered = [s for s in hairstyles if maint_ok(s)] or hairstyles

    # ── Recommended styles ──
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Styles for your roots</span><span class="section-line"></span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(filtered), gap="medium")
    for i, (col, style) in enumerate(zip(cols, filtered)):
        meta = STYLE_META.get(style, {"desc": "A great fit for your profile.", "tags": []})
        with col:
            st.markdown(f"""
            <div class="style-card">
                <div class="style-card-num">0{i+1}</div>
                <div class="style-card-name">{style}</div>
                <div class="style-card-desc">{meta['desc']}</div>
                <div style="margin-top:10px">{tag_pills(meta['tags'])}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── See the styles + cultural context ──
    st.markdown("""
    <div class="section-header">
        <span class="section-title">See the styles &amp; their stories</span><span class="section-line"></span>
    </div>
    """, unsafe_allow_html=True)

    for style in filtered:
        culture = get_style_culture(style)
        img_path = get_reference_image(style, gender)
        ic, tc = st.columns([1, 2], gap="large")
        with ic:
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
        with tc:
            st.markdown(f"""
            <div class="fmr-card">
                <div class="style-card-name" style="margin-bottom:10px;">{style}</div>
                <div class="culture-meta"><b>Origin:</b> {culture['origin']}</div>
                <div class="culture-meta"><b>Meaning:</b> {culture['meaning']}</div>
                <div class="culture-note">
                    <div class="culture-label">Cultural context</div>
                    <div class="culture-text">{culture['cultural_note']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Care routine ──
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Caring for your hair</span><span class="section-line"></span>
    </div>
    """, unsafe_allow_html=True)

    routine = result["routine"]
    st.markdown(f"""
    <div class="fmr-card">
        <div class="routine-row"><div class="routine-step">Cleanse</div><div class="routine-text">{routine['cleanse']}</div></div>
        <div class="routine-row"><div class="routine-step">Condition</div><div class="routine-text">{routine['condition']}</div></div>
        <div class="routine-row"><div class="routine-step">Style</div><div class="routine-text">{routine['style']}</div></div>
        <div class="routine-row"><div class="routine-step">Tips</div><div class="routine-text">{routine['extras']}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Products ──
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Products near you</span><span class="section-line"></span>
    </div>
    <div style="font-size:12px;color:#A8917C;margin-bottom:1rem;">Affordable and available across Southern &amp; East Africa</div>
    """, unsafe_allow_html=True)

    for p in result["products"]:
        st.markdown(f"""
        <div class="product-card">
            <div>
                <div class="product-name">{p['name']}</div>
                <div class="product-why">{p['why']}</div>
                <div class="product-where">{p['where']}</div>
            </div>
            <div class="product-price">R{p['price_ZAR']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fmr-divider" style="margin-top:3rem"></div>
    <div style="font-size:11px;color:#A8917C;text-align:center;padding-bottom:2rem;">
        ForMyRoots by DeepRooted Intelligence - reconnecting us with our roots, one strand at a time.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="margin-top:3rem;text-align:center;padding:4rem 2rem;">
        <div style="font-size:34px;margin-bottom:1rem;">🌿</div>
        <div style="font-size:14px;color:#A8917C;">
            Share both photos above to discover styles made for your roots.
        </div>
    </div>
    """, unsafe_allow_html=True)