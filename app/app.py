"""
ForMyRoots - by DeepRooted Intelligence
A cultural-intelligence hair assistant for Sub-Saharan African hair.

Mobile-optimized build:
  - Controls live on the main page (no sidebar dependency / broken toggle)
  - Camera captures one photo at a time (face, then hair) so mobile browsers
    don't choke on two live camera streams
  - Responsive CSS: columns stack on narrow screens
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

st.set_page_config(
    page_title="ForMyRoots - DeepRooted Intelligence",
    page_icon="🌿",
    layout="centered",          # centered works far better on mobile than wide
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — earth-tone palette + responsive
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
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }

/* keep the sidebar toggle usable if the sidebar is ever opened */
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }

.block-container {
    padding: 1.5rem 1.25rem 3rem !important;
    max-width: 760px !important;
}
@media (min-width: 768px) {
    .block-container { padding: 2.5rem 2rem 3rem !important; }
}

/* Brand */
.fmr-wordmark { font-family: 'Playfair Display', serif; font-size: 13px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; color: #A44A3F; }
.fmr-company { font-size: 10px; color: #A8917C; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 2px; }
.fmr-hero-title { font-family: 'Playfair Display', serif; font-size: clamp(26px, 7vw, 42px);
    font-weight: 600; line-height: 1.15; color: #3A2A24; margin: 0.5rem 0 0.5rem; }
.fmr-hero-sub { font-size: 15px; color: #7A6354; font-weight: 300; line-height: 1.6; }

/* Cards */
.fmr-card { background: #FBF4EC; border: 1px solid #E4D3C0; border-radius: 16px;
    padding: 1.25rem; box-shadow: 0 1px 3px rgba(58,42,36,0.04); }
.fmr-card-label { font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #A8917C; margin-bottom: 0.75rem; }

.pill { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px;
    font-weight: 600; letter-spacing: 0.03em; margin: 3px 3px 0 0; }
.pill-clay { background: rgba(164,74,63,0.10); color: #A44A3F; border: 1px solid rgba(164,74,63,0.22); }
.pill-gold { background: rgba(200,162,74,0.14); color: #97751F; border: 1px solid rgba(200,162,74,0.3); }
.pill-bark { background: rgba(90,64,52,0.08); color: #5A4034; border: 1px solid rgba(90,64,52,0.18); }

.fmr-metric-value { font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600;
    color: #3A2A24; line-height: 1.1; }

.conf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.conf-name { font-size: 12px; color: #8A6F5C; width: 48px; flex-shrink: 0; }
.conf-bar-bg { flex: 1; height: 5px; background: #E4D3C0; border-radius: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #A44A3F, #C8A24A); }
.conf-pct { font-size: 11px; color: #A8917C; width: 30px; text-align: right; flex-shrink: 0; }

.style-card { background: #FBF4EC; border: 1px solid #E4D3C0; border-radius: 14px;
    padding: 1rem; margin-bottom: 10px; }
.style-card-num { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: #A44A3F;
    text-transform: uppercase; margin-bottom: 6px; }
.style-card-name { font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 600;
    color: #3A2A24; margin-bottom: 6px; }
.style-card-desc { font-size: 12px; color: #8A6F5C; line-height: 1.55; }

.culture-note { background: #F3E7DA; border-left: 3px solid #C8A24A; border-radius: 0 10px 10px 0;
    padding: 0.7rem 0.9rem; margin-top: 10px; }
.culture-label { font-size: 9px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: #97751F; margin-bottom: 4px; }
.culture-text { font-size: 11.5px; color: #6B5444; line-height: 1.55; }
.culture-meta { font-size: 11px; color: #8A6F5C; margin-top: 5px; }
.culture-meta b { color: #5A4034; font-weight: 600; }

.routine-row { display: grid; grid-template-columns: 84px 1fr; gap: 0 0.8rem;
    padding: 0.8rem 0; border-bottom: 1px solid #EADBC9; }
.routine-row:last-child { border-bottom: none; }
.routine-step { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #A8917C; padding-top: 2px; }
.routine-text { font-size: 13px; color: #5A4034; line-height: 1.55; }

.product-card { background: #F3E7DA; border: 1px solid #E4D3C0; border-radius: 12px;
    padding: 0.9rem 1rem; margin-bottom: 8px; display: grid; grid-template-columns: 1fr auto;
    gap: 0.5rem; align-items: start; }
.product-name { font-size: 13px; font-weight: 600; color: #3A2A24; margin-bottom: 3px; }
.product-why { font-size: 11px; color: #8A6F5C; line-height: 1.45; }
.product-where { font-size: 10px; color: #A8917C; margin-top: 4px; }
.product-price { font-family: 'Playfair Display', serif; font-size: 15px; font-weight: 600;
    color: #A44A3F; white-space: nowrap; }

.section-header { margin: 2rem 0 1rem; display: flex; align-items: baseline; gap: 12px; }
.section-title { font-family: 'Playfair Display', serif; font-size: 19px; font-weight: 600; color: #3A2A24; }
.section-line { flex: 1; height: 1px; background: #E4D3C0; }

.fmr-divider { height: 1px; background: #E4D3C0; margin: 2rem 0; }

/* Inputs */
[data-testid="stFileUploader"] { background: #FBF4EC !important; border: 1.5px dashed #D8C3AC !important;
    border-radius: 14px !important; padding: 1.25rem !important; }
[data-testid="stFileUploader"]:hover { border-color: #A44A3F !important; }
[data-testid="stCameraInput"] { background: #FBF4EC !important; border: 1.5px dashed #D8C3AC !important;
    border-radius: 14px !important; padding: 0.75rem !important; }
[data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid #E4D3C0 !important; }

/* Radio chips */
.stRadio [role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
.stRadio [role="radiogroup"] label {
    background: #FBF4EC !important; border: 1px solid #E4D3C0 !important;
    border-radius: 999px !important; padding: 6px 14px !important; color: #5A4034 !important;
    font-size: 13px !important; }

/* Buttons */
.stButton button {
    background: #A44A3F !important; color: #FBF4EC !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important; font-size: 14px !important;
    padding: 0.5rem 1rem !important; }
.stButton button:hover { background: #8E3D33 !important; }
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
        num_faces=1, min_face_detection_confidence=0.5)
    detector = vision.FaceLandmarker.create_from_options(opts)
    return (face_svm, face_sc, face_pca, face_le,
            hair_model, hair_sc, hair_pca, hair_le, detector)


def load_image(source, max_dim=1024):
    if source is None:
        return None, None
    pil = ImageOps.exif_transpose(Image.open(source).convert("RGB"))
    if max(pil.size) > max_dim:
        pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, bgr


# ── Prediction helpers ─────────────────────────────────────────────────────────
def _angle(a, b, c):
    ba = a - b; bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


def predict_face_shape(img_arr, detector, svm, scaler, pca, le):
    FOREHEAD_TOP=10; CHIN_BOTTOM=152; LEFT_CHEEK=234; RIGHT_CHEEK=454
    LEFT_JAW=172; RIGHT_JAW=397; LEFT_FOREHEAD=70; RIGHT_FOREHEAD=300
    LEFT_TEMPLE=162; RIGHT_TEMPLE=389; LEFT_MIDJAW=135; RIGHT_MIDJAW=364
    CHIN_LEFT=149; CHIN_RIGHT=378; NOSE_BRIDGE=168; UPPER_LIP=0
    LEFT_CHEEKBONE=116; RIGHT_CHEEKBONE=345

    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_arr.shape
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not result.face_landmarks:
        return None, None
    lm = result.face_landmarks[0]
    def pt(i): return np.array([lm[i].x * w, lm[i].y * h])

    face_height    = np.linalg.norm(pt(FOREHEAD_TOP) - pt(CHIN_BOTTOM))
    face_width     = np.linalg.norm(pt(LEFT_CHEEK) - pt(RIGHT_CHEEK))
    jaw_width      = np.linalg.norm(pt(LEFT_JAW) - pt(RIGHT_JAW))
    forehead_width = np.linalg.norm(pt(LEFT_FOREHEAD) - pt(RIGHT_FOREHEAD))
    temple_width   = np.linalg.norm(pt(LEFT_TEMPLE) - pt(RIGHT_TEMPLE))
    cheekbone_w    = np.linalg.norm(pt(LEFT_CHEEKBONE) - pt(RIGHT_CHEEKBONE))
    midjaw_width   = np.linalg.norm(pt(LEFT_MIDJAW) - pt(RIGHT_MIDJAW))
    chin_width     = np.linalg.norm(pt(CHIN_LEFT) - pt(CHIN_RIGHT))
    if face_width == 0 or face_height == 0 or forehead_width == 0:
        return None, None

    upper_third = np.linalg.norm(pt(FOREHEAD_TOP) - pt(NOSE_BRIDGE))
    mid_third   = np.linalg.norm(pt(NOSE_BRIDGE) - pt(UPPER_LIP))
    lower_third = np.linalg.norm(pt(UPPER_LIP) - pt(CHIN_BOTTOM))
    jaw_angle = (_angle(pt(LEFT_MIDJAW), pt(LEFT_JAW), pt(CHIN_BOTTOM)) +
                 _angle(pt(RIGHT_MIDJAW), pt(RIGHT_JAW), pt(CHIN_BOTTOM))) / 2.0
    chin_angle = _angle(pt(CHIN_LEFT), pt(CHIN_BOTTOM), pt(CHIN_RIGHT))

    feats = [[
        face_height/face_width, jaw_width/face_width, forehead_width/face_width,
        temple_width/face_width, cheekbone_w/face_width, midjaw_width/face_width,
        chin_width/face_width, jaw_width/forehead_width, jaw_width/cheekbone_w,
        forehead_width/cheekbone_w, chin_width/jaw_width, jaw_width/face_height,
        cheekbone_w/face_height, upper_third/face_height, mid_third/face_height,
        lower_third/face_height, upper_third/lower_third, jaw_angle/180.0,
        chin_angle/180.0, midjaw_width/jaw_width,
    ]]
    X = pca.transform(scaler.transform(feats))
    pred = svm.predict(X); prob = svm.predict_proba(X)[0]
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
    X = pca.transform(scaler.transform(np.concatenate([hist, gf]).reshape(1, -1)))
    pred = model.predict(X); prob = model.predict_proba(X)[0]
    return le.inverse_transform(pred)[0], dict(zip(le.classes_, prob))


STYLE_META = {
    "Afro Puff":     {"desc": "Quick, elegant gathered volume for any density.",      "tags": ["Casual", "Low Maintenance"]},
    "Bantu Knots":   {"desc": "Defined knots that create a striking silhouette.",      "tags": ["Trendy", "Bold"]},
    "Box Braids":    {"desc": "Protective style with excellent length versatility.",   "tags": ["Low Maintenance", "Trendy"]},
    "Cornrows":      {"desc": "Sleek, structured rows that work for any occasion.",    "tags": ["Professional", "Classic"]},
    "Locs":          {"desc": "Long-term protective style with cultural depth.",       "tags": ["Classic", "Low Maintenance"]},
    "High Top Fade": {"desc": "Sharp silhouette with clean lines at the sides.",       "tags": ["Bold", "Professional"]},
    "TWA":           {"desc": "Teeny-weeny afro - clean, confident, minimal.",         "tags": ["Low Maintenance", "Bold"]},
}
TAG_COLORS = {"Bold":"pill-clay","Classic":"pill-bark","Trendy":"pill-gold",
    "Professional":"pill-bark","Casual":"pill-bark","Low Maintenance":"pill-bark"}
HAIR_TYPE_DESC = {
    "3C":"Tight corkscrew curls, springy and well-defined.",
    "4A":"Tightly coiled S-pattern with visible definition.",
    "4B":"Z-shaped bends, soft and fluffy texture.",
    "4C":"Very tight coils, maximum shrinkage."}
FACE_SHAPE_DESC = {
    "oval":"Balanced proportions - the most versatile shape.",
    "round":"Equal width and length, soft features.",
    "square":"Strong jaw equal to forehead width.",
    "heart":"Wide forehead tapering to a narrow chin.",
    "oblong":"Face length clearly greater than width."}


def conf_bars_html(conf_dict, top_n=3):
    rows = ""
    for name, prob in sorted(conf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        pct = int(prob*100)
        rows += f'<div class="conf-row"><span class="conf-name">{name}</span><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%"></div></div><span class="conf-pct">{pct}%</span></div>'
    return rows

def tag_pills(tags):
    return " ".join(f'<span class="pill {TAG_COLORS.get(t,"pill-bark")}">{t}</span>' for t in tags)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:1.5rem;">
    <div class="fmr-wordmark">ForMyRoots</div>
    <div class="fmr-company">by DeepRooted Intelligence</div>
    <div class="fmr-hero-title">Let's understand your roots</div>
    <div class="fmr-hero-sub">Hairstyles designed for your hair, your features, and your culture - built for African hair.</div>
</div>
""", unsafe_allow_html=True)

# ── Controls on the main page (mobile-friendly) ────────────────────────────────
st.markdown('<div class="fmr-card-label">Hair State</div>', unsafe_allow_html=True)
state_pref = st.radio("Hair state", ["Natural", "Relaxed", "Fluffy"],
                      horizontal=True, label_visibility="collapsed")
hair_state = state_pref.lower()

st.markdown('<div class="fmr-card-label" style="margin-top:1rem;">Style Presentation</div>', unsafe_allow_html=True)
gender_pref = st.radio("Presentation", ["Either", "Masculine", "Feminine"],
                       horizontal=True, label_visibility="collapsed")
gender = {"Either":"unisex","Masculine":"masculine","Feminine":"feminine"}[gender_pref]

if hair_state == "natural":
    st.caption("Natural hair: we'll read your coil type (3C-4C) and face shape. Two photos needed.")
else:
    st.caption(f"{state_pref} hair: we'll read your face shape and match {hair_state}-appropriate styles. Face photo only.")

# ── Capture: one camera at a time ──────────────────────────────────────────────
st.markdown("""
<div class="section-header"><span class="section-title">Your photos</span><span class="section-line"></span></div>
""", unsafe_allow_html=True)

mode = st.radio("Input method", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")

face_src = None
hair_src = None

# FACE
st.markdown('<div class="fmr-card-label">1 · Your face</div>', unsafe_allow_html=True)
st.caption("Front-facing, good lighting, no sunglasses")
if mode == "Camera":
    face_src = st.camera_input("Face", key="face_cam", label_visibility="collapsed")
else:
    face_src = st.file_uploader("Face", type=["jpg","jpeg","png","webp","jfif"],
                                key="face_up", label_visibility="collapsed")
face_pil, face_arr = load_image(face_src)

# HAIR — only needed for natural; only show the camera AFTER face is captured
if hair_state == "natural":
    st.markdown('<div class="fmr-card-label" style="margin-top:1rem;">2 · Your hair texture</div>', unsafe_allow_html=True)
    st.caption("Close-up of your hair texture")
    if face_pil is None and mode == "Camera":
        st.info("Capture your face first, then your hair — one camera at a time on mobile.")
        hair_src = None
    else:
        if mode == "Camera":
            hair_src = st.camera_input("Hair", key="hair_cam", label_visibility="collapsed")
        else:
            hair_src = st.file_uploader("Hair", type=["jpg","jpeg","png","webp","jfif"],
                                        key="hair_up", label_visibility="collapsed")
    hair_pil, hair_arr = load_image(hair_src)
else:
    hair_pil, hair_arr = None, None


# ── Analysis ───────────────────────────────────────────────────────────────────
ready = face_pil is not None and (hair_state != "natural" or hair_pil is not None)

if ready:
    with st.spinner("Reading your roots..."):
        (face_svm, face_sc, face_pca, face_le,
         hair_model, hair_sc, hair_pca, hair_le, detector) = load_models()
        face_shape, face_conf = predict_face_shape(face_arr, detector, face_svm, face_sc, face_pca, face_le)
        if hair_state == "natural" and hair_pil is not None:
            hair_type, hair_conf = predict_hair_type(hair_arr, hair_model, hair_sc, hair_pca, hair_le)
        else:
            hair_type, hair_conf = None, None

    if face_shape is None:
        st.error("We couldn't find a face. Please use a clear, front-facing photo.")
        st.stop()

    # Profile
    st.markdown('<div class="section-header"><span class="section-title">Your hair profile</span><span class="section-line"></span></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="fmr-card" style="margin-bottom:10px;">
        <div class="fmr-card-label">Face Shape</div>
        <div class="fmr-metric-value">{face_shape.title()}</div>
        <div style="font-size:12px;color:#8A6F5C;margin:6px 0 1rem;">{FACE_SHAPE_DESC.get(face_shape,"")}</div>
        {conf_bars_html(face_conf)}
    </div>""", unsafe_allow_html=True)
    if hair_type is not None:
        st.markdown(f"""
        <div class="fmr-card">
            <div class="fmr-card-label">Hair Type · Natural</div>
            <div class="fmr-metric-value">{hair_type}</div>
            <div style="font-size:12px;color:#8A6F5C;margin:6px 0 1rem;">{HAIR_TYPE_DESC.get(hair_type,"")}</div>
            {conf_bars_html(hair_conf)}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="fmr-card">
            <div class="fmr-card-label">Hair State</div>
            <div class="fmr-metric-value">{hair_state.title()}</div>
            <div style="font-size:12px;color:#8A6F5C;margin-top:6px;">Coil-type analysis applies to natural hair. Styles below are matched to your face shape and {hair_state} hair.</div>
        </div>""", unsafe_allow_html=True)

    result = get_full_recommendation(face_shape, hair_state, hair_type, gender)
    filtered = result["hairstyles"]

    # Recommended styles
    st.markdown('<div class="section-header"><span class="section-title">Styles for your roots</span><span class="section-line"></span></div>', unsafe_allow_html=True)
    for i, style in enumerate(filtered):
        meta = STYLE_META.get(style, {"desc":"A great fit for your profile.","tags":[]})
        st.markdown(f"""
        <div class="style-card">
            <div class="style-card-num">0{i+1}</div>
            <div class="style-card-name">{style}</div>
            <div class="style-card-desc">{meta['desc']}</div>
            <div style="margin-top:8px">{tag_pills(meta['tags'])}</div>
        </div>""", unsafe_allow_html=True)

    # Styles + culture
    st.markdown('<div class="section-header"><span class="section-title">See the styles &amp; their stories</span><span class="section-line"></span></div>', unsafe_allow_html=True)
    for style in filtered:
        culture = get_style_culture(style)
        img_path = get_reference_image(style, gender)
        if img_path and os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.markdown(f"""
        <div class="fmr-card" style="margin-bottom:14px;">
            <div class="style-card-name" style="margin-bottom:8px;">{style}</div>
            <div class="culture-meta"><b>Origin:</b> {culture['origin']}</div>
            <div class="culture-meta"><b>Meaning:</b> {culture['meaning']}</div>
            <div class="culture-note">
                <div class="culture-label">Cultural context</div>
                <div class="culture-text">{culture['cultural_note']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Routine
    st.markdown('<div class="section-header"><span class="section-title">Caring for your hair</span><span class="section-line"></span></div>', unsafe_allow_html=True)
    r = result["routine"]
    st.markdown(f"""
    <div class="fmr-card">
        <div class="routine-row"><div class="routine-step">Cleanse</div><div class="routine-text">{r['cleanse']}</div></div>
        <div class="routine-row"><div class="routine-step">Condition</div><div class="routine-text">{r['condition']}</div></div>
        <div class="routine-row"><div class="routine-step">Style</div><div class="routine-text">{r['style']}</div></div>
        <div class="routine-row"><div class="routine-step">Tips</div><div class="routine-text">{r['extras']}</div></div>
    </div>""", unsafe_allow_html=True)

    # Products
    st.markdown('<div class="section-header"><span class="section-title">Products near you</span><span class="section-line"></span></div>', unsafe_allow_html=True)
    for p in result["products"]:
        st.markdown(f"""
        <div class="product-card">
            <div><div class="product-name">{p['name']}</div>
            <div class="product-why">{p['why']}</div>
            <div class="product-where">{p['where']}</div></div>
            <div class="product-price">R{p['price_ZAR']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="fmr-divider" style="margin-top:2.5rem"></div>
    <div style="font-size:11px;color:#A8917C;text-align:center;padding-bottom:1.5rem;">
        ForMyRoots by DeepRooted Intelligence - reconnecting us with our roots, one strand at a time.
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="margin-top:2rem;text-align:center;padding:2.5rem 1rem;">
        <div style="font-size:30px;margin-bottom:0.75rem;">🌿</div>
        <div style="font-size:13px;color:#A8917C;">Add your photo(s) above to discover styles made for your roots.</div>
    </div>""", unsafe_allow_html=True)