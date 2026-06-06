# ForMyRoots

**An AI hair assistant built for African hair — by DeepRooted Intelligence.**

ForMyRoots analyses your face shape and hair texture, then recommends
hairstyles made for African hair — complete with the cultural story behind
each one, a personalised care routine, and budget-friendly products available
across Southern & East Africa.

Most beauty AI was trained on hair textures that aren't ours. ForMyRoots is
built specifically for the 3C–4C coily and kinky textures common across
Sub-Saharan Africa, starting in Lesotho.

---

## About DeepRooted Intelligence

DeepRooted Intelligence builds AI systems that help people reconnect with
their cultural identity — starting with African hair in Lesotho. ForMyRoots
is our first product: hair is the most personal, everyday expression of
culture, and the first root we're pulling on.

---

## What it does

- **Face shape analysis** — MediaPipe facial landmarks + geometric ratios, classified with an SVM (5 shapes: oval, round, square, heart, oblong)
- **Hair type analysis** — Local Binary Pattern + Gabor texture features, classified with a Random Forest (4 types: 3C, 4A, 4B, 4C)
- **Style recommendations** — query a tagged styles database by face shape, hair type, and presentation
- **Cultural context** — each style carries its origin, meaning, and a Southern African cultural note
- **Care routine** — personalised by hair type
- **Local products** — budget-friendly options available in the region

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Project structure

```
app/app.py              Streamlit web app
recommender.py          Recommendation engine (queries styles_db.json)
build_styles_db.py      Builds the tagged styles database
styles_db.json          Tagged styles with cultural metadata
models/                 Trained classifiers (.pkl) + face landmarker
data/styles/            Reference images for each style
```

## Status

Working prototype. Face classifier ~77% cross-validation accuracy, hair
classifier ~72%. Both trained on a small dataset; accuracy improves as the
consented dataset grows.

---

*Recommendations are AI-generated and intended for inspiration. Cultural notes
are starting points, to be verified and deepened with community knowledge.*
