import os

print("Model files:")
md = "models"
if os.path.exists(md):
    for f in sorted(os.listdir(md)):
        path = os.path.join(md, f)
        if os.path.isfile(path):
            print(f"  {os.path.getsize(path)/1e6:.2f} MB  {f}")

for extra in ["face_landmarker.task", "styles_db.json"]:
    if os.path.exists(extra):
        print(f"  {os.path.getsize(extra)/1e6:.2f} MB  {extra}")

print("\nAnything over 100 MB cannot go on GitHub.")
