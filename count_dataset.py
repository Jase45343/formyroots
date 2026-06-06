# count_dataset.py  — replace entire file with this
import os

def audit_folder(folder):
    """Count all files and group by extension."""
    if not os.path.exists(folder):
        return 0, {}
    
    all_files = [f for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f))]
    
    ext_counts = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        ext = ext if ext else "(no extension)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    return len(all_files), ext_counts


print("=" * 60)
print("  BLACK HAIR INTELLIGENCE — Full Dataset Audit")
print("=" * 60)

total_all = 0

for bucket in ["faces", "hair", "styles"]:
    bucket_path = os.path.join("data", bucket)
    print(f"\n📁 data/{bucket}/")

    if not os.path.exists(bucket_path):
        print("   ⚠ Folder not found")
        continue

    subfolders = sorted([
        f for f in os.listdir(bucket_path)
        if os.path.isdir(os.path.join(bucket_path, f))
    ])

    bucket_total = 0
    for label in subfolders:
        path = os.path.join(bucket_path, label)
        total, ext_counts = audit_folder(path)
        bucket_total += total

        # Format extension breakdown
        ext_str = "  |  ".join(
            f"{ext}: {count}" for ext, count in sorted(ext_counts.items())
        )

        flag = "  ← old folder" if len(label) > 15 else ""
        print(f"   {label:<42} {total:>3} files  [{ext_str}]{flag}")

    print(f"   {'TOTAL':<42} {bucket_total:>3} files")
    total_all += bucket_total

print("\n" + "=" * 60)
print(f"  GRAND TOTAL: {total_all} files across all buckets")
print("=" * 60)
print("\nNOTE: Supported image formats for training: .jpg .jpeg .png .webp")
print("      Any other extensions will be skipped during feature extraction.")
print("      Rename or convert unsupported files if needed.")