"""
Held-out end-to-end test set from the TEST splits of the public face-swap datasets
(never used for training, head selection or fusion calibration):

  test_sets/external/deepfake : DeepFakeFace InsightFace swaps, Celeb-DF (DF40) swaps, bitmind face swaps
  test_sets/external/real     : DeepFakeFace real IMDB-WIKI photos, Celeb-DF (DF40) real faces

Run: python build_external_test_set.py   then   python acceptance_test.py --set external --out test_results/external_<label>
"""
import glob
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
N = 40
SOURCES = [
    ("deepfake", "dff_insightface", "external_data/dff/test/fake/*"),
    ("deepfake", "celebdf", "external_data/df40/test/fake/*DF40_CDF_FS*"),
    ("deepfake", "bitmind", "external_data/bitmind_faceswap/test/fake/*"),
    ("real", "dff_wiki", "external_data/dff/test/real/*"),
    ("real", "celebdf", "external_data/df40/test/real/*DF40_CDF_FS*"),
]

for folder, tag, pattern in SOURCES:
    files = sorted(glob.glob(pattern))
    random.Random(99).shuffle(files)
    out = os.path.join("test_sets", "external", folder)
    os.makedirs(out, exist_ok=True)
    for p in files[:N]:
        shutil.copy2(p, os.path.join(out, f"{tag}__{os.path.basename(p)}"))
    print(f"{folder:9s} {tag:16s} {min(N, len(files))} images")
