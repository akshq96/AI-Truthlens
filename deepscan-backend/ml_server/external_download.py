"""
Selective download of public face-swap deepfake datasets (HTTP range requests, only the
parts used), into external_data/<dataset>/<split>/<real|fake>/.

  python external_download.py dff        # OpenRL/DeepFakeFace: InsightFace swaps + paired real wiki photos
  python external_download.py bitmind    # bitmind/face-swap (fakes only)
  python external_download.py df40 PLAN  # ThinothW/Deepfake-Identity-Isolated-Dataset-PreP, row groups in PLAN json

Splits are identity-safe:
  DeepFakeFace: by IMDB-WIKI folder (00-69 train, 70-84 val, 85-99 test); a fake and its
    paired real photo always land in the same split.
  Identity-Isolated: the dataset's own identity-isolated train / validation / test splits.
  bitmind: the dataset's own splits.
Every saved file is listed in external_data/metadata.jsonl (dataset, split, label, source name).
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import io
import json
import os
import random
import sys
import threading
import time
import zipfile

import pyarrow.parquet as pq

from openfake_download import RangeFile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "external_data")
SEED = 5
_meta_lock = threading.Lock()


def log(*a):
    print(*a, flush=True)


def save(dataset, split, label, name, data, source):
    d = os.path.join(OUT, dataset, split, label)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "wb") as fh:
        fh.write(data)
    with _meta_lock, open(os.path.join(OUT, "metadata.jsonl"), "a") as fh:
        fh.write(json.dumps({"dataset": dataset, "split": split, "label": label,
                             "path": os.path.relpath(path, HERE), "source": source}) + "\n")


def _ext(data: bytes) -> str:
    return ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"


# ------------------------------------------------------------------ DeepFakeFace
DFF = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
DFF_PAIRS = {"train": 1500, "val": 300, "test": 300}


def dff_split(folder: int) -> str:
    return "train" if folder < 70 else ("val" if folder < 85 else "test")


def dff():
    z = zipfile.ZipFile(RangeFile(DFF + "insight.zip"))
    names = sorted(i.filename.split("/", 1)[1] for i in z.infolist() if not i.is_dir())
    by_split = collections.defaultdict(list)
    for n in names:
        by_split[dff_split(int(n.split("/")[0]))].append(n)
    jobs = []
    for split, want in DFF_PAIRS.items():
        pick = sorted(by_split[split])
        random.Random(SEED).shuffle(pick)
        jobs += [(split, n) for n in pick[:want]]
    log(f"DeepFakeFace: {len(jobs)} pairs to fetch")
    # One HTTP range request per member: local header + compressed data, located from the
    # central directory (zipfile's own reader issues several small requests per member).
    wiki = RangeFile(DFF + "wiki.zip")
    infos = {"insight": {i.filename: i for i in z.infolist()},
             "wiki": {i.filename: i for i in zipfile.ZipFile(wiki).infolist()}}
    urls = {"insight": z.fp.url if hasattr(z.fp, "url") else RangeFile(DFF + "insight.zip").url, "wiki": wiki.url}
    local = threading.local()

    def member(kind, name):
        import requests
        import struct
        import zlib
        if not hasattr(local, "s"):
            local.s = requests.Session()
        info = infos[kind][name]
        start = info.header_offset
        end = start + 30 + len(info.filename.encode()) + 512 + info.compress_size
        for attempt in range(5):
            try:
                r = local.s.get(urls[kind], headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                r.raise_for_status()
                b = r.content
                sig, *_rest = struct.unpack("<IHHHHHIIIHH", b[:30])
                if sig != 0x04034B50:
                    raise ValueError("bad local header")
                nlen, elen = _rest[-2], _rest[-1]
                data = b[30 + nlen + elen:30 + nlen + elen + info.compress_size]
                if len(data) < info.compress_size:
                    raise ValueError("short read")
                return zlib.decompress(data, -15) if info.compress_type == 8 else data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    def work(job):
        split, n = job
        flat = n.replace("/", "_")
        for kind, label in (("insight", "fake"), ("wiki", "real")):
            name = f"{kind}_{flat}"
            if not os.path.exists(os.path.join(OUT, "dff", split, label, name)):
                save("dff", split, label, name, member(kind, f"{kind}/{n}"), f"{kind}/{n}")

    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for _ in ex.map(work, jobs):
            done += 1
            if done % 300 == 0:
                log(f"  {done}/{len(jobs)} pairs ({time.time() - t0:.0f}s)")
    log(f"DeepFakeFace done in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ bitmind/face-swap
BITMIND = "https://huggingface.co/datasets/bitmind/face-swap/resolve/main/data/"
BITMIND_N = {"train": ("face_train.parquet", 1200), "val": ("face_validation.parquet", 300), "test": ("face_test.parquet", 300)}


def bitmind():
    t0 = time.time()
    for split, (fname, want) in BITMIND_N.items():
        tbl = pq.ParquetFile(RangeFile(BITMIND + fname), pre_buffer=False, buffer_size=0).read(columns=["filename", "image"])
        rows = list(zip(tbl.column("filename").to_pylist(), tbl.column("image").to_pylist()))
        random.Random(SEED).shuffle(rows)
        for i, (fn, data) in enumerate(rows[:want]):
            name = f"bitmind_{i:05d}_{os.path.basename(fn) or 'img'}"
            if not os.path.splitext(name)[1]:
                name += _ext(data)
            save("bitmind_faceswap", split, "fake", name, data, fn)
        log(f"bitmind {split}: {min(want, len(rows))} saved ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ Identity-Isolated (DF40 / CDF / EFS)
DF40 = "https://huggingface.co/datasets/ThinothW/Deepfake-Identity-Isolated-Dataset-PreP/resolve/main/data/"


def df40(plan_path):
    """plan: [{"file", "group", "split", "n_fake", "n_real"}]; rows sampled evenly per group."""
    plan = json.load(open(plan_path))
    t0 = time.time()
    for item in plan:
        pf = pq.ParquetFile(RangeFile(DF40 + item["file"]), pre_buffer=False, buffer_size=0)
        tbl = pf.read_row_group(item["group"], columns=["image", "label"])
        imgs, labels = tbl.column("image").to_pylist(), tbl.column("label").to_pylist()
        for lab_id, lab, want in ((0, "fake", item.get("n_fake", 0)), (1, "real", item.get("n_real", 0))):
            idx = [i for i, l in enumerate(labels) if l == lab_id
                   and os.path.basename(imgs[i].get("path") or "").startswith(tuple(item.get("prefixes", [""])))]
            random.Random(SEED + item["group"]).shuffle(idx)
            for i in idx[:want]:
                data, src = imgs[i]["bytes"], imgs[i].get("path") or ""
                name = f"df40_{item['file'][:5]}{item['group']}_{i:05d}_{os.path.basename(src) or 'img'}"
                if not os.path.splitext(name)[1]:
                    name += _ext(data)
                save("df40", item["split"], lab, name, data, src)
        log(f"df40 {item['file']} g{item['group']} -> {item['split']} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"dff": dff, "bitmind": bitmind}.get(cmd, lambda: df40(sys.argv[2]))()
