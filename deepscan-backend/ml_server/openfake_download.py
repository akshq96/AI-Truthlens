"""Selective OpenFake access over HTTP range requests (no full-file downloads).

  index:   python openfake_http.py index test            -> per-row-group label/model counts
  extract: python openfake_http.py extract test OUTDIR    -> images for chosen generators + reals
"""
import collections
import io
import json
import os
import sys
import time

import pyarrow.parquet as pq
import requests

REPO = "https://huggingface.co/datasets/ComplexDataLab/OpenFake/resolve/main/core/"
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openfake_data")
SESSION = requests.Session()


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        r = SESSION.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers.get("Content-Length") or r.headers.get("X-Linked-Size"))
        self.pos = 0
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(5):
            try:
                r = SESSION.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        data = r.content
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_pf(name):
    f = RangeFile(REPO + name)
    return f, pq.ParquetFile(f, pre_buffer=False, buffer_size=0)


def index(split):
    names = [f"{split}-{i:05d}-of-00013.parquet" for i in range(13)] if split == "test" else []
    out = {}
    t0 = time.time()
    for name in names:
        f, pf = open_pf(name)
        groups = []
        for g in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(g, columns=["label", "model"])
            c = collections.Counter(f"{l}|{m}" for l, m in zip(tbl.column("label").to_pylist(), tbl.column("model").to_pylist()))
            groups.append({"g": g, "rows": tbl.num_rows, "counts": dict(c)})
        out[name] = {"size": f.size, "groups": groups}
        tot = collections.Counter()
        for gr in groups:
            tot.update(gr["counts"])
        print(f"{name} {f.size/1e9:.1f}GB groups={len(groups)} fetched={f.fetched/1e6:.1f}MB "
              f"({time.time()-t0:.0f}s) {dict(tot.most_common(10))}", flush=True)
        json.dump(out, open(os.path.join(SP, f"openfake_{split}_index.json"), "w"), indent=1)


def index_files(names, out_json):
    out = json.load(open(out_json)) if os.path.exists(out_json) else {}
    t0 = time.time()
    for name in names:
        if name in out:
            continue
        f, pf = open_pf(name)
        groups = []
        for g in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(g, columns=["label", "model"])
            c = collections.Counter(f"{l}|{m}" for l, m in zip(tbl.column("label").to_pylist(), tbl.column("model").to_pylist()))
            groups.append({"g": g, "rows": tbl.num_rows, "bytes": pf.metadata.row_group(g).total_byte_size, "counts": dict(c)})
        out[name] = {"size": f.size, "groups": groups}
        tot = collections.Counter()
        for gr in groups:
            tot.update(gr["counts"])
        print(f"{name} {f.size/1e9:.1f}GB groups={len(groups)} ({time.time()-t0:.0f}s) {dict(tot.most_common(14))}", flush=True)
        json.dump(out, open(out_json, "w"), indent=1)


def extract(plan_json, outdir, byte_cap):
    """plan: [[file, group], ...]. Saves every image with label/model into outdir/<label>/<model>/."""
    plan = json.load(open(plan_json))
    meta_path = os.path.join(outdir, "metadata.jsonl")
    done = set()
    if os.path.exists(meta_path):
        for line in open(meta_path):
            r = json.loads(line)
            done.add((r["file"], r["group"]))
    fetched = 0
    t0 = time.time()
    for name, g in plan:
        if (name, g) in done:
            continue
        if fetched > byte_cap:
            print("byte cap reached", fetched / 1e9, "GB", flush=True)
            break
        f, pf = open_pf(name)
        tbl = pf.read_row_group(g, columns=["image", "label", "model", "prompt"])
        fetched += f.fetched
        imgs, labels, models, prompts = (tbl.column(c).to_pylist() for c in ("image", "label", "model", "prompt"))
        with open(meta_path, "a") as meta:
            for i, (im, lab, mod, pr) in enumerate(zip(imgs, labels, models, prompts)):
                data = im.get("bytes") if isinstance(im, dict) else None
                if not data:
                    continue
                d = os.path.join(outdir, lab, mod.replace("/", "_"))
                os.makedirs(d, exist_ok=True)
                ext = ".png" if data[:4] == b"\x89PNG" else ".jpg" if data[:2] == b"\xff\xd8" else ".webp" if data[8:12] == b"WEBP" else ".img"
                fn = os.path.join(d, f"{name.split('.')[0]}_g{g}_{i}{ext}")
                with open(fn, "wb") as fh:
                    fh.write(data)
                meta.write(json.dumps({"file": name, "group": g, "row": i, "path": fn, "label": lab, "model": mod,
                                       "prompt": (pr or "")[:300]}) + "\n")
        print(f"{name} g{g}: {len(imgs)} images, total fetched {fetched/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "index":
        index(sys.argv[2])
    elif sys.argv[1] == "index_files":
        index_files(sys.argv[3:], sys.argv[2])
    elif sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3], float(sys.argv[4]) * 1e9)
