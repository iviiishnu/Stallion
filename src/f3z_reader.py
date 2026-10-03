"""
f3z_reader.py — Read body geometry straight out of a Fusion 360 archive
(.f3z / .f3d) without Fusion installed. This is how the measurements in
sofa_structure.py were taken from data/cad/Sofa Internal Structure.f3z.

What it understands
  * .f3z  — zip of .f3d designs (Manifest.json names the root design)
  * .f3d  — zip whose entries are zstd-compressed (zip method 93)
  * Breps.BlobParts/*.smb — ACIS/ASM binary B-rep ("ASM BinaryFile4"),
    units of 1 cm, one `body` record per Fusion body

Body *names* and component membership live in Fusion's proprietary design
stream, so bodies come back by index with exact vertex / edge geometry;
sofa_structure.py records which body became which part ("#NNN" comments).

CLI
    python src/f3z_reader.py "data/cad/Sofa Internal Structure.f3z" --out outputs/reference

writes  reference_bodies.csv   one row per solid body (bbox, faces) in the
                               project frame (X length, Y depth from front, Z height)
        reference_overlay.png  reference wireframe vs the regenerated 3-seater
Requires the `zstandard` package.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Archive layer
# ─────────────────────────────────────────────────────────────────────────────


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.compress_type != 93:                        # 93 = Zstandard
        return zf.read(info)
    import zstandard
    fp = zf.fp
    fp.seek(info.header_offset)
    head = fp.read(30)
    n, extra = struct.unpack_from("<HH", head, 26)
    fp.seek(info.header_offset + 30 + n + extra)
    return zstandard.ZstdDecompressor().decompress(fp.read(info.compress_size), max_output_size=info.file_size)


def designs(f3z_path) -> dict:
    """{design name: {entry name: bytes}} for every .f3d inside a .f3z (or a single .f3d)."""
    path = Path(f3z_path)
    out = {}
    if path.suffix.lower() == ".f3d":
        blobs = {path.stem: path.read_bytes()}
        names = {path.stem: path.stem}
    else:
        z = zipfile.ZipFile(path)
        blobs = {i.filename: z.read(i) for i in z.infolist() if i.filename.endswith(".f3d")}
        names = {k: k for k in blobs}
        if "DesignDescription.json" in z.namelist():
            desc = json.loads(z.read("DesignDescription.json"))
            for obj in desc["designDescription"]["designGraphs"][0]["designObjects"]:
                names[obj["relativePath"]] = obj["displayName"]
    for key, blob in blobs.items():
        inner = zipfile.ZipFile(io.BytesIO(blob))
        out[names[key]] = {i.filename: _read_member(inner, i) for i in inner.infolist() if not i.is_dir()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ACIS SAB layer
# ─────────────────────────────────────────────────────────────────────────────


class Rec:
    __slots__ = ("idx", "name", "fields")

    def __init__(self, idx, name, fields):
        self.idx, self.name, self.fields = idx, name, fields

    def refs(self):
        return [v for t, v in self.fields if t == "ref"]

    def of(self, kind):
        return [v for t, v in self.fields if t == kind]


_FIXED = {0x04: ("int", "<i", 4), 0x05: ("float", "<f", 4), 0x06: ("dbl", "<d", 8),
          0x0C: ("ref", "<i", 4), 0x13: ("pos", "<ddd", 24), 0x14: ("vec", "<ddd", 24),
          0x15: ("enum", "<i", 4), 0x16: ("vec2", "<dd", 16), 0x17: ("i64", "<q", 8),
          0x03: ("short", "<h", 2)}


def parse_sab(data: bytes) -> list:
    """Records of an 'ASM BinaryFile4' stream. Index 0 is the asmheader, so
    list positions equal the file's own entity reference numbers."""
    if not data.startswith(b"ASM BinaryFile4"):
        raise ValueError("not an ASM BinaryFile4 stream")
    p = 15 + 16
    recs, name, cur = [], [], []
    while p < len(data):
        t = data[p]; p += 1
        if t in _FIXED:
            kind, fmt, size = _FIXED[t]
            v = struct.unpack_from(fmt, data, p)
            cur.append((kind, v if len(v) > 1 else v[0])); p += size
        elif t == 0x02:
            cur.append(("char", data[p])); p += 1
        elif t in (0x07, 0x08, 0x09, 0x12, 0x0D, 0x0E):
            if t in (0x07, 0x0D, 0x0E):
                n = data[p]; p += 1
            elif t == 0x08:
                n = struct.unpack_from("<H", data, p)[0]; p += 2
            else:
                n = struct.unpack_from("<I", data, p)[0]; p += 4
            s = data[p:p + n].decode("latin1"); p += n
            if t in (0x0D, 0x0E):
                name.append(s)
            else:
                cur.append(("str", s))
        elif t in (0x0A, 0x0B):
            cur.append(("bool", t == 0x0A))
        elif t in (0x0F, 0x10):
            cur.append(("open" if t == 0x0F else "close", None))
        elif t == 0x11:
            nm = "-".join(name)
            recs.append(Rec(len(recs), nm, cur))
            name, cur = [], []
            if nm.startswith(("End-of-ASM", "End-of-ACIS")):
                break
        else:
            raise ValueError(f"unknown SAB tag {t:#x} at byte {p - 1}")
    return recs


def _transform(R, idx):
    M = np.eye(4)
    if idx < 0:
        return M
    f = R[idx].fields
    v = [x[1] for x in f if x[0] == "vec"]
    s = [x[1] for x in f if x[0] == "dbl"][0]
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = v[0], v[1], v[2], v[3]
    M[:3, :3] *= s
    return M


_TOPO = {"lump", "shell", "subshell", "face", "loop", "coedge", "edge", "vertex"}


def bodies(R, unit_mm=10.0, arc_step=0.2):
    """Per body: edge polylines (mm), face count, bbox. Straight and elliptic
    edges are exact; spline edges are drawn as chords."""
    pt = lambda v: np.array(R[R[v].fields[5][1]].of("pos")[0])
    out = []
    for b in R:
        if b.name != "body":
            continue
        M = _transform(R, b.fields[5][1])
        seen, stack, edges, nfaces = set(), [b.fields[3][1]], [], 0
        while stack:
            i = stack.pop()
            if i < 0 or i in seen:
                continue
            seen.add(i)
            base = R[i].name.split("-")[0]
            if base not in _TOPO and base not in ("tcoedge", "tedge", "tvertex"):
                continue
            nfaces += R[i].name == "face"
            if base in ("edge", "tedge"):
                edges.append(i)
            stack.extend(r for r in R[i].refs() if r >= 0 and r not in seen and R[r].name != "body")
        polys = []
        for ei in edges:
            f = R[ei].fields
            v0, t0, v1, t1, cv = f[3][1], f[4][1], f[5][1], f[6][1], f[8][1]
            if v0 < 0 or v1 < 0:
                continue
            a, c1 = pt(v0), pt(v1)
            seg = np.array([a, c1])
            if cv >= 0 and R[cv].name == "ellipse-curve":
                cf = R[cv].fields
                c, n, maj, ratio = np.array(cf[3][1]), np.array(cf[4][1]), np.array(cf[5][1]), cf[6][1]
                if t1 < t0:
                    t1 += 2 * np.pi
                if abs(t1 - t0) < 1e-9 and np.allclose(a, c1):
                    t1 = t0 + 2 * np.pi
                ts = np.linspace(t0, t1, max(3, int(abs(t1 - t0) / arc_step) + 2))
                arc = c + np.outer(np.cos(ts), maj) + np.outer(np.sin(ts), np.cross(n, maj) * ratio)
                if np.linalg.norm(arc[0] - a) < 1e-3 and np.linalg.norm(arc[-1] - c1) < 1e-3:
                    seg = arc
            polys.append((np.c_[seg, np.ones(len(seg))] @ M.T)[:, :3] * unit_mm)
        P = np.vstack(polys) if polys else np.zeros((1, 3))
        out.append(dict(idx=b.idx, polys=polys, nfaces=nfaces, bmin=P.min(0), bmax=P.max(0)))
    return out


def load_bodies(f3z_path, design="Sofa Internal Structure"):
    d = designs(f3z_path)[design]
    smb = next(v for k, v in d.items() if k.endswith(".smb"))
    return bodies(parse_sab(smb))


# ─────────────────────────────────────────────────────────────────────────────
# CLI: catalogue + overlay against the regenerated model
# ─────────────────────────────────────────────────────────────────────────────

def _ref_to_project(p):
    """Fusion reference frame (x length centred, y up, z depth +front) ->
    project frame (X 0..L, Y depth from front, Z height) at the reference's
    own size (L 2060, front z 423) with 100 mm legs."""
    return np.c_[p[:, 0] + 1030, 423 - p[:, 2], p[:, 1] + 100]


def main():
    import argparse
    import csv
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("f3z")
    ap.add_argument("--design", default="Sofa Internal Structure")
    ap.add_argument("--out", default="outputs/reference")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    B = load_bodies(a.f3z, a.design)
    solids = [b for b in B if b["nfaces"] >= 4 and min(b["bmax"] - b["bmin"]) > 0.8 and b["bmin"][1] > -50]
    with open(out / "reference_bodies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["body", "faces", "dx_mm", "dy_mm", "dz_mm", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"])
        for b in solids:
            q = _ref_to_project(np.array([b["bmin"], b["bmax"]]))
            lo, hi = q.min(0), q.max(0)
            w.writerow([b["idx"], b["nfaces"], *np.round(hi - lo, 1), *np.round(lo, 1), *np.round(hi, 1)])
    print(f"{len(B)} bodies, {len(solids)} solids -> {out / 'reference_bodies.csv'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import sofa_structure as ss
    parts = ss.build_structure(2060, 837, 793, "3_seater", arm_width=230, leg_height=100)
    ref = [_ref_to_project(q) for b in solids if (b["bmax"] - b["bmin"])[0] < 1900 for q in b["polys"]]
    gen = [p.verts[f + [f[0]]] for p in parts for f in p.faces if len(f) <= 12]
    fig, axs = plt.subplots(3, 1, figsize=(16, 20))
    for ax, (ij, title) in zip(axs, [((0, 2), "FRONT  (X length / Z height)"),
                                     ((1, 2), "SIDE  (Y depth from front / Z height)"),
                                     ((0, 1), "TOP  (X length / Y depth)")]):
        ax.add_collection(LineCollection([s[:, list(ij)] for s in ref], colors="red", linewidths=1.1, alpha=.55))
        ax.add_collection(LineCollection([s[:, list(ij)] for s in gen], colors="navy", linewidths=.35, alpha=.85))
        ax.plot([], [], color="red", label="industry reference (Fusion)")
        ax.plot([], [], color="navy", label="regenerated (sofa_structure.py)")
        ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title); ax.grid(alpha=.3); ax.legend(loc="upper right")
        if ij == (1, 2):
            ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(out / "reference_overlay.png", dpi=70)
    print(f"overlay -> {out / 'reference_overlay.png'}")


if __name__ == "__main__":
    main()
