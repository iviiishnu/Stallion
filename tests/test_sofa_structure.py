"""
Stallion 3D CAD structure tests
===============================
Checks the industry-derived parametric sofa (sofa_structure.py) across the
1 / 2 / 3-seater dimension ranges in data/master_template/master_dimensions.csv:

  - every part is a closed solid with positive volume
  - the model stays inside the requested L x W x H envelope
  - all component groups of the reference Fusion model are present
  - spring / belt arrays grow with seat count
  - STEP / GLB / take-off exports are complete

Run from the Stallion root directory:
    python tests/test_sofa_structure.py
"""

import csv
import io
import os
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import trimesh

import sofa_structure as ss
from sofa_3d_generator import Sofa3DGenerator

PASS = "  [PASS]"
FAIL = "  [FAIL]"
results = {"passed": 0, "failed": 0}


def check(label, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        results["passed"] += 1
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"         -> {e}")
        results["failed"] += 1


def sizes():
    with open(ROOT / "data" / "master_template" / "master_dimensions.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            v = row["variant"]
            if v not in ("1-seater", "2-seater", "3-seater"):
                continue
            n = int(v[0])
            for tag in ("min", "base", "max"):
                if tag == "base":
                    dims = (row["base_length"], row["base_width"], row["base_height"])
                else:
                    dims = (row[f"{tag}_length"], row[f"{tag}_width"], row[f"{tag}_height"])
                yield n, tag, tuple(float(d) for d in dims)


FUSION_GROUPS = []
with open(ROOT / "data" / "fusion_mapping" / "fusion_component_map.csv", newline="") as fh:
    FUSION_GROUPS = [r["component_group"] for r in csv.DictReader(fh)]


print("\n== Geometry across the master dimension ranges")
counts = {}
for n, tag, (L, W, H) in sizes():
    gen = Sofa3DGenerator.from_dimensions(L, W, H, sofa_type=f"{n}_seater")
    parts = gen.parts

    def closed_solids():
        bad = [p.name for p in parts
               if p.volume <= 0 or not trimesh.Trimesh(p.verts, p.triangles(), process=True).is_watertight]
        assert not bad, f"open / inverted parts: {bad[:5]}"

    def envelope():
        V = np.vstack([p.verts for p in parts])
        lo, hi = V.min(0), V.max(0)
        ext = hi - lo
        tol = 10.0                                     # 3 mm fabric skins + rounding
        assert lo[2] >= -1e-6, f"parts below the floor: z={lo[2]:.1f}"
        for axis, want in zip("LWH", (L, W, H)):
            got = ext["LWH".index(axis)]
            assert want - 60 <= got <= want + tol, f"{axis}: model {got:.0f} vs requested {want:.0f}"

    def groups():
        present = {p.group for p in parts}
        missing = [g for g in FUSION_GROUPS if g not in present]
        assert not missing, f"missing groups {missing}"

    check(f"{n}-seater {tag} {int(L)}x{int(W)}x{int(H)}: closed solids", closed_solids)
    check(f"{n}-seater {tag}: inside requested envelope", envelope)
    check(f"{n}-seater {tag}: all Fusion component groups present", groups)
    if tag == "base":
        counts[n] = {g: sum(p.group == g for p in parts) for g in (ss.SPRINGS, ss.BACK_BELTS, ss.CLIPS)}


print("\n== Arrays scale with seat count")


def arrays_grow():
    for g in (ss.SPRINGS, ss.BACK_BELTS):
        assert counts[1][g] < counts[2][g] < counts[3][g], f"{g}: {[counts[k][g] for k in (1, 2, 3)]}"
    for k in (1, 2, 3):
        assert counts[k][ss.CLIPS] == 3 * counts[k][ss.SPRINGS], "3 clips per spring"


check(f"springs / back belts grow 1->2->3 seats {[counts[k][ss.SPRINGS] for k in (1, 2, 3)]}", arrays_grow)


def reference_counts():
    # At the reference's own size the arrays must match the Fusion model:
    # 11 sinuous springs at 127 mm pitch (#127..#138 clips) and 2 seat belts.
    parts = ss.build_structure(2060, 837, 793, "3_seater", arm_width=230, leg_height=100)
    n_spr = sum(p.group == ss.SPRINGS for p in parts)
    n_sb = sum(p.group == ss.SEAT_BELTS for p in parts)
    assert (n_spr, n_sb) == (11, 2), (n_spr, n_sb)


check("reference-size 3-seater reproduces 11 springs / 2 seat belts", reference_counts)


print("\n== Photo-driven styles and L-shape")


def solid_and_inside(parts, L, D, H, tol=12.0):
    bad = [p.name for p in parts
           if p.volume <= 0 or not trimesh.Trimesh(p.verts, p.triangles(), process=True).is_watertight]
    assert not bad, f"open / inverted parts: {bad[:5]}"
    V = np.vstack([p.verts for p in parts])
    lo, hi = V.min(0), V.max(0)
    assert lo.min() >= -tol, f"parts outside the envelope (min {lo.round(1)})"
    for axis, want in zip("LDH", (L, D, H)):
        got = hi["LDH".index(axis)]
        assert got <= want + tol, f"{axis}: model reaches {got:.0f} vs {want:.0f}"


STYLES = [
    ("track arms, bench cushion", {"arm_style": "track", "seat_cushions": 1, "back_cushions": 1, "arm_top": 0.9}),
    ("rolled arms, 3+3 cushions", {"arm_style": "rolled", "seat_cushions": 3, "back_cushions": 3, "leg_style": "tapered"}),
    ("armless, loose cushions", {"arms": "none", "seat_cushions": 2, "back_cushions": 2}),
    ("sloped industry arm, low", {"arm_style": "sloped", "arm_top": 0.5}),
    ("photo colours", {"fabric_rgb": (40, 51, 66), "leg_rgb": (79, 61, 54), "leg_style": "tapered"}),
]
for n, (L, W, H) in ((1, (850, 850, 850)), (2, (1550, 900, 850)), (3, (2100, 900, 850))):
    for label, style in STYLES:
        def run(n=n, L=L, W=W, H=H, style=style):
            parts = ss.build_structure(L, W, H, f"{n}_seater", style=style)
            solid_and_inside(parts, L, W, H)
            if style.get("seat_cushions"):
                cores = [p for p in parts if p.name.startswith("seat_cushion") and p.name.endswith("_core")]
                assert len(cores) == style["seat_cushions"], len(cores)
            if style.get("arms") == "none":
                assert not any(p.group == ss.HFRAME for p in parts), "armless sofa has arm frames"
            if "fabric_rgb" in style:
                fab = [p for p in parts if p.group == ss.FABRIC]
                assert all(tuple(p.meta["rgb"][:3]) == style["fabric_rgb"] for p in fab), "fabric colour not applied"
        check(f"{n}-seater {label}", run)

for side in ("right", "left"):
    def l_shape(side=side):
        L, W, H, D = 2700, 900, 850, 1600
        parts, info = ss.build_l_shape(L, W, H, D, side, {"arm_style": "track", "seat_cushions": 3, "back_cushions": 3})
        solid_and_inside(parts, L, D, H)
        chaise = [p for p in parts if p.name.startswith("chaise_")]
        assert any(p.group == ss.SPRINGS for p in chaise), "chaise has no springs"
        cx = np.vstack([p.verts for p in chaise])[:, 0].mean()
        assert (cx > L / 2) == (side == "right"), f"chaise on the wrong side (x={cx:.0f})"
    check(f"L-shape, chaise {side}: closed, inside envelope, sprung chaise", l_shape)


def l_shape_export():
    g = Sofa3DGenerator.from_dimensions(2700, 900, 850, sofa_type="l_shape", request_id="tl",
                                        style={"seat_cushions": 3})
    out = g.export_all(tempfile.mkdtemp(prefix="stallion_l_"))
    txt = Path(out["step"]).read_text(encoding="ascii")
    assert txt.count("MANIFOLD_SOLID_BREP(") == len(g.parts)
    assert g.style_summary()["l_shape"] is True


check("L-shape exports through Sofa3DGenerator (sofa_type='l_shape')", l_shape_export)


print("\n== Exports")
tmp = tempfile.mkdtemp(prefix="stallion_cad_")
gen = Sofa3DGenerator.from_dimensions(1550, 900, 850, sofa_type="2_seater", request_id="t2")
files = {}


def export_all():
    files.update(gen.export_all(tmp))
    for k in ("glb", "glb_exploded", "stl", "step", "takeoff_csv", "takeoff_json"):
        assert os.path.getsize(files[k]) > 0, f"{k} empty"


def step_bodies():
    text = Path(files["step"]).read_text(encoding="ascii")
    n = text.count("MANIFOLD_SOLID_BREP(")
    assert n == len(gen.parts), f"{n} solids in STEP vs {len(gen.parts)} parts"
    for g in FUSION_GROUPS:
        assert f"PRODUCT('{g}'" in text, f"component {g} missing from STEP assembly"


def glb_nodes():
    scene = trimesh.load(files["glb"])
    assert len(scene.geometry) == len(gen.parts), f"{len(scene.geometry)} meshes vs {len(gen.parts)} parts"


check("export_all writes glb / exploded glb / stl / step / take-off", export_all)
check("STEP has one named body per part and one component per group", step_bodies)
check("GLB has one mesh per part", glb_nodes)

print(f"\n{results['passed']} passed, {results['failed']} failed")
sys.exit(1 if results["failed"] else 0)
