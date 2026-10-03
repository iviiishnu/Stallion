"""
sofa_3d_generator.py — 3D CAD model of the sofa with all internal components.

The geometry comes from sofa_structure.py, which rebuilds the industry
reference model (data/cad/Sofa Internal Structure.f3z) parametrically for
1 / 2 / 3-seaters. Component groups match the reference's Fusion
components and data/fusion_mapping/fusion_component_map.csv:

    Wood Frame, Plywood, Handle Frame, Springs, Clips, Seat Belts,
    Back Rest Belts, Foam, Handle Foam, Fabric (+ Legs)

Coordinate system (mm), shared with cad_generator.py:
    X = length (0..L), Y = depth (0 = front), Z = height (0 = floor)

Output (outputs/cad/):
    {id}.step                 — CAD assembly, one component per group (Fusion 360 etc.)
    {id}.glb                  — assembled 3D model, glTF Y-up, metres
    {id}_exploded.glb         — layers lifted apart (fabric / foam / suspension / frame)
    {id}.stl                  — all solids merged, mm
    {id}_takeoff.csv / .json  — per-part and per-group quantities measured from the model
    {id}_*.png                — static previews

Usage:
    from cad_generator import SofaGeometry
    from sofa_3d_generator import Sofa3DGenerator

    geo = SofaGeometry(L=2100, W=900, H=850, bom=bom_dict)
    gen = Sofa3DGenerator(geo, request_id="web_xxx", sofa_type="3_seater")
    gen.export_all(output_dir)

    # or straight from dimensions
    gen = Sofa3DGenerator.from_dimensions(1550, 900, 850, sofa_type="2_seater")

CLI:
    python sofa_3d_generator.py --seats 2 --L 1550 --W 900 --H 850 --out ../outputs/cad
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import trimesh

import sofa_structure as ss
from step_writer import write_step


# Layer lift for the exploded view, as a fraction of overall height
EXPLODE = {
    ss.FABRIC: 0.95,
    ss.FOAM: 0.55, ss.HFOAM: 0.55,
    ss.SPRINGS: 0.25, ss.CLIPS: 0.25, ss.SEAT_BELTS: 0.25, ss.BACK_BELTS: 0.25,
}

# Our frame (X length, Y depth from front, Z up) -> glTF (Y up, +Z to viewer), mm -> m
_TO_GLTF = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]], float) * np.array([[1e-3], [1e-3], [1e-3], [1]])


class Sofa3DGenerator:
    """Builds and exports the 3D sofa model (shell + all internal components)."""

    EXTERNAL_GROUPS = {ss.FABRIC, ss.LEGS}
    INTERNAL_GROUPS = set(ss.GROUP_ORDER) - EXTERNAL_GROUPS

    def __init__(self, geo, request_id: str = "sofa_cad", sofa_type: str = "3_seater",
                 style: dict = None, chaise_length: float = None, chaise_side: str = "right"):
        """style: output of sofa_style.style_from_photo() — arm style, cushions,
        colours etc. taken from the customer's photo (None = industry look).
        chaise_length / chaise_side are used for L-shaped sofas."""
        self.geo = geo
        self.request_id = request_id
        self.sofa_type = sofa_type
        self.style = dict(style or {})
        self.is_l_shape = ss.is_l_shape(sofa_type)
        self.chaise_length = chaise_length
        self.chaise_side = chaise_side if chaise_side in ("left", "right") else "right"
        self.l_info = None
        self.params = ss.StructureParams.from_style(
            float(geo.L), float(geo.W), float(geo.H),
            ss.seats_from_type(None if self.is_l_shape else sofa_type, geo.L), self.style,
            arm_width=getattr(geo, "arm_width", None),
            leg_height=getattr(geo, "leg_height", None),
        )
        self._parts = None

    @classmethod
    def from_dimensions(cls, L, W, H, sofa_type="3_seater", request_id=None,
                        arm_width=None, leg_height=None, style=None,
                        chaise_length=None, chaise_side="right"):
        class _Geo:
            pass
        g = _Geo()
        g.L, g.W, g.H = L, W, H
        g.arm_width, g.leg_height = arm_width, leg_height
        return cls(g, request_id or f"sofa_{int(L)}x{int(W)}x{int(H)}", sofa_type,
                   style=style, chaise_length=chaise_length, chaise_side=chaise_side)

    # ─────────────────────────────────────────────────────────────────────
    # Geometry
    # ─────────────────────────────────────────────────────────────────────

    @property
    def parts(self) -> list:
        if self._parts is None:
            if self.is_l_shape:
                p = self.params
                self._parts, self.l_info = ss.build_l_shape(
                    p.L, p.W, p.H, self.chaise_length, self.chaise_side, self.style,
                    arm_width=p.arm_width, leg_height=p.leg_height)
                self.params = self.l_info["main"]
            else:
                self._parts = ss.SofaStructure(self.params).build()
        return self._parts

    def group_color(self, group):
        """Colour actually used for a group (photo colours override defaults)."""
        for p in self.parts:
            if p.group == group and "rgb" in p.meta:
                return p.meta["rgb"]
        return ss.GROUP_COLORS.get(group, (180, 180, 180, 255))

    def _mesh(self, part, offset=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
        m = trimesh.Trimesh(part.verts + np.asarray(offset), part.triangles(), process=False)
        rgba = part.meta.get("rgb") or ss.GROUP_COLORS.get(part.group, (180, 180, 180, 255))
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=part.group.replace(" ", "_"),
            baseColorFactor=_srgb_to_linear(rgba),
            metallicFactor=0.6 if part.group in (ss.SPRINGS, ss.CLIPS) else 0.0,
            roughnessFactor=0.45 if part.group in (ss.SPRINGS, ss.CLIPS) else 0.85,
        ))
        return m

    def _offset(self, group, exploded):
        if not exploded:
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, EXPLODE.get(group, 0.0) * self.params.H)

    def _scene(self, exploded=False, groups=None) -> trimesh.Scene:
        scene = trimesh.Scene()
        root = self.request_id
        scene.graph.update(frame_to=root, frame_from=scene.graph.base_frame, matrix=_TO_GLTF)
        for g in ss.GROUP_ORDER:
            if groups is not None and g not in groups:
                continue
            gparts = [p for p in self.parts if p.group == g]
            if not gparts:
                continue
            gnode = g.replace(" ", "_")
            scene.graph.update(frame_to=gnode, frame_from=root)
            for p in gparts:
                scene.add_geometry(self._mesh(p, self._offset(g, exploded)),
                                   node_name=f"{gnode}/{p.name}", geom_name=f"{gnode}/{p.name}",
                                   parent_node_name=gnode)
        return scene

    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────

    def _path(self, output_dir, suffix):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{self.request_id}{suffix}"

    def export_glb(self, output_dir: str, exploded: bool = False) -> str:
        path = self._path(output_dir, "_exploded.glb" if exploded else ".glb")
        self._scene(exploded=exploded).export(str(path))
        return str(path)

    def export_stl(self, output_dir: str) -> str:
        path = self._path(output_dir, ".stl")
        trimesh.util.concatenate([self._mesh(p) for p in self.parts]).export(str(path))
        return str(path)

    def export_step(self, output_dir: str) -> str:
        path = self._path(output_dir, ".step")
        return write_step(self.parts, path, ss.GROUP_COLORS,
                          product_name=f"Stallion {self.sofa_type} {self.request_id}",
                          group_order=ss.GROUP_ORDER)

    def takeoff(self):
        return ss.takeoff(self.parts)

    def style_summary(self) -> dict:
        """The look actually built (after defaults / clamping)."""
        _ = self.parts
        p = self.params
        return {
            "arms": p.arms, "arm_style": p.arm_style,
            "seat_cushions": p.seat_cushions, "back_cushions": p.back_cushions,
            "leg_style": p.leg_style,
            "fabric_rgb": list(self.group_color(ss.FABRIC)[:3]),
            "leg_rgb": list(self.group_color(ss.LEGS)[:3]),
            "l_shape": bool(self.is_l_shape),
            **({"chaise_side": self.chaise_side, "chaise_depth_mm": self.l_info["chaise_depth"]}
               if self.l_info else {}),
        }

    def export_takeoff(self, output_dir: str) -> dict:
        rows, summary = self.takeoff()
        csv_path = self._path(output_dir, "_takeoff.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        json_path = self._path(output_dir, "_takeoff.json")
        p = self.params
        info = {
            "request_id": self.request_id,
            "sofa_type": self.sofa_type,
            "seats": p.seats,
            "dimensions_mm": {"length": p.L, "depth": p.W, "height": p.H},
            "derived_mm": {"arm_width": round(p.arm_width, 1), "seat_span": round(p.seat_span, 1),
                           "leg_height": round(p.leg_height, 1), "frame_height": round(p.frame_h, 1)},
            "style": self.style_summary(),
            "groups": summary,
        }
        if self.l_info:
            info["l_shape"] = {k: v for k, v in self.l_info.items() if k not in ("main", "chaise")}
        json_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        return {"takeoff_csv": str(csv_path), "takeoff_json": str(json_path)}

    def export_all(self, output_dir: str) -> dict:
        return {
            "glb": self.export_glb(output_dir),
            "glb_exploded": self.export_glb(output_dir, exploded=True),
            "stl": self.export_stl(output_dir),
            "step": self.export_step(output_dir),
            **self.export_takeoff(output_dir),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Static preview — small z-buffer rasteriser (correct occlusion, part
    # outlines), no GPU or extra dependencies needed
    # ─────────────────────────────────────────────────────────────────────

    def export_preview_png(self, output_dir: str, exploded: bool = True,
                           elev: float = 24, azim: float = -125,
                           groups: set = None, label: str = None,
                           size=(1500, 1000)) -> str:
        """groups: restrict the render to a subset (e.g. INTERNAL_GROUPS).
        azim is measured from +X towards +Y; -125 looks at the front-left."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        suffix = f"_{label}" if label else ("_exploded_preview" if exploded else "_preview")
        path = self._path(output_dir, f"{suffix}.png")

        sel = [p for p in self.parts if groups is None or p.group in groups]
        if not sel:
            raise ValueError("no parts selected for preview")
        tris, cols, ids = [], [], []
        for i, p in enumerate(sel):
            T = (p.verts + np.asarray(self._offset(p.group, exploded)))[p.triangles()]
            tris.append(T)
            rgb = p.meta.get("rgb") or ss.GROUP_COLORS[p.group]
            cols.append(np.tile(np.array(rgb[:3]) / 255, (len(T), 1)))
            ids.append(np.full(len(T), i + 1))
        img = _rasterize(np.concatenate(tris), np.concatenate(cols), np.concatenate(ids),
                         elev, azim, size, bg=(0.169, 0.184, 0.212))

        fig = plt.figure(figsize=(size[0] / 130, size[1] / 130 + 0.5), dpi=130)
        fig.patch.set_facecolor("#2b2f36")
        ax = fig.add_axes([0, 0, 1, 0.94])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ext = np.ptp(np.vstack([q.verts for q in self.parts]), axis=0)
        fig.suptitle(f"{self.sofa_type.replace('_', ' ').title()} — {'Exploded' if exploded else 'Assembled'}  "
                     f"({int(round(ext[0]))}×{int(round(ext[1]))}×{int(round(ext[2]))} mm)",
                     color="white", fontsize=12, y=0.985)
        present = [g for g in ss.GROUP_ORDER if any(q.group == g for q in sel)]
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(self.group_color(g)[:3]) / 255) for g in present]
        ax.legend(handles, present, loc="lower left", fontsize=8, facecolor="#3a3f47",
                  edgecolor="#555", labelcolor="white", ncol=2, framealpha=0.9)
        plt.savefig(str(path), facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(path)


def _srgb_to_linear(rgba):
    """glTF baseColorFactor is linear; photo / palette colours are sRGB."""
    out = []
    for c in rgba[:3]:
        c = c / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out + [(rgba[3] if len(rgba) > 3 else 255) / 255.0]


def _rasterize(T, C, ids, elev, azim, size, bg, ss_factor=2):
    """Flat-shaded z-buffer render of triangles T (N,3,3) with colours C (N,3).
    Returns an RGB float image. Part boundaries (id changes) are outlined."""
    W, H = size[0] * ss_factor, size[1] * ss_factor
    a, e = np.radians(azim), np.radians(elev)
    fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # camera looks along fwd
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    P = T.reshape(-1, 3)
    sx, sy, sz = P @ right, P @ up, P @ fwd
    lo_x, hi_x, lo_y, hi_y = sx.min(), sx.max(), sy.min(), sy.max()
    scale = 0.92 * min(W / (hi_x - lo_x), H / (hi_y - lo_y))
    px = (sx - (lo_x + hi_x) / 2) * scale + W / 2
    py = H / 2 - (sy - (lo_y + hi_y) / 2) * scale
    X, Y, Z = px.reshape(-1, 3), py.reshape(-1, 3), sz.reshape(-1, 3)

    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[(n @ fwd) > 0] *= -1                        # shade the visible side
    key = -fwd * 0.55 + up * 0.55 + right * -0.35
    key /= np.linalg.norm(key)
    shade = 0.38 + 0.52 * np.clip(n @ key, 0, 1) + 0.18 * np.clip(-(n @ fwd), 0, 1)
    shaded = np.clip(C * shade[:, None], 0, 1)

    zbuf = np.full((H, W), np.inf)
    cbuf = np.zeros((H, W, 3)); cbuf[:] = bg
    ibuf = np.zeros((H, W), dtype=np.int64)
    x0s = np.clip(np.floor(X.min(1)).astype(int), 0, W - 1)
    x1s = np.clip(np.ceil(X.max(1)).astype(int), 0, W - 1)
    y0s = np.clip(np.floor(Y.min(1)).astype(int), 0, H - 1)
    y1s = np.clip(np.ceil(Y.max(1)).astype(int), 0, H - 1)
    for t in range(len(T)):
        x0, x1, y0, y1 = x0s[t], x1s[t], y0s[t], y1s[t]
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (xa, xb, xc), (ya, yb, yc) = X[t], Y[t]
        den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(den) < 1e-12:
            continue
        w0 = ((yb - yc) * (gx - xc) + (xc - xb) * (gy - yc)) / den
        w1 = ((yc - ya) * (gx - xc) + (xa - xc) * (gy - yc)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        z = w0 * Z[t, 0] + w1 * Z[t, 1] + w2 * Z[t, 2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (z < sub)
        if win.any():
            sub[win] = z[win]
            cbuf[y0:y1 + 1, x0:x1 + 1][win] = shaded[t]
            ibuf[y0:y1 + 1, x0:x1 + 1][win] = ids[t]
    edge = np.zeros((H, W), bool)
    edge[:, 1:] |= ibuf[:, 1:] != ibuf[:, :-1]
    edge[1:, :] |= ibuf[1:, :] != ibuf[:-1, :]
    cbuf[edge] *= 0.35
    img = cbuf.reshape(size[1], ss_factor, size[0], ss_factor, 3).mean((1, 3))
    return img


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate a Stallion sofa 3D CAD model")
    ap.add_argument("--seats", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--l-shape", action="store_true", help="L-shaped sofa with chaise")
    ap.add_argument("--chaise-length", type=float, help="overall depth at the chaise end, mm")
    ap.add_argument("--chaise-side", default="right", choices=["left", "right"])
    ap.add_argument("--L", type=float, help="overall length mm (default: master template base)")
    ap.add_argument("--W", type=float, help="overall depth mm")
    ap.add_argument("--H", type=float, help="overall height mm")
    ap.add_argument("--arm-style", choices=list(ss.ARM_STYLES))
    ap.add_argument("--arms", choices=["both", "none"])
    ap.add_argument("--arm-top", type=float, help="arm top height as a fraction of H")
    ap.add_argument("--seat-cushions", type=int)
    ap.add_argument("--back-cushions", type=int)
    ap.add_argument("--legs", choices=["block", "tapered"])
    ap.add_argument("--fabric", help="fabric colour, e.g. '#2f3b55'")
    ap.add_argument("--photo", help="sofa photo: read the style from it (runs the YOLO detector)")
    ap.add_argument("--id", default=None, help="output file prefix")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "cad"))
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    style = {}
    if a.photo:
        import tempfile
        from sofa_validator import validate_sofa
        from sofa_style import style_from_photo, describe
        analysis = validate_sofa(a.photo, tempfile.mkdtemp())
        style = style_from_photo(a.photo, analysis.get("component_detections", []))
        print("photo style:", describe(style))
    for key, val in (("arm_style", a.arm_style), ("arms", a.arms), ("arm_top", a.arm_top),
                     ("seat_cushions", a.seat_cushions), ("back_cushions", a.back_cushions),
                     ("leg_style", a.legs)):
        if val is not None:
            style[key] = val
    if a.fabric:
        h = a.fabric.lstrip("#")
        style["fabric_rgb"] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    if a.l_shape:
        base, kind = (2700, 900, 850), "l_shape"
    else:
        base = {1: (850, 850, 850), 2: (1550, 900, 850), 3: (2100, 900, 850), 4: (2700, 950, 850)}[a.seats]
        kind = f"{a.seats}_seater"
    L, W, H = a.L or base[0], a.W or base[1], a.H or base[2]
    tag = "lshape" if a.l_shape else f"{a.seats}seater"
    gen = Sofa3DGenerator.from_dimensions(L, W, H, sofa_type=kind, style=style,
                                          chaise_length=a.chaise_length, chaise_side=a.chaise_side,
                                          request_id=a.id or f"{tag}_{int(L)}x{int(W)}x{int(H)}")
    files = gen.export_all(a.out)
    if not a.no_preview:
        files["preview_assembled"] = gen.export_preview_png(a.out, exploded=False, label="preview_assembled")
        files["preview_exploded"] = gen.export_preview_png(a.out, exploded=True, label="preview_exploded")
        files["preview_structure"] = gen.export_preview_png(a.out, exploded=False, label="preview_structure",
                                                            groups=Sofa3DGenerator.INTERNAL_GROUPS - {ss.FOAM, ss.HFOAM})
    print("style built:", gen.style_summary())
    for k, v in files.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
