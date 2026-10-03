"""
sofa_structure.py — Parametric sofa internal structure, rebuilt from the
industry reference CAD (data/cad/Sofa Internal Structure.f3z).

The reference is a 3-seater modelled in Fusion 360. Every body in it was
extracted with f3z_reader.py and measured; the numbers below (all mm) are
those measurements. "#NNN" comments name the Fusion BREP body each value
came from, so the template can be audited against the original file.

How the reference is turned into 1 / 2 / 3-seaters
--------------------------------------------------
  * Timber / plywood cross-sections (25x40, 25x65, 45x25, 12 mm ply ...) are
    real stock sizes, so they never scale.
  * Spans, heights and depths scale with the requested L / W / H.
  * Repeated parts (sinuous springs, back-rest belts, centre supports) are
    re-arrayed from the reference pitch (127 mm springs, 107 mm back belts)
    so a 1-seater gets fewer springs instead of fatter ones.

Reference frame used while building (the Fusion file's own axes):
    x = length, centred on 0
    y = height, 0 = underside of the frame (legs are added below it)
    z = depth,  + = front of the sofa, - = back
At the end every part is mapped to the project convention shared with
cad_generator.py:  X = length (0..L), Y = depth (0 = front), Z = height
(0 = floor).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


# ── Reference 3-seater (measured) ──────────────────────────────────────────
REF_SEAT_SPAN = 1600.0      # between the two 12 mm side panels (#417)
REF_ARM_FRAME = 203.0       # arm frame width, end board #204
REF_ARM_GAP = 13.0          # side panel -> arm shell clearance (#13 vs #417)
REF_ARM_SKIN = 7.0          # arm fabric outside the arm frame (#225)
REF_FRAME_H = 693.0         # underside of frame -> top of back (#1)
REF_Z_BACK = -414.0         # rear face of the upholstered body (#1)
REF_Z_FRONT = 423.0         # front face of the upholstered body (#1)
REF_DEPTH = REF_Z_FRONT - REF_Z_BACK          # 837
REF_ARM_H = 332.0           # arm frame height at the outer edge (#204)

SPRING_PITCH = 127.0        # sinuous spring spacing (#342, #139 ...)
SPRING_MARGIN = 165.0       # side panel -> first spring centre
BACK_BELT_PITCH = 107.0     # back-rest belt spacing (#150, #151 ...)
BACK_BELT_MARGIN = 159.0
MAX_SUPPORT_BAY = 850.0     # reference 1600 span has one centre support (#380, #393)

# Groups — names match data/fusion_mapping/fusion_component_map.csv
WOOD, PLY, FOAM, HFOAM, FABRIC = "Wood Frame", "Plywood", "Foam", "Handle Foam", "Fabric"
SPRINGS, CLIPS, SEAT_BELTS, BACK_BELTS, HFRAME = (
    "Springs", "Clips", "Seat Belts", "Back Rest Belts", "Handle Frame")
LEGS = "Legs"

GROUP_ORDER = [WOOD, PLY, HFRAME, SPRINGS, CLIPS, SEAT_BELTS, BACK_BELTS,
               FOAM, HFOAM, FABRIC, LEGS]

# Display colours (RGBA) — close to the reference render: olive fabric,
# pink foam, warm timber, dark webbing.
GROUP_COLORS = {
    WOOD:       (214, 168, 108, 255),
    PLY:        (176, 124, 72, 255),
    HFRAME:     (196, 146, 84, 255),
    SPRINGS:    (160, 166, 172, 255),
    CLIPS:      (225, 225, 230, 255),
    SEAT_BELTS: (40, 40, 44, 255),
    BACK_BELTS: (62, 62, 70, 255),
    FOAM:       (230, 136, 168, 255),
    HFOAM:      (240, 170, 120, 255),
    FABRIC:     (94, 115, 62, 255),
    LEGS:       (60, 40, 25, 255),
}

# Nominal densities for the take-off (kg/m^3)
DENSITY = {FOAM: 32.0, HFOAM: 28.0}


# ─────────────────────────────────────────────────────────────────────────────
# Polyhedral part
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Part:
    """A closed polyhedron. Faces are planar polygons, CCW seen from outside."""
    name: str
    group: str
    verts: np.ndarray
    faces: list
    meta: dict = field(default_factory=dict)

    def triangles(self) -> np.ndarray:
        tris = []
        for f in self.faces:
            if len(f) == 3:
                tris.append(f)
            else:
                tris.extend(_triangulate_face(self.verts, f))
        return np.array(tris, dtype=np.int64)

    @property
    def volume(self) -> float:
        v = self.verts
        t = self.triangles()
        return float(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0)

    @property
    def area(self) -> float:
        v = self.verts
        t = self.triangles()
        return float(np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1).sum() / 2.0)

    def bounds(self):
        return self.verts.min(0), self.verts.max(0)


def _fix_orientation(verts, faces):
    p = Part("", "", verts, faces)
    if p.volume < 0:
        faces = [list(reversed(f)) for f in faces]
    return faces


def _poly_area2(P):
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _triangulate_face(V, f):
    """Ear-clip a planar polygon face (indices into V). Keeps winding."""
    pts = V[f]
    n = np.zeros(3)
    for i in range(len(pts)):          # Newell normal
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    ax = int(np.argmax(np.abs(n)))
    keep = [i for i in range(3) if i != ax]
    P2 = pts[:, keep]
    if n[ax] < 0:
        P2 = P2[:, ::-1]
    idx = list(range(len(f)))
    out = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = P2[i0], P2[i1], P2[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-9:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = P2[j]
                d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
                d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
                if d1 > 1e-9 and d2 > 1e-9 and d3 > 1e-9:
                    inside = True
                    break
            if not inside:
                out.append([f[i0], f[i1], f[i2]])
                idx.pop(k)
                break
        else:                           # degenerate — fall back to a fan
            break
    if len(idx) >= 3:
        for k in range(1, len(idx) - 1):
            out.append([f[idx[0]], f[idx[k]], f[idx[k + 1]]])
    return out


# ── Primitive builders (reference frame: x length, y height, z depth) ───────

def box(name, group, x0, x1, y0, y1, z0, z1, **meta):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    meta.setdefault("section", "x".join(str(int(round(s))) for s in sorted([x1 - x0, y1 - y0, z1 - z0])[:2]))
    meta.setdefault("length", round(max(x1 - x0, y1 - y0, z1 - z0), 1))
    return Part(name, group, V, _fix_orientation(V, F), meta)


def prism(name, group, profile, axis, a0, a1, **meta):
    """Extrude a 2D profile along `axis`. Profile coordinates are the two
    remaining axes in (x,y,z) order with the extrusion axis removed, except
    for axis 0 where the profile is given as (z, y) to match side views."""
    P = np.asarray(profile, float)
    if abs(_poly_area2(P)) < 1e-6:
        raise ValueError(f"degenerate profile for {name}")

    def lift(p, a):
        if axis == 0:
            return [a, p[1], p[0]]      # (z, y) profile
        if axis == 1:
            return [p[0], a, p[1]]      # (x, z)
        return [p[0], p[1], a]          # (x, y)
    n = len(P)
    V = np.array([lift(p, a0) for p in P] + [lift(p, a1) for p in P], float)
    F = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, n + j, n + i])
    meta.setdefault("length", round(abs(a1 - a0), 1))
    return Part(name, group, V, _fix_orientation(V, F), meta)


def tube(name, group, path, radius, sides=6, **meta):
    """Polyhedral wire swept along a path lying in a plane normal to y.
    Segments are mitred at the bisector planes, so every side face is a
    planar quad (keeps STEP output compact and exact)."""
    path = np.asarray(path, float)
    up = np.array([0.0, 1.0, 0.0])
    d = np.diff(path, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ang = [2 * math.pi * (k + 0.5) / sides for k in range(sides)]
    rings = []
    for i in range(len(path)):
        da = d[max(i - 1, 0)]
        db = d[min(i, len(d) - 1)]
        m = da + db
        m /= np.linalg.norm(m)
        na = np.cross(da, up)
        ring = []
        for th in ang:
            off = radius * (math.cos(th) * na + math.sin(th) * up)
            t = -np.dot(off, m) / np.dot(da, m)
            ring.append(path[i] + off + t * da)
        rings.append(ring)
    V = np.array([p for r in rings for p in r], float)
    F = []
    for i in range(len(rings) - 1):
        for k in range(sides):
            a, b = i * sides + k, i * sides + (k + 1) % sides
            F.append([a, b, b + sides, a + sides])
    F.append(list(range(sides - 1, -1, -1)))
    last = (len(rings) - 1) * sides
    F.append(list(range(last, last + sides)))
    wire_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    meta.setdefault("length", round(wire_len, 1))
    return Part(name, group, V, _fix_orientation(V, F), meta)


def frustum(name, group, cx, cz, top_w, bot_w, y0, y1, **meta):
    """Square tapered leg: top_w square at y1, bot_w square at y0."""
    t, b = top_w / 2, bot_w / 2
    V = np.array([[cx - b, y0, cz - b], [cx + b, y0, cz - b], [cx + b, y0, cz + b], [cx - b, y0, cz + b],
                  [cx - t, y1, cz - t], [cx + t, y1, cz - t], [cx + t, y1, cz + t], [cx - t, y1, cz + t]], float)
    F = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    meta.setdefault("length", round(abs(y1 - y0), 1))
    return Part(name, group, V, _fix_orientation(V, F), meta)


def _convex_hull(pts):
    P = sorted(set((round(x, 6), round(y, 6)) for x, y in pts))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def offset_polygon(P, t):
    """Offset a simple polygon outward by t (miter joins, clamped)."""
    P = np.asarray(P, float)
    if _poly_area2(P) < 0:
        return offset_polygon(P[::-1], t)[::-1]
    n = len(P)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = b - a, c - b
        n1 = np.array([e1[1], -e1[0]]) / (np.linalg.norm(e1) or 1)
        n2 = np.array([e2[1], -e2[0]]) / (np.linalg.norm(e2) or 1)
        m = n1 + n2
        ml = np.linalg.norm(m)
        if ml < 1e-9:
            out.append(b + n1 * t)
            continue
        m /= ml
        cosh = max(np.dot(m, n1), 0.35)
        out.append(b + m * (t / cosh))
    return np.array(out)


def skin(name, group, profile, axis, a0, a1, t=3.0, open_edges=(), caps=True):
    """Thin upholstery skin around a prism: one plate per profile edge plus
    end caps. `open_edges` lists edge indices left uncovered (e.g. floor)."""
    P = np.asarray(profile, float)
    if _poly_area2(P) < 0:
        P = P[::-1]
        n = len(P)
        open_edges = {(n - 2 - i) % n for i in open_edges}
    Q = offset_polygon(P, t)
    parts = []
    n = len(P)
    for i in range(n):
        if i in open_edges:
            continue
        j = (i + 1) % n
        quad = [P[i], P[j], Q[j], Q[i]]
        if abs(_poly_area2(quad)) < 1e-3:
            continue
        parts.append(prism(f"{name}_panel{i}", group, quad, axis, a0, a1))
    if caps:
        parts.append(prism(f"{name}_end_a", group, Q, axis, a0 - t, a0))
        parts.append(prism(f"{name}_end_b", group, Q, axis, a1, a1 + t))
    for p in parts:
        p.meta["sheet_thickness"] = t
    return parts


def sinuous_path(xc, y, z0, z1, width=35.0, wire=4.0, bend_r=13.0):
    """Zig-zag (sinuous / no-sag) spring centre-line running along z."""
    a = width / 2 - wire / 2
    n_straight = max(3, int(round((z1 - z0) / (2 * bend_r))) + 1)
    rb = (z1 - z0) / (2 * (n_straight - 1))
    pts = []
    side = -1
    for k in range(n_straight):
        z = z0 + 2 * rb * k
        x_from, x_to = xc + side * (a - rb), xc - side * (a - rb)
        pts.append([x_from, y, z])
        pts.append([x_to, y, z])
        if k < n_straight - 1:
            cx, cz = x_to, z + rb
            for th in np.linspace(-math.pi / 2, math.pi / 2, 5)[1:-1]:
                pts.append([cx - side * rb * math.cos(th), y, cz + rb * math.sin(th)])
        side = -side
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

def seats_from_type(sofa_type, length_mm=None) -> int:
    m = re.search(r"(\d)", str(sofa_type or ""))
    if m:
        return max(1, min(int(m.group(1)), 4))
    if length_mm:
        return 1 if length_mm < 1300 else 2 if length_mm < 1800 else 3
    return 3


ARM_STYLES = ("sloped", "track", "rolled")
REF_ARM_TOP = 416.0          # top of the arm upholstery above the frame underside (#13 + pad)


@dataclass
class StructureParams:
    L: float                    # overall length (mm)
    W: float                    # overall depth (mm)
    H: float                    # overall height incl. legs (mm)
    seats: int = 3
    arm_width: float = None     # overall arm width incl. clearance + fabric
    leg_height: float = None
    # ── look (from the customer's photo, see sofa_style.py) ──
    arms: str = "both"          # both | left | right | none   (left = X near 0)
    arm_style: str = "sloped"   # sloped (industry) | track | rolled
    arm_top: float = None       # arm top height / H; None = industry proportion
    back: bool = True           # False for a chaise extension
    seat_cushions: int = 0      # loose seat cushions (0 = tight seat, as the reference)
    back_cushions: int = 0
    seat_top: float = None      # seat cushion top / H (sets cushion thickness)
    leg_style: str = "block"    # block | tapered
    fabric_rgb: tuple = None
    leg_rgb: tuple = None
    arm_front_ext: float = 0.0  # extra arm length in front (L-shape chaise arm)
    arm_ext_side: str = None    # which arm gets arm_front_ext: "left" | "right"
    legs: str = "corners"       # corners | front

    def __post_init__(self):
        if self.arm_width is None:
            self.arm_width = self.L * 0.11                      # 230 on the 2060 reference
        self.arm_width = float(np.clip(self.arm_width, 120.0, 280.0))
        if self.leg_height is None:
            self.leg_height = self.H * 0.12
        self.leg_height = float(np.clip(self.leg_height, 40.0, 200.0))
        if self.arms not in ("both", "left", "right", "none"):
            self.arms = "both"
        if self.arm_style not in ARM_STYLES:
            self.arm_style = "sloped"
        self.seat_cushions = int(np.clip(self.seat_cushions or 0, 0, 6))
        self.back_cushions = int(np.clip(self.back_cushions or 0, 0, 6))

    @property
    def arm_sides(self):
        return {"both": (-1, 1), "left": (-1,), "right": (1,), "none": ()}[self.arms]

    @property
    def seat_span(self):
        return self.L - len(self.arm_sides) * self.arm_width

    @classmethod
    def from_style(cls, L, W, H, seats, style=None, **kw):
        """Build params from sofa_style.style_from_photo() output."""
        s = dict(style or {})
        keys = ("arms", "arm_style", "arm_top", "seat_cushions", "back_cushions",
                "seat_top", "leg_style", "fabric_rgb", "leg_rgb")
        args = {k: s[k] for k in keys if s.get(k) is not None}
        args.update({k: v for k, v in kw.items() if v is not None})
        return cls(L=L, W=W, H=H, seats=seats, **args)

    @property
    def frame_h(self):
        return self.H - self.leg_height


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

class SofaStructure:
    """Builds every part of the sofa in the reference frame, then maps it to
    the project frame (X length, Y depth from front, Z height).

    Internally the seat is centred on x = 0; `self.cx` is where that centre
    sits inside the overall length (non-zero when only one arm exists)."""

    def __init__(self, params: StructureParams):
        self.p = params
        self.S = params.seat_span
        if self.S < 400:
            raise ValueError(f"seat span {self.S:.0f} mm too small — check L and arm width")
        self.hs = self.S / 2
        sides = params.arm_sides
        left = -params.L / 2 + (params.arm_width if -1 in sides else 0)
        right = params.L / 2 - (params.arm_width if 1 in sides else 0)
        self.cx = (left + right) / 2
        self.ky = params.frame_h / REF_FRAME_H
        self.kz = params.W / REF_DEPTH
        # arm heights scale on their own so photo-measured arm height is honoured
        self.kya = self.ky
        if params.arm_top:
            target = params.arm_top * params.H - params.leg_height
            self.kya = float(np.clip(target / REF_ARM_TOP, 0.45 * self.ky, 0.98 * params.frame_h / REF_ARM_TOP))
        self.colors = dict(GROUP_COLORS)
        if params.fabric_rgb:
            self.colors[FABRIC] = tuple(int(c) for c in params.fabric_rgb[:3]) + (255,)
        if params.leg_rgb:
            self.colors[LEGS] = tuple(int(c) for c in params.leg_rgb[:3]) + (255,)
        self.parts: list[Part] = []

    # coordinate helpers: scale a reference position, keep stock sections
    def Y(self, v):
        return v * self.ky

    def Z(self, v):
        return v * self.kz

    def zy(self, pts):
        return [(z * self.kz, y * self.ky) for z, y in pts]

    def add(self, part_or_list):
        if isinstance(part_or_list, Part):
            self.parts.append(part_or_list)
        else:
            self.parts.extend(part_or_list)

    def xs_array(self, pitch, margin):
        m = min(margin, self.S * 0.1)
        usable = self.S - 2 * m
        n = max(2, int(round(usable / pitch)) + 1)
        return list(np.linspace(-self.hs + m, self.hs - m, n))

    def support_xs(self):
        bays = max(1, math.ceil(self.S / MAX_SUPPORT_BAY))
        return [(-self.hs + self.S * k / bays) for k in range(1, bays)]

    def side_profile(self):
        if self.p.back:
            return self.zy([(395, 0), (-395, 0), (-395, 680), (-330, 680),
                            (-220, 243), (-220, 153), (395, 153)])
        return self.zy([(395, 0), (-395, 0), (-395, 153), (395, 153)])

    # ── build ─────────────────────────────────────────────────────────────
    def build(self) -> list[Part]:
        self.parts = []
        self._seat_frame()
        if self.p.back:
            self._back_frame()
        self._supports()
        self._suspension()
        self._foam()
        for side in self.p.arm_sides:
            self._arm(side)
        self._fabric()
        self._cushions()
        self._legs()
        for part in self.parts:
            part.meta["rgb"] = self.colors.get(part.group, (180, 180, 180, 255))
        self._to_project_frame()
        return self.parts

    # Seat box: side panels + rails
    def _seat_frame(self):
        hs, Y, Z = self.hs, self.Y, self.Z
        xi0, xi1 = -hs + 12, hs - 12
        # 12 mm side panels with the reclined back cut-out (#417)
        side = self.side_profile()
        self.add(prism("side_panel_L", PLY, side, 0, -hs, -hs + 12, thickness=12))
        self.add(prism("side_panel_R", PLY, side, 0, hs - 12, hs, thickness=12))
        # Floor rails, 25 x 40 (#396, #397)
        self.add(box("rail_back_bottom", WOOD, xi0, xi1, 0, 25, Z(-395), Z(-395) + 40))
        self.add(box("rail_mid_bottom", WOOD, xi0, xi1, 0, 25, -20, 20))
        self.add(box("rail_front_bottom", WOOD, xi0, xi1, 0, 25, Z(395) - 25, Z(395)))
        # Spring rails 65 x 25: front (#400) and rear (carries the rear clips)
        self.add(box("rail_front_spring", WOOD, xi0, xi1, Y(153) - 65, Y(153), Z(370), Z(370) + 25))
        self.add(box("rail_rear_spring", WOOD, xi0, xi1, Y(153) - 65, Y(153), Z(-156), Z(-156) + 25))
        # Front fascia board 25 x 180 (#309) and top lip 19 x 37 (#305)
        # (the reference overlaps these two by 19 mm; the fascia stops under the lip here)
        self.add(box("front_fascia", PLY, -hs, hs, Y(84), Y(264) - 19, Z(396), Z(396) + 25, thickness=25))
        self.add(box("front_lip", WOOD, -hs, hs, Y(264) - 19, Y(264), Z(396) + 25 - 37, Z(396) + 25))

    # Reclined back: uprights, top rail, top boards, side bracing
    def _back_frame(self):
        hs, Y, Z = self.hs, self.Y, self.Z
        xi0, xi1 = -hs + 12, hs - 12
        upright = self.zy([(-229, 282), (-321, 651), (-345, 645), (-253, 276)])   # #388
        self.add(prism("back_upright_L", WOOD, upright, 0, xi0, xi0 + 40))
        self.add(prism("back_upright_R", WOOD, upright, 0, xi1 - 40, xi1))
        self.add(box("rail_back_top", WOOD, xi0, xi1, Y(680) - 25, Y(680), Z(-395), Z(-395) + 65))   # #410
        # 12 mm top boards (#298, #300)
        self.add(box("top_board_rear", PLY, -hs, hs, Y(681), Y(681) + 12, Z(-395), Z(-395) + 65, thickness=12))
        self.add(box("top_board_front", PLY, -hs, hs, Y(680), Y(680) + 12, Z(-324), Z(-324) + 82, thickness=12))
        # Side bracing tying the back to the seat box (#366, #365)
        for sx, (x0, x1) in (("L", (xi0, xi0 + 25)), ("R", (xi1 - 25, xi1))):
            self.add(box(f"back_brace_low_{sx}", WOOD, x0, x1, 0, 65, Z(-345), Z(-175)))
            self.add(box(f"back_brace_high_{sx}", WOOD, x0, x1, 65, Y(290), Z(-345), Z(-345) + 65))

    # Centre supports at seat boundaries (#393 runner, #384 post, #360 front
    # post, #380 back post, #377 upright)
    def _supports(self):
        Y, Z = self.Y, self.Z
        upright = self.zy([(-229, 282), (-321, 651), (-345, 645), (-253, 276)])
        for i, xs in enumerate(self.support_xs()):
            self.add(box(f"support{i}_runner", WOOD, xs - 32.5, xs + 32.5, 0, 25, Z(-355), Z(370)))
            self.add(box(f"support{i}_rear_post", WOOD, xs - 32.5, xs + 32.5, 25, Y(153) - 65, Z(-156), Z(-156) + 25))
            self.add(box(f"support{i}_front_post", WOOD, xs - 32.5, xs + 32.5, 25, Y(153) - 65, Z(370), Z(370) + 25))
            if self.p.back:
                self.add(box(f"support{i}_back_post", WOOD, xs - 12.5, xs + 12.5, 25, Y(680) - 25, Z(-395), Z(-395) + 65))
                self.add(prism(f"support{i}_back_upright", WOOD, upright, 0, xs - 20, xs + 20))

    # Springs, clips, seat belts, back-rest belts
    def _suspension(self):
        hs, Y, Z = self.hs, self.Y, self.Z
        y_spring = Y(160)
        z0, z1 = Z(-142), Z(387)                       # #342: 529 long
        for i, xc in enumerate(self.xs_array(SPRING_PITCH, SPRING_MARGIN)):
            self.add(tube(f"spring_{i + 1:02d}", SPRINGS, sinuous_path(xc, y_spring, z0, z1), 2.0,
                          wire_diameter=4))
            yt = Y(153)                                # clips sit on the spring rails
            self.add(box(f"clip_front_{i + 1:02d}", CLIPS, xc - 10, xc + 10, yt, yt + 10, Z(395) - 14, Z(395)))
            self.add(box(f"clip_front_plate_{i + 1:02d}", CLIPS, xc - 10, xc + 10, yt, yt + 1.5, Z(395) - 22, Z(395) - 14))
            self.add(box(f"clip_rear_{i + 1:02d}", CLIPS, xc - 10, xc + 10, yt, yt + 16, Z(-154), Z(-154) + 22))
        # Two 70 mm cross straps tying the springs (#148, #149)
        for i, zc in enumerate((Z(-23), Z(119))):
            self.add(box(f"seat_belt_{i + 1}", SEAT_BELTS, -hs + 14, hs - 14, Y(153), Y(153) + 2,
                         zc - 35, zc + 35, width=70))
        if not self.p.back:
            return
        # Vertical back-rest webbing, 50 wide, following the recline (#150)
        belt = self.zy([(-218, 255), (-322, 667), (-324, 667), (-220, 255)])
        for i, xc in enumerate(self.xs_array(BACK_BELT_PITCH, BACK_BELT_MARGIN)):
            self.add(prism(f"back_belt_{i + 1:02d}", BACK_BELTS, belt, 0, xc - 25, xc + 25, width=50))
        # Two horizontal back cross belts (#318, #317), woven just behind the
        # vertical webbing
        for i, (ya, yb) in enumerate(((448, 498), (543, 592))):
            za = -220.5 - (ya - 255) * (104 / 412)
            zb = -220.5 - (yb - 255) * (104 / 412)
            prof = self.zy([(za, ya), (zb, yb), (zb - 2, yb), (za - 2, ya)])
            self.add(prism(f"back_cross_belt_{i + 1}", BACK_BELTS, prof, 0, -hs + 25, hs - 25, width=50))

    def _foam(self):
        hs, Y, Z = self.hs, self.Y, self.Z
        # 51 mm seat foam over the springs (#312)
        z0 = Z(-220) if self.p.back else Z(-395)
        self.add(box("seat_foam", FOAM, -hs, hs, Y(162), Y(162) + 51, z0, Z(395), thickness=51))
        if self.p.back:
            # Reclined back foam with rolled top (#2 / #302)
            back = self.zy([(-137, 263), (-136, 264), (-241, 680), (-244, 693),
                            (-324, 693), (-321, 680), (-216, 263)])
            self.add(prism("back_foam", FOAM, back, 0, -hs, hs, thickness=79))

    # ── arm ("handle") ────────────────────────────────────────────────────
    # Reference is the right arm, frame 203 wide (#204). "sloped" is the
    # industry arm; "track" (square) and "rolled" keep the same rail set and
    # stock sections but change the top.
    def _arm_ext(self, side):
        want = "left" if side < 0 else "right"
        return self.p.arm_front_ext if self.p.arm_ext_side == want else 0.0

    def _arm(self, side):
        p = self.p
        Z = self.Z
        ky = self.kya
        af = p.arm_width - REF_ARM_GAP - REF_ARM_SKIN
        ka = af / REF_ARM_FRAME
        x_in = self.hs + REF_ARM_GAP
        tag = "R" if side > 0 else "L"
        ext = self._arm_ext(side)

        def X(u):                         # arm-local u -> reference x
            return side * (x_in + u)

        def U(u0, u1):                    # stock sections keep their size, spans scale;
            w = u1 - u0                   # anchor to the nearest frame edge
            if w > 70:
                w *= ka
            if u0 <= 0.5:
                return 0, w
            if u1 >= REF_ARM_FRAME - 0.5:
                return af - w, af
            c = (u0 + u1) / 2 * ka
            return c - w / 2, c + w / 2

        def V(y0, y1):
            h = y1 - y0
            if h > 70:
                h *= ky
            if y0 <= 0.5:
                return 0, h
            c = (y0 + y1) / 2 * ky
            return c - h / 2, c + h / 2

        def arm_prism(name, group, prof_uy, z0, z1, **meta):
            P = [(X(u * ka), y * ky) for u, y in prof_uy]
            if side < 0:
                P = P[::-1]
            return prism(name, group, P, 2, z0, z1, **meta)

        def arm_box(name, group, u, y, z0, z1, **meta):
            u0, u1 = U(*u)
            y0, y1 = V(*y)
            return box(name, group, X(u0), X(u1), y0, y1, z0, z1, **meta)

        zr0, zr1 = Z(-388), Z(383) + ext   # rail length 771 (+ chaise extension)
        style = p.arm_style
        if style == "track":
            end = [(0, 0), (203, 0), (203, 371), (0, 371)]
            upper_outer = (283, 328)
            top_board = [(0, 346), (203, 346), (203, 371), (0, 371)]
            pad = [(0, 371), (203, 371), (203, 411), (0, 411)]
        elif style == "rolled":
            end = [(0, 0), (203, 0), (203, 315), (0, 315)]
            upper_outer = (265, 290)
            top_board = [(0, 290), (203, 290), (203, 315), (0, 315)]
            pad = [(103 + 104 * math.cos(t), 335 + 75 * math.sin(t))
                   for t in np.radians(np.linspace(-15, 195, 16))]
        else:
            end = [(0, 230), (0, 0), (203, 0), (203, 332)]              # #204, #205
            upper_outer = (283, 328)
            top_board = [(12, 243), (9, 267), (174, 362), (176, 340)]   # #210 / #186
            pad = [(173, 364), (164, 411), (-6, 315), (9, 269)]         # #184
        # End boards, 12 mm plywood
        self.add(arm_prism(f"arm_{tag}_end_board_front", PLY, end, zr1, zr1 + 12, thickness=12))
        self.add(arm_prism(f"arm_{tag}_end_board_back", PLY, end, zr0 - 12, zr0, thickness=12))
        if style == "sloped":
            # Sloped top end plates (#213, #247)
            top = [(-6, 315), (7, 235), (190, 338), (164, 411)]
            self.add(arm_prism(f"arm_{tag}_top_plate_front", PLY, top, zr1, zr1 + 12, thickness=12))
            self.add(arm_prism(f"arm_{tag}_top_plate_back", PLY, top, zr0 - 12, zr0, thickness=12))
        # Long rails (#200, #201, #206, #207, #209)
        for nm, u, y in (("bottom_inner", (0, 25), (0, 45)),
                         ("bottom_outer", (178, 203), (0, 45)),
                         ("mid_inner", (0, 25), (92, 157)),
                         ("upper_inner", (0, 45), (205, 230)),
                         ("upper_outer", (178, 203), upper_outer)):
            self.add(arm_box(f"arm_{tag}_rail_{nm}", HFRAME, u, y, zr0, zr1))
        if style == "sloped":
            # Rotated top rail (#208) and outer cap rail (#224)
            self.add(arm_prism(f"arm_{tag}_rail_top_centre", HFRAME,
                               [(132, 270), (116, 289), (81, 260), (97, 241)], zr0, zr1))
            self.add(arm_prism(f"arm_{tag}_rail_cap", HFRAME,
                               [(194, 332), (203, 333), (175, 412), (165, 412)], Z(-400), Z(395) + ext))
        # Cross rails 153 x 25 x 65 (#202, #203)
        for nm, z0 in (("front", zr1 - 65), ("back", zr0)):
            self.add(arm_box(f"arm_{tag}_cross_{nm}", HFRAME, (25, 178), (0, 25), z0, z0 + 65))
        # Top board (25 thick) and handle foam pad
        self.add(arm_prism(f"arm_{tag}_top_board", HFRAME, top_board, Z(-387), Z(382) + ext))
        self.add(arm_prism(f"arm_{tag}_handle_foam", HFOAM, pad, zr0, Z(395) + ext, thickness=48))

    def _arm_outline(self):
        """Arm upholstery outline in (u, y) reference units for the arm style."""
        style = self.p.arm_style
        if style == "track":
            return [(-6, 0), (212, 0), (212, 404), (200, 416), (6, 416), (-6, 404)]
        if style == "rolled":
            pts = [(-4, 0), (210, 0), (210, 300), (-4, 300)]
            pts += [(103 + 108 * math.cos(t), 335 + 79 * math.sin(t))
                    for t in np.radians(np.linspace(-15, 195, 16))]
            return _convex_hull(pts)
        return [(-6, 0), (210, 0), (210, 340), (175, 416), (164, 416), (-10, 318), (-6, 236)]

    def _side_cover_profile(self):
        """Outline of the upholstered end of the seat box where there is no arm."""
        Y, Z = self.Y, self.Z
        zf = Z(396) + 25 + 3
        yd = Y(162) + 51 + 3
        yl = Y(264) + 3
        if self.p.back:
            return [(zf, 0), (Z(-403), 0), (Z(-403), Y(693) + 3), (Z(-244), Y(693) + 3),
                    (Z(-137) + 3, Y(263)), (Z(-137) + 3, yd), (Z(396) - 3, yd), (Z(396) - 3, yl), (zf, yl)]
        return [(zf, 0), (Z(-398), 0), (Z(-398), yd), (Z(396) - 3, yd), (Z(396) - 3, yl), (zf, yl)]

    # Upholstery skins (3 mm) — back wrap (#289), arm covers (#13), seat deck
    def _fabric(self):
        hs, Y, Z = self.hs, self.Y, self.Z
        af = self.p.arm_width - REF_ARM_GAP - REF_ARM_SKIN
        ka = af / REF_ARM_FRAME
        # carry the covers across the side-panel-to-arm clearance so no frame shows
        gx0 = -hs - (REF_ARM_GAP if -1 in self.p.arm_sides else 0)
        gx1 = hs + (REF_ARM_GAP if 1 in self.p.arm_sides else 0)
        if self.p.back:
            # Outer back wrap: 159 deep, full height, open at the floor
            back = self.zy([(-245, 0), (-245, 692), (-400, 692), (-400, 0)])
            self.add(skin("fabric_back_wrap", FABRIC, back, 0, gx0, gx1, open_edges=(3,)))
            # Back-rest cover wrapped round the back foam, open underneath
            back_foam = self.zy([(-137, 263), (-136, 264), (-241, 680), (-244, 693),
                                 (-324, 693), (-321, 680), (-216, 263)])
            self.add(skin("fabric_back_rest", FABRIC, back_foam, 0, -hs + 1, hs - 1, open_edges=(6,)))
        # Seat deck: over the seat foam, over the front lip, down the fascia
        zs = Z(-220) - 3 if self.p.back else Z(-395)
        zl, zf = Z(396), Z(396) + 25
        ys, yl, yb = Y(162) + 51, Y(264), 0.0            # front cover runs to the floor
        deck = [(zs, ys + 3), (zl - 3, ys + 3), (zl - 3, yl + 3), (zf + 3, yl + 3), (zf + 3, yb),
                (zf, yb), (zf, yl), (zl, yl), (zl, ys), (zs, ys)]
        self.add(prism("fabric_seat_deck", FABRIC, deck, 0, gx0, gx1, sheet_thickness=3))
        # Arm covers: skin around the arm outline, open at the floor
        outline = self._arm_outline()
        for side in (-1, 1):
            tag = "R" if side > 0 else "L"
            if side not in self.p.arm_sides:
                # no arm on this side: upholster the exposed side panel
                sp = self._side_cover_profile()
                x0 = hs + 0.5 if side > 0 else -hs - 3.5     # clear of the back wrap's end cap
                self.add(prism(f"fabric_side_{tag}", FABRIC, sp, 0, x0, x0 + 3, sheet_thickness=3))
                continue
            x_in = self.hs + REF_ARM_GAP
            P = [(side * (x_in + u * ka), y * self.kya) for u, y in outline]
            n = len(P)
            floor = [i for i in range(n) if P[i][1] <= 1 and P[(i + 1) % n][1] <= 1]
            if side < 0:
                P = P[::-1]
                floor = [(n - 2 - i) % n for i in floor]
            self.add(skin(f"fabric_arm_{tag}", FABRIC, P, 2,
                          Z(-406), Z(401) + self._arm_ext(side), open_edges=tuple(floor)))

    # Loose cushions (from the photo) — foam core wrapped in fabric
    def _cushions(self):
        p = self.p
        if not (p.seat_cushions or (p.back_cushions and p.back)):
            return
        hs, Y, Z = self.hs, self.Y, self.Z
        yd = Y(213) + 3                                   # top of the seat deck fabric
        tc = 110.0
        if p.seat_top:
            tc = float(np.clip(p.seat_top * p.H - p.leg_height - yd, 60, 200))
        tb = 150.0 if (p.back_cushions and p.back) else 0.0
        A = np.array([Z(-137) + 3, Y(263)])               # front face of back foam (z, y)
        B = np.array([Z(-241) + 3, Y(680)])
        d = (B - A) / np.linalg.norm(B - A)
        nrm = np.array([d[1], -d[0]])                     # points forward / slightly up

        def on_back(y):
            return A + (y - A[1]) / (B[1] - A[1]) * (B - A)

        def split(n):
            gap = 6.0
            w = (self.S - gap * (n + 1)) / n
            return [(-hs + gap + k * (w + gap), -hs + gap + k * (w + gap) + w) for k in range(n)]

        def cushion(name, core, x0, x1):
            self.add(prism(f"{name}_core", FOAM, core, 0, x0 + 3, x1 - 3))
            self.add(skin(f"{name}_cover", FABRIC, core, 0, x0 + 3, x1 - 3))

        if p.seat_cushions:
            zr = on_back(yd)[0] + tb if p.back else Z(-395)
            zfr = Z(396) + 25 - 5                         # just behind the front cover
            c = 15.0
            y0, y1 = yd + 3, yd + tc - 3
            z0, z1 = zr + 3, zfr - 3
            core = [(z0 + c, y0), (z1 - c, y0), (z1, y0 + c), (z1, y1 - c),
                    (z1 - c, y1), (z0 + c, y1), (z0, y1 - c), (z0, y0 + c)]
            for i, (x0, x1) in enumerate(split(p.seat_cushions)):
                cushion(f"seat_cushion_{i + 1}", core, x0, x1)
        if p.back_cushions and p.back:
            y0 = yd + (tc if p.seat_cushions else 0)
            y1 = Y(660)
            if y1 - y0 > 150:
                a, b = on_back(y0) + 3 * nrm, on_back(y1) + 3 * nrm
                core = [tuple(a), tuple(b), tuple(b + (tb - 6) * nrm), tuple(a + (tb - 6) * nrm)]
                for i, (x0, x1) in enumerate(split(p.back_cushions)):
                    cushion(f"back_cushion_{i + 1}", core, x0, x1)

    def _legs(self):
        p = self.p
        lh = p.leg_height
        left, right = -p.L / 2 - self.cx, p.L / 2 - self.cx
        xs = [left + 50, right - 50] + ([(left + right) / 2] if p.L > 2400 else [])
        spots = [(x, self.Z(370)) for x in xs]
        if p.legs == "corners":
            spots += [(x, self.Z(-370)) for x in xs]
        for side in p.arm_sides:
            ext = self._arm_ext(side)
            if ext:
                spots.append((left + 50 if side < 0 else right - 50, self.Z(370) + ext))
        for x, zc in spots:
            name = f"leg_{'F' if zc > 0 else 'B'}_{int(round(x + self.cx + p.L / 2))}"
            if p.leg_style == "tapered":
                self.add(frustum(name, LEGS, x, zc, 50, 30, -lh, 0))
            else:
                self.add(box(name, LEGS, x - 22.5, x + 22.5, -lh, 0, zc - 22.5, zc + 22.5))

    def _to_project_frame(self):
        """(x, y, z)_ref -> (X length 0..L, Y depth 0=front, Z height 0=floor)."""
        z_front = self.Z(REF_Z_FRONT)
        for part in self.parts:
            v = part.verts
            part.verts = np.c_[v[:, 0] + self.cx + self.p.L / 2, z_front - v[:, 2], v[:, 1] + self.p.leg_height]
            # the map is a proper rotation, so face winding is preserved


# ─────────────────────────────────────────────────────────────────────────────
# L-shape: main sofa + chaise extension + full-length chaise arm
# ─────────────────────────────────────────────────────────────────────────────

def build_l_shape(L, W, H, chaise_length=None, side="right", style=None,
                  arm_width=None, leg_height=None) -> tuple[list[Part], dict]:
    """L-shaped sofa (sofa + chaise). L = overall length, W = depth of the
    main back section, chaise_length = overall depth at the chaise end."""
    style = dict(style or {})
    D = float(chaise_length) if chaise_length else float(np.clip(1.75 * W, W + 450, 2200))
    E = max(D - W, 400.0)
    seats = seats_from_type(None, L)
    arms = "none" if style.get("arms") == "none" else "both"
    main_p = StructureParams.from_style(L, W, H, seats, style, arm_width=arm_width, leg_height=leg_height,
                                        arms=arms, arm_front_ext=E, arm_ext_side="right")
    main = SofaStructure(main_p).build()

    a_eff = main_p.arm_width if arms == "both" else 0.0
    Wc = float(np.clip(0.3 * L, 650, 1000))
    ch_style = dict(style, back_cushions=0, seat_cushions=1 if style.get("seat_cushions") else 0)
    ch_p = StructureParams.from_style(Wc, E, H, 1, ch_style, leg_height=main_p.leg_height,
                                      arms="none", back=False, legs="front")
    chaise = SofaStructure(ch_p).build()

    x0 = L - a_eff - Wc
    for part in chaise:
        part.name = f"chaise_{part.name}"
        part.verts = part.verts + [x0, 0.0, 0.0]
    for part in main:
        part.verts = part.verts + [0.0, E, 0.0]
    parts = main + chaise
    if side == "left":
        for part in parts:
            part.verts = part.verts * [-1, 1, 1] + [L, 0, 0]
            part.faces = [list(reversed(f)) for f in part.faces]
    info = {"chaise_depth": round(D, 1), "chaise_width": round(Wc, 1), "chaise_side": side,
            "main": main_p, "chaise": ch_p}
    return parts, info


# ─────────────────────────────────────────────────────────────────────────────
# Material take-off
# ─────────────────────────────────────────────────────────────────────────────

CUFT_PER_M3 = 35.3147


def takeoff(parts: list[Part]) -> tuple[list[dict], list[dict]]:
    """Per-part rows and per-group summary computed from the geometry."""
    rows = []
    for p in parts:
        mn, mx = p.bounds()
        vol_m3 = abs(p.volume) / 1e9
        rows.append({
            "group": p.group,
            "part": p.name,
            "dx_mm": round(mx[0] - mn[0], 1),
            "dy_mm": round(mx[1] - mn[1], 1),
            "dz_mm": round(mx[2] - mn[2], 1),
            "section": p.meta.get("section", ""),
            "length_mm": p.meta.get("length", ""),
            "volume_m3": round(vol_m3, 6),
            "surface_m2": round(p.area / 1e6, 4),
        })
    summary = []
    for g in GROUP_ORDER:
        gp = [p for p in parts if p.group == g]
        if not gp:
            continue
        vol = sum(abs(p.volume) for p in gp) / 1e9
        item = {"group": g, "pieces": len(gp), "volume_m3": round(vol, 5)}
        if g in (WOOD, HFRAME):
            item.update(qty=round(vol * CUFT_PER_M3, 3), unit="cubic feet")
        elif g == PLY:
            sheet = sum(abs(p.volume) / p.meta.get("thickness", 12) for p in gp) / 1e6
            item.update(qty=round(sheet, 3), unit="square meters")
        elif g in (FOAM, HFOAM):
            item.update(qty=round(vol * DENSITY[g], 2), unit="kg")
        elif g == FABRIC:
            sheet = sum(abs(p.volume) / p.meta.get("sheet_thickness", 3) for p in gp) / 1e6
            item.update(qty=round(sheet, 2), unit="square meters")
        elif g == SPRINGS:
            wire = sum(float(p.meta.get("length", 0)) for p in gp) / 1000
            item.update(qty=len(gp), unit="pieces", wire_length_m=round(wire, 2))
        elif g in (SEAT_BELTS, BACK_BELTS):
            run = sum(float(p.meta.get("length", 0)) for p in gp) / 1000
            item.update(qty=len(gp), unit="pieces", running_length_m=round(run, 2))
        else:
            item.update(qty=len(gp), unit="pieces")
        summary.append(item)
    return rows, summary


def is_l_shape(sofa_type) -> bool:
    t = str(sofa_type or "").lower().replace("-", "_")
    return "l_shape" in t or "sectional" in t or "corner" in t


def build_structure(L, W, H, sofa_type="3_seater", arm_width=None, leg_height=None,
                    style=None, chaise_length=None, chaise_side="right") -> list[Part]:
    if is_l_shape(sofa_type):
        return build_l_shape(L, W, H, chaise_length, chaise_side, style,
                             arm_width=arm_width, leg_height=leg_height)[0]
    params = StructureParams.from_style(L, W, H, seats_from_type(sofa_type, L), style,
                                        arm_width=arm_width, leg_height=leg_height)
    return SofaStructure(params).build()
