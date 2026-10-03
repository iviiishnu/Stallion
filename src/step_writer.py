"""
step_writer.py — Minimal STEP (ISO 10303-21, AP214) writer for polyhedral
solids, so generated sofas open as real CAD bodies in Fusion 360,
SolidWorks, FreeCAD, etc.

Output is an assembly: one component per group ("Wood Frame", "Springs",
...) mirroring the component tree of the industry reference model, each
holding named, coloured MANIFOLD_SOLID_BREP bodies with planar faces.

Every Part must be a closed 2-manifold whose faces are planar polygons
wound counter-clockwise when seen from outside (sofa_structure.Part).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np


def _r(v: float) -> str:
    s = f"{v:.4f}".rstrip("0")
    return "0." if s in ("-0.", "0.") else s


def _s(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


class _Writer:
    def __init__(self):
        self.lines = []
        self.n = 0

    def add(self, entity: str) -> str:
        self.n += 1
        self.lines.append(f"#{self.n}={entity};")
        return f"#{self.n}"

    def point(self, p):
        return self.add(f"CARTESIAN_POINT('',({_r(p[0])},{_r(p[1])},{_r(p[2])}))")

    def direction(self, d):
        d = np.asarray(d, float)
        d = d / (np.linalg.norm(d) or 1.0)
        return self.add(f"DIRECTION('',({_r(d[0])},{_r(d[1])},{_r(d[2])}))")

    def axis(self, origin=(0, 0, 0), z=(0, 0, 1), x=(1, 0, 0)):
        return self.add(f"AXIS2_PLACEMENT_3D('',{self.point(origin)},{self.direction(z)},{self.direction(x)})")


def _newell(pts):
    n = np.zeros(3)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    return n


def _solid(w: _Writer, part) -> str:
    V = part.verts
    vp = {}

    def vertex(i):
        if i not in vp:
            vp[i] = w.add(f"VERTEX_POINT('',{w.point(V[i])})")
        return vp[i]

    edges = {}

    def edge(a, b):
        key = (min(a, b), max(a, b))
        if key not in edges:
            i, j = key
            d = V[j] - V[i]
            length = float(np.linalg.norm(d))
            vec = w.add(f"VECTOR('',{w.direction(d)},{_r(length)})")
            line = w.add(f"LINE('',{w.point(V[i])},{vec})")
            edges[key] = w.add(f"EDGE_CURVE('',{vertex(i)},{vertex(j)},{line},.T.)")
        return edges[key], a < b

    faces = []
    for f in part.faces:
        pts = V[f]
        nrm = _newell(pts)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        oes = []
        for k in range(len(f)):
            ec, same = edge(f[k], f[(k + 1) % len(f)])
            oes.append(w.add(f"ORIENTED_EDGE('',*,*,{ec},{'.T.' if same else '.F.'})"))
        loop = w.add(f"EDGE_LOOP('',({','.join(oes)}))")
        bound = w.add(f"FACE_OUTER_BOUND('',{loop},.T.)")
        xdir = pts[1] - pts[0]
        nz = nrm / np.linalg.norm(nrm)
        xdir = xdir - nz * np.dot(xdir, nz)
        plane = w.add(f"PLANE('',{w.axis(pts[0], nz, xdir)})")
        faces.append(w.add(f"ADVANCED_FACE('',({bound}),{plane},.T.)"))
    shell = w.add(f"CLOSED_SHELL('',({','.join(faces)}))")
    return w.add(f"MANIFOLD_SOLID_BREP({_s(part.name)},{shell})")


def write_step(parts, path, colors: dict, product_name="Sofa", group_order=None) -> str:
    """Write `parts` (list of sofa_structure.Part, mm) to an AP214 STEP file."""
    w = _Writer()
    app = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
    pctx = w.add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
    pdctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")

    mm = w.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    rad = w.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    sr = w.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-03),{mm},'distance_accuracy_value','confusion accuracy')")
    ctx = w.add(f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc})) "
                f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({mm},{rad},{sr})) REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")

    def product(name):
        prod = w.add(f"PRODUCT({_s(name)},{_s(name)},'',({pctx}))")
        w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({prod}))")
        pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
        pd = w.add(f"PRODUCT_DEFINITION('design','',{pdf},{pdctx})")
        pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
        return pd, pds

    root_pd, root_pds = product(product_name)
    root_axis = w.axis()
    groups = []
    for g in (group_order or []):
        if any(p.group == g for p in parts) and g not in groups:
            groups.append(g)
    for p in parts:
        if p.group not in groups:
            groups.append(p.group)

    styled = []
    styles = {}
    for gi, g in enumerate(groups):
        gparts = [p for p in parts if p.group == g]
        pd, pds = product(g)
        axis = w.axis()
        solids = [_solid(w, p) for p in gparts]
        rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_s(g)},({','.join([axis] + solids)}),{ctx})")
        w.add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")

        for part, s in zip(gparts, solids):
            rgba = tuple(part.meta.get("rgb") or colors.get(g, (180, 180, 180, 255)))[:3]
            if rgba not in styles:
                col = w.add(f"COLOUR_RGB('',{_r(rgba[0] / 255)},{_r(rgba[1] / 255)},{_r(rgba[2] / 255)})")
                fill = w.add(f"FILL_AREA_STYLE('',(FILL_AREA_STYLE_COLOUR('',{col})))")
                side = w.add(f"SURFACE_SIDE_STYLE('',(SURFACE_STYLE_FILL_AREA({fill})))")
                styles[rgba] = w.add(f"PRESENTATION_STYLE_ASSIGNMENT((SURFACE_STYLE_USAGE(.BOTH.,{side})))")
            styled.append(w.add(f"STYLED_ITEM('color',({styles[rgba]}),{s})"))

        idt = w.add(f"ITEM_DEFINED_TRANSFORMATION('','',{root_axis},{axis})")
        root_rep_ref = "{ROOT_REP}"
        rr = w.add(f"( REPRESENTATION_RELATIONSHIP('','',{rep},{root_rep_ref}) "
                   f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION({idt}) SHAPE_REPRESENTATION_RELATIONSHIP() )")
        nauo = w.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE({_s(gi + 1)},{_s(g)},'',{root_pd},{pd},$)")
        occ = w.add(f"PRODUCT_DEFINITION_SHAPE('Placement','Placement of an item',{nauo})")
        w.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION({rr},{occ})")

    root_rep = w.add(f"SHAPE_REPRESENTATION({_s(product_name)},({root_axis}),{ctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION({root_pds},{root_rep})")
    w.add(f"MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',({','.join(styled)}),{ctx})")

    body = "\n".join(line.replace("{ROOT_REP}", root_rep) for line in w.lines)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    text = (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_DESCRIPTION(('Stallion parametric sofa'),'2;1');\n"
        f"FILE_NAME({_s(Path(path).name)},'{stamp}',('Stallion'),('Stallion'),'stallion step_writer','stallion','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\nDATA;\n" + body + "\nENDSEC;\nEND-ISO-10303-21;\n"
    )
    Path(path).write_text(text, encoding="ascii")
    return str(path)
