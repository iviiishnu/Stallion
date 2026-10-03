"""
bom_bridge.py — Converts cost_engine's scaled BOM (a DataFrame keyed by
'component_group' / 'new_qty') into the flat dict that SofaGeometry
(cad_generator.py, sofa_3d_generator.py) expects:

    {"Springs": 11, "Seat Belts": 3, "Back Rest Belts": 15, "Clips": 45}

Matching is case-insensitive and substring-based, because component
naming in master_template_spec.csv may vary slightly (e.g. "Back Belts"
vs "Back Rest Belts") between data revisions. If a component can't be
found, the SofaGeometry default for that key is used instead (it already
has sane fallbacks), so a partially-populated BOM never crashes CAD
generation — it just falls back to base-sofa defaults for that one item.
"""

# geometry_key -> substrings to match against component_group (all must
# appear, case-insensitive) — ordered most-specific first so e.g.
# "back rest belts" doesn't get caught by the plain "belts" rule meant
# for seat belts.
_MATCH_RULES = [
    ("Springs",          ["spring"]),
    ("Back Rest Belts",  ["back", "belt"]),
    ("Seat Belts",       ["seat", "belt"]),
    ("Clips",            ["clip"]),
]


def bom_df_to_geometry_dict(bom_df) -> dict:
    """
    bom_df: the DataFrame returned by SofaCostEngine.generate_scaled_bom(),
            with at least 'component_group' and 'new_qty' columns.

    Returns a dict suitable for SofaGeometry(..., bom=<this dict>).
    """
    result = {}
    if bom_df is None or bom_df.empty:
        return result

    rows = list(zip(
        bom_df["component_group"].astype(str).str.lower(),
        bom_df["new_qty"],
    ))

    used_rows = set()
    for geo_key, needles in _MATCH_RULES:
        for i, (name, qty) in enumerate(rows):
            if i in used_rows:
                continue
            if all(n in name for n in needles):
                result[geo_key] = float(qty)
                used_rows.add(i)
                break  # take first match for this geo_key

    return result
