"""
cad_generator.py — Parametric sofa CAD drawing generator.

Generates a fully annotated 3-view technical drawing (Top / Front / Side)
with both external shell and internal components, driven entirely by the
user-supplied L × W × H dimensions and the scaled BOM.

External components:
  - Overall sofa envelope
  - Left & right arms
  - Seat area
  - Backrest
  - 4 legs

Internal components (shown as hatched / dashed layers):
  - Wood frame (inner structural rectangle)
  - Plywood back panel
  - Seat foam layer
  - Back foam layer
  - Handle / arm frame
  - Springs (evenly spaced dots along seat width)
  - Seat belts (horizontal lines across seat)
  - Back rest belts (vertical lines on backrest)
  - Clips (small squares at spring-belt intersections)

Output:
  outputs/cad/{request_id}_sheet.png    — A3 sheet with all 3 views + title block
  outputs/cad/{request_id}.dxf          — AutoCAD-compatible DXF file

Usage:
  from cad_generator import SofaCADGenerator
  gen = SofaCADGenerator(L=2100, W=900, H=850, bom=bom_dict, request_id="web_xxx")
  gen.export_png(output_dir)
  gen.export_dxf(output_dir)
"""

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

import ezdxf
from ezdxf import colors as dxf_colors
from ezdxf.enums import TextEntityAlignment


# ── Colour palette (CAD standard-ish) ────────────────────────────────────────
C_OUTLINE    = "#000000"   # black — outer envelope
C_FRAME      = "#1a3a6e"   # dark blue — wood frame
C_PLYWOOD    = "#8B5E3C"   # brown — plywood
C_FOAM       = "#e8a045"   # amber — foam
C_FABRIC     = "#6c8ebf"   # steel blue — fabric / upholstery
C_SPRING     = "#cc2222"   # red — springs
C_BELT       = "#228b22"   # green — belts
C_CLIP       = "#9932cc"   # purple — clips
C_LEG        = "#555555"   # grey — legs
C_DIM        = "#333333"   # dimension lines
C_HATCH_FOAM = "#f5d090"   # light amber — foam hatch fill
C_HATCH_PLY  = "#d2a67a"   # light brown — plywood hatch fill
C_TITLE_BG   = "#1a3a6e"   # title block background


class SofaGeometry:
    """
    Derives all internal component dimensions from L × W × H.
    All values in mm (drawn to scale on canvas in mm).
    """

    def __init__(self, L: float, W: float, H: float, bom: dict, design_params: Optional[dict] = None):
        # Overall sofa
        self.L = L   # total length (x-axis in top view)
        self.W = W   # total width  (y-axis in top view)
        self.H = H   # total height (z-axis in front/side view)

        # ── Structural ratios ─────────────────────────────────────────────────
        # Photo-derived values are optional. Every parameter is bounded and
        # falls back to the original calibrated template when unavailable.
        dp = design_params or {}

        def _ratio(key, default, lo, hi):
            try:
                value = float(dp.get(key, default))
            except (TypeError, ValueError):
                value = default
            return min(max(value, lo), hi)

        self.arm_width_ratio = _ratio("arm_width_ratio", 0.095, 0.045, 0.18)
        self.back_depth_ratio = _ratio("backrest_depth_ratio", 0.200, 0.10, 0.32)
        self.seat_height_ratio = _ratio("seat_height_ratio", 0.47, 0.30, 0.68)
        self.leg_height_ratio = _ratio("leg_height_ratio", 0.094, 0.035, 0.20)
        self.arm_roundness = _ratio("arm_roundness", 0.55, 0.0, 1.0)
        self.back_roundness = _ratio("back_roundness", 0.55, 0.0, 1.0)

        self.arm_width   = round(L * self.arm_width_ratio)
        self.back_depth  = round(W * self.back_depth_ratio)
        self.seat_depth  = W - self.back_depth
        self.seat_height = round(H * self.seat_height_ratio)
        self.leg_height  = round(H * self.leg_height_ratio)
        self.leg_size    = round(L * 0.030)
        self.back_height = H - self.seat_height
        self.frame_inset = 40

        # ── Fabric / upholstery thickness ────────────────────────────────────
        self.fabric_t    = 20

        # ── Foam layers ──────────────────────────────────────────────────────
        self.seat_foam_t = round(self.seat_height * 0.40)   # ~160mm thick foam
        self.back_foam_t = round(self.back_depth  * 0.45)

        # ── Springs from BOM ─────────────────────────────────────────────────
        self.spring_count = max(1, int(round(bom.get("Springs",       11))))
        self.belt_count   = max(1, int(round(bom.get("Seat Belts",     3))))
        self.back_belt_count = max(1, int(round(bom.get("Back Rest Belts", 15))))
        self.clip_count   = max(1, int(round(bom.get("Clips",         45))))

        # Inner seat box (between arms, in front of back)
        self.seat_x0 = self.arm_width
        self.seat_x1 = L - self.arm_width
        self.seat_w  = self.seat_x1 - self.seat_x0


class SofaCADGenerator:
    """
    Generates annotated 3-view sofa CAD drawings (Top / Front / Side)
    with external shell and internal components.
    """

    def __init__(
        self,
        L: float,
        W: float,
        H: float,
        bom: Optional[dict] = None,
        request_id: str = "sofa_cad",
        sofa_type: str = "3_seater",
        design_params: Optional[dict] = None,
    ):
        self.request_id = request_id
        self.sofa_type  = sofa_type
        self.geo = SofaGeometry(L, W, H, bom or {}, design_params=design_params)

    # ─────────────────────────────────────────────────────────────────────────
    # PNG export — matplotlib
    # ─────────────────────────────────────────────────────────────────────────

    def export_png(self, output_dir: str) -> str:
        """Render all 3 views + title block onto a single A3-proportioned sheet."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.request_id}_sheet.png"

        # A3 landscape at 150 dpi → 16.5 × 11.7 inches
        fig = plt.figure(figsize=(16.5, 11.7), dpi=150)
        fig.patch.set_facecolor("#f8f8f0")

        # Grid: 2 rows × 2 cols
        # Top row: top-view (wide) | side-view
        # Bottom row: front-view (wide) | title block
        gs = fig.add_gridspec(
            2, 2,
            left=0.04, right=0.98, top=0.92, bottom=0.08,
            hspace=0.35, wspace=0.25,
            width_ratios=[1.6, 1.0],
            height_ratios=[1.0, 1.0],
        )

        ax_top   = fig.add_subplot(gs[0, 0])
        ax_side  = fig.add_subplot(gs[0, 1])
        ax_front = fig.add_subplot(gs[1, 0])
        ax_title = fig.add_subplot(gs[1, 1])

        self._draw_top_view(ax_top)
        self._draw_front_elevation(ax_front)
        self._draw_side_elevation(ax_side)
        self._draw_title_block(ax_title, fig)
        self._draw_sheet_border(fig)

        fig.suptitle(
            "STALLION SOFA — TECHNICAL DRAWING",
            fontsize=13, fontweight="bold", color="#1a3a6e",
            y=0.97
        )

        plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(out_path)

    # ─────────────────────────────────────────────────────────────────────────
    # TOP VIEW
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_top_view(self, ax):
        g = self.geo
        ax.set_aspect("equal")
        ax.set_title("TOP VIEW", fontsize=9, fontweight="bold", pad=6, color="#1a3a6e")
        ax.set_facecolor("#fafafa")
        ax.axis("off")

        # ── External shell ───────────────────────────────────────────────────
        # Full sofa footprint
        _rect(ax, 0, 0, g.L, g.W, ec=C_OUTLINE, fc="#e8e8e8", lw=2.0, zorder=2)

        # Left arm (top view: narrow strip on left)
        _rect(ax, 0, 0, g.arm_width, g.W, ec=C_OUTLINE, fc="#d0d0d0", lw=1.5, zorder=3)

        # Right arm
        _rect(ax, g.L - g.arm_width, 0, g.arm_width, g.W,
              ec=C_OUTLINE, fc="#d0d0d0", lw=1.5, zorder=3)

        # Backrest strip (rear of sofa = top of top view)
        _rect(ax, g.arm_width, g.W - g.back_depth, g.seat_w, g.back_depth,
              ec=C_OUTLINE, fc="#c8c8d8", lw=1.5, zorder=3)

        # Seat area
        _rect(ax, g.arm_width, 0, g.seat_w, g.seat_depth,
              ec=C_OUTLINE, fc="#dde8f0", lw=1.5, zorder=3)

        # ── Internal: Wood frame ──────────────────────────────────────────────
        fi = g.frame_inset
        _rect(ax, fi, fi, g.L - 2*fi, g.W - 2*fi,
              ec=C_FRAME, fc="none", lw=1.2, ls="--", zorder=4,
              label="Wood Frame")

        # ── Internal: Springs (dots along seat centre line) ──────────────────
        spring_y = g.seat_depth * 0.5
        seat_span = g.seat_w - 2 * fi
        if g.spring_count > 1:
            spring_xs = [
                g.seat_x0 + fi + i * seat_span / (g.spring_count - 1)
                for i in range(g.spring_count)
            ]
        else:
            spring_xs = [g.seat_x0 + g.seat_w / 2]

        for sx in spring_xs:
            ax.plot(sx, spring_y, "o", color=C_SPRING, ms=5, zorder=6)

        # ── Internal: Seat belts (lines perpendicular to springs) ─────────────
        if g.belt_count > 1:
            belt_xs = [
                g.seat_x0 + fi + i * seat_span / (g.belt_count - 1)
                for i in range(g.belt_count)
            ]
        else:
            belt_xs = [g.seat_x0 + g.seat_w / 2]

        for bx in belt_xs:
            ax.plot([bx, bx], [fi, g.seat_depth - fi],
                    color=C_BELT, lw=0.8, ls="-.", zorder=5)

        # ── Internal: Arm frames (dashed inset boxes inside arms) ─────────────
        afi = fi + 10
        _rect(ax, afi, afi, g.arm_width - 2*afi, g.W - 2*afi,
              ec=C_FRAME, fc="none", lw=0.9, ls=":", zorder=4)
        _rect(ax, g.L - g.arm_width + afi, afi, g.arm_width - 2*afi, g.W - 2*afi,
              ec=C_FRAME, fc="none", lw=0.9, ls=":", zorder=4)

        # ── Dimension arrows ──────────────────────────────────────────────────
        # Overall length
        _dim_arrow(ax, 0, -g.W*0.18, g.L, -g.W*0.18,
                   f"L = {g.L} mm", color=C_DIM, fontsize=7)
        # Overall width
        _dim_arrow(ax, -g.L*0.10, 0, -g.L*0.10, g.W,
                   f"W = {g.W} mm", color=C_DIM, fontsize=7, vertical=True)
        # Arm width
        _dim_arrow(ax, 0, g.W + g.W*0.08, g.arm_width, g.W + g.W*0.08,
                   f"Arm {g.arm_width}", color=C_DIM, fontsize=6)
        # Back depth
        _dim_arrow(ax, g.L + g.L*0.05, g.W - g.back_depth, g.L + g.L*0.05, g.W,
                   f"Back\n{g.back_depth}", color=C_DIM, fontsize=6, vertical=True)

        # ── Legend ────────────────────────────────────────────────────────────
        legend_items = [
            mpatches.Patch(fc="#d0d0d0", ec=C_OUTLINE, label="Arms"),
            mpatches.Patch(fc="#dde8f0", ec=C_OUTLINE, label="Seat"),
            mpatches.Patch(fc="#c8c8d8", ec=C_OUTLINE, label="Backrest"),
            mlines.Line2D([], [], color=C_FRAME, ls="--", lw=1.2, label="Wood Frame"),
            mlines.Line2D([], [], color=C_SPRING, marker="o", ls="none", ms=5, label=f"Springs ×{g.spring_count}"),
            mlines.Line2D([], [], color=C_BELT, ls="-.", lw=0.8, label=f"Seat Belts ×{g.belt_count}"),
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=5.5,
                  framealpha=0.85, edgecolor="#aaa")

        # Padding
        pad_x = g.L * 0.25
        pad_y = g.W * 0.30
        ax.set_xlim(-pad_x, g.L + pad_x)
        ax.set_ylim(-pad_y, g.W + pad_y)

    # ─────────────────────────────────────────────────────────────────────────
    # FRONT ELEVATION
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_front_elevation(self, ax):
        g = self.geo
        ax.set_aspect("equal")
        ax.set_title("FRONT ELEVATION", fontsize=9, fontweight="bold", pad=6, color="#1a3a6e")
        ax.set_facecolor("#fafafa")
        ax.axis("off")

        # ── External: Full sofa front face ────────────────────────────────────
        _rect(ax, 0, 0, g.L, g.H, ec=C_OUTLINE, fc="#e8e8e8", lw=2.0, zorder=2)

        # Left arm block
        _rect(ax, 0, 0, g.arm_width, g.H - g.back_height * 0.4,
              ec=C_OUTLINE, fc="#d0d0d0", lw=1.5, zorder=3)
        # Right arm block
        _rect(ax, g.L - g.arm_width, 0, g.arm_width, g.H - g.back_height * 0.4,
              ec=C_OUTLINE, fc="#d0d0d0", lw=1.5, zorder=3)

        # Backrest (full width, upper portion)
        _rect(ax, 0, g.seat_height, g.L, g.back_height,
              ec=C_OUTLINE, fc="#c8c8d8", lw=1.5, zorder=3)

        # Seat cushion block
        _rect(ax, g.arm_width, g.leg_height, g.seat_w, g.seat_height - g.leg_height,
              ec=C_OUTLINE, fc="#dde8f0", lw=1.5, zorder=3)

        # Legs (4 corners, front 2 visible)
        leg_y = 0
        _rect(ax, g.arm_width * 0.3, leg_y, g.leg_size, g.leg_height,
              ec=C_OUTLINE, fc="#888", lw=1.2, zorder=4)
        _rect(ax, g.L - g.arm_width * 0.3 - g.leg_size, leg_y, g.leg_size, g.leg_height,
              ec=C_OUTLINE, fc="#888", lw=1.2, zorder=4)

        # ── Internal: Wood frame (dashed inner rect) ──────────────────────────
        fi = g.frame_inset
        _rect(ax, fi, g.leg_height + fi, g.L - 2*fi, g.H - g.leg_height - 2*fi,
              ec=C_FRAME, fc="none", lw=1.2, ls="--", zorder=5)

        # ── Internal: Seat foam (hatched) ──────────────────────────────────────
        foam_y0 = g.leg_height + fi
        foam_h  = g.seat_foam_t
        foam_patch = mpatches.FancyBboxPatch(
            (g.arm_width + fi, foam_y0), g.seat_w - 2*fi, foam_h,
            boxstyle="square,pad=0", ec=C_FOAM, fc=C_HATCH_FOAM,
            lw=1.0, hatch="///", zorder=5, label="Seat Foam"
        )
        ax.add_patch(foam_patch)

        # ── Internal: Back foam (hatched) ────────────────────────────────────
        back_foam_patch = mpatches.FancyBboxPatch(
            (fi, g.seat_height + fi), g.L - 2*fi, g.back_foam_t,
            boxstyle="square,pad=0", ec=C_FOAM, fc=C_HATCH_FOAM,
            lw=1.0, hatch="\\\\\\", zorder=5, label="Back Foam"
        )
        ax.add_patch(back_foam_patch)

        # ── Internal: Springs (front view — vertical coils as rectangles) ─────
        spring_y0 = foam_y0 + foam_h
        spring_h  = g.seat_height - foam_y0 - foam_h - fi
        spring_h  = max(spring_h, 20)
        seat_span = g.seat_w - 2 * fi
        if g.spring_count > 1:
            spring_xs = [
                g.seat_x0 + fi + i * seat_span / (g.spring_count - 1)
                for i in range(g.spring_count)
            ]
        else:
            spring_xs = [g.seat_x0 + g.seat_w / 2]

        sw = max(seat_span / (g.spring_count * 2), 8)
        for sx in spring_xs:
            _rect(ax, sx - sw/2, spring_y0, sw, spring_h,
                  ec=C_SPRING, fc="none", lw=0.9, ls="-", zorder=6)
            # Coil zigzag hint (3 lines)
            for k in range(3):
                yy = spring_y0 + spring_h * (k + 0.5) / 3
                ax.plot([sx - sw/2, sx + sw/2], [yy, yy],
                        color=C_SPRING, lw=0.5, zorder=6)

        # ── Internal: Seat belts (horizontal across seat, below foam) ─────────
        if g.belt_count > 1:
            belt_ys = [
                spring_y0 + spring_h * (i + 0.5) / g.belt_count
                for i in range(g.belt_count)
            ]
        else:
            belt_ys = [spring_y0 + spring_h / 2]

        for by in belt_ys:
            ax.plot([g.seat_x0 + fi, g.seat_x1 - fi], [by, by],
                    color=C_BELT, lw=1.0, ls="-.", zorder=7)

        # ── Internal: Back rest belts (vertical on backrest) ─────────────────
        back_x0 = fi * 2
        back_x1 = g.L - fi * 2
        bk_span = back_x1 - back_x0
        if g.back_belt_count > 1:
            bk_xs = [
                back_x0 + i * bk_span / (g.back_belt_count - 1)
                for i in range(g.back_belt_count)
            ]
        else:
            bk_xs = [g.L / 2]

        for bkx in bk_xs:
            ax.plot([bkx, bkx],
                    [g.seat_height + fi + g.back_foam_t,
                     g.H - fi],
                    color=C_BELT, lw=0.7, ls="-.", zorder=7)

        # ── Internal: Plywood (thin panel at back of seat base) ───────────────
        ply_patch = mpatches.FancyBboxPatch(
            (fi, g.seat_height - 15), g.L - 2*fi, 15,
            boxstyle="square,pad=0", ec=C_PLYWOOD, fc=C_HATCH_PLY,
            lw=1.0, hatch="---", zorder=5
        )
        ax.add_patch(ply_patch)

        # ── Clips (small squares at spring-belt intersections) ────────────────
        clip_size = max(sw * 0.4, 5)
        n_clips_drawn = 0
        for sx in spring_xs:
            for by in belt_ys:
                if n_clips_drawn >= g.clip_count:
                    break
                _rect(ax, sx - clip_size/2, by - clip_size/2,
                      clip_size, clip_size,
                      ec=C_CLIP, fc="#e8b4ff", lw=0.7, zorder=8)
                n_clips_drawn += 1

        # ── Arm frame (internal dashed box inside arm) ────────────────────────
        afi = fi + 10
        _rect(ax, afi, g.leg_height + afi,
              g.arm_width - 2*afi, g.H - g.leg_height - 2*afi,
              ec=C_FRAME, fc="none", lw=0.9, ls=":", zorder=5)
        _rect(ax, g.L - g.arm_width + afi, g.leg_height + afi,
              g.arm_width - 2*afi, g.H - g.leg_height - 2*afi,
              ec=C_FRAME, fc="none", lw=0.9, ls=":", zorder=5)

        # ── Dimension arrows ──────────────────────────────────────────────────
        # Overall height
        _dim_arrow(ax, -g.L*0.10, 0, -g.L*0.10, g.H,
                   f"H = {g.H} mm", color=C_DIM, fontsize=7, vertical=True)
        # Overall length
        _dim_arrow(ax, 0, -g.H*0.15, g.L, -g.H*0.15,
                   f"L = {g.L} mm", color=C_DIM, fontsize=7)
        # Seat height
        _dim_arrow(ax, g.L + g.L*0.05, 0, g.L + g.L*0.05, g.seat_height,
                   f"Seat H\n{g.seat_height}", color=C_DIM, fontsize=6, vertical=True)
        # Leg height
        _dim_arrow(ax, g.L + g.L*0.14, 0, g.L + g.L*0.14, g.leg_height,
                   f"Leg\n{g.leg_height}", color=C_DIM, fontsize=6, vertical=True)
        # Back height
        _dim_arrow(ax, -g.L*0.20, g.seat_height, -g.L*0.20, g.H,
                   f"Back H\n{g.back_height}", color=C_DIM, fontsize=6, vertical=True)
        # Spring count label
        ax.text(g.L/2, spring_y0 + spring_h/2, f"×{g.spring_count} springs",
                ha="center", va="center", fontsize=5.5, color=C_SPRING,
                bbox=dict(fc="white", ec=C_SPRING, pad=1.5, alpha=0.8), zorder=9)

        # ── Legend ────────────────────────────────────────────────────────────
        legend_items = [
            mpatches.Patch(fc="#dde8f0", ec=C_OUTLINE, label="Seat Cushion"),
            mpatches.Patch(fc="#c8c8d8", ec=C_OUTLINE, label="Backrest"),
            mpatches.Patch(fc=C_HATCH_FOAM, ec=C_FOAM, hatch="///", label="Seat Foam"),
            mpatches.Patch(fc=C_HATCH_FOAM, ec=C_FOAM, hatch="\\\\\\", label="Back Foam"),
            mpatches.Patch(fc=C_HATCH_PLY,  ec=C_PLYWOOD, hatch="---", label="Plywood"),
            mlines.Line2D([], [], color=C_FRAME, ls="--", lw=1.2, label="Wood Frame"),
            mlines.Line2D([], [], color=C_SPRING, lw=0.9, label=f"Springs ×{g.spring_count}"),
            mlines.Line2D([], [], color=C_BELT,   ls="-.", lw=1.0, label=f"Seat Belts ×{g.belt_count}"),
            mlines.Line2D([], [], color=C_BELT,   ls="-.", lw=0.7, label=f"Back Belts ×{g.back_belt_count}"),
            mpatches.Patch(fc="#e8b4ff", ec=C_CLIP, label=f"Clips ×{g.clip_count}"),
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=5,
                  framealpha=0.88, edgecolor="#aaa", ncol=2)

        pad_x = g.L * 0.28
        pad_y = g.H * 0.25
        ax.set_xlim(-pad_x, g.L + pad_x)
        ax.set_ylim(-pad_y, g.H + pad_y)

    # ─────────────────────────────────────────────────────────────────────────
    # SIDE ELEVATION
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_side_elevation(self, ax):
        g = self.geo
        ax.set_aspect("equal")
        ax.set_title("SIDE ELEVATION", fontsize=9, fontweight="bold", pad=6, color="#1a3a6e")
        ax.set_facecolor("#fafafa")
        ax.axis("off")

        # ── External: Side profile ────────────────────────────────────────────
        _rect(ax, 0, 0, g.W, g.H, ec=C_OUTLINE, fc="#e8e8e8", lw=2.0, zorder=2)

        # Seat (lower front block)
        _rect(ax, 0, 0, g.seat_depth, g.seat_height,
              ec=C_OUTLINE, fc="#dde8f0", lw=1.5, zorder=3)

        # Backrest (rear column, full height)
        _rect(ax, g.seat_depth, 0, g.back_depth, g.H,
              ec=C_OUTLINE, fc="#c8c8d8", lw=1.5, zorder=3)

        # Leg (front bottom)
        _rect(ax, g.seat_depth * 0.15, 0, g.leg_size, g.leg_height,
              ec=C_OUTLINE, fc="#888", lw=1.2, zorder=4)

        # ── Internal: Wood frame ──────────────────────────────────────────────
        fi = g.frame_inset
        _rect(ax, fi, g.leg_height + fi, g.W - 2*fi, g.H - g.leg_height - 2*fi,
              ec=C_FRAME, fc="none", lw=1.2, ls="--", zorder=5)

        # ── Internal: Seat foam (hatched) ─────────────────────────────────────
        _rect(ax, fi, g.leg_height + fi, g.seat_depth - fi, g.seat_foam_t,
              ec=C_FOAM, fc=C_HATCH_FOAM, lw=1.0, zorder=5,
              hatch="///")

        # ── Internal: Back foam ───────────────────────────────────────────────
        _rect(ax, g.seat_depth + fi, fi, g.back_foam_t, g.H - 2*fi,
              ec=C_FOAM, fc=C_HATCH_FOAM, lw=1.0, zorder=5,
              hatch="\\\\\\")

        # ── Internal: Back rest belts (horizontal lines on back) ─────────────
        bk_y0 = g.seat_height + fi
        bk_h  = g.H - bk_y0 - fi
        if g.back_belt_count > 1:
            bk_ys = [
                bk_y0 + i * bk_h / (g.back_belt_count - 1)
                for i in range(g.back_belt_count)
            ]
        else:
            bk_ys = [bk_y0 + bk_h / 2]

        for bky in bk_ys:
            ax.plot([g.seat_depth + fi, g.W - fi], [bky, bky],
                    color=C_BELT, lw=0.7, ls="-.", zorder=6)

        # ── Internal: Springs (side view: circles in seat) ────────────────────
        spring_x = g.seat_depth * 0.5
        spring_y_bot = g.leg_height + fi + g.seat_foam_t
        spring_y_top = g.seat_height - fi
        spring_mid   = (spring_y_bot + spring_y_top) / 2
        spring_r     = max((spring_y_top - spring_y_bot) * 0.25, 8)
        circle = plt.Circle((spring_x, spring_mid), spring_r,
                             ec=C_SPRING, fc="none", lw=1.0, zorder=6)
        ax.add_patch(circle)
        ax.text(spring_x, spring_mid, "S", ha="center", va="center",
                fontsize=5, color=C_SPRING, zorder=7)

        # ── Dimension arrows ──────────────────────────────────────────────────
        _dim_arrow(ax, -g.W*0.12, 0, -g.W*0.12, g.H,
                   f"H = {g.H} mm", color=C_DIM, fontsize=7, vertical=True)
        _dim_arrow(ax, 0, -g.H*0.15, g.W, -g.H*0.15,
                   f"W = {g.W} mm", color=C_DIM, fontsize=7)
        _dim_arrow(ax, g.W + g.W*0.08, 0, g.W + g.W*0.08, g.seat_height,
                   f"Seat H\n{g.seat_height}", color=C_DIM, fontsize=6, vertical=True)
        _dim_arrow(ax, 0, g.H + g.H*0.08, g.seat_depth, g.H + g.H*0.08,
                   f"Seat D\n{g.seat_depth}", color=C_DIM, fontsize=6)
        _dim_arrow(ax, g.seat_depth, g.H + g.H*0.08, g.W, g.H + g.H*0.08,
                   f"Back D\n{g.back_depth}", color=C_DIM, fontsize=6)

        pad_x = g.W * 0.30
        pad_y = g.H * 0.30
        ax.set_xlim(-pad_x, g.W + pad_x)
        ax.set_ylim(-pad_y, g.H + pad_y)

    # ─────────────────────────────────────────────────────────────────────────
    # TITLE BLOCK
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_title_block(self, ax, fig):
        g = self.geo
        ax.axis("off")
        ax.set_facecolor("#f8f8f0")

        # Header
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.72), 0.96, 0.26, transform=ax.transAxes,
            boxstyle="square,pad=0", fc=C_TITLE_BG, ec="#000", lw=1.5
        ))
        ax.text(0.5, 0.88, "STALLION FURNITURE", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white")
        ax.text(0.5, 0.78, "SOFA TECHNICAL DRAWING", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#cce0ff")

        # Info rows
        rows = [
            ("Sofa Type",   self.sofa_type.replace("_", " ").title()),
            ("Length",      f"{g.L} mm"),
            ("Width",       f"{g.W} mm"),
            ("Height",      f"{g.H} mm"),
            ("Seat Height", f"{g.seat_height} mm"),
            ("Arm Width",   f"{g.arm_width} mm"),
            ("Back Depth",  f"{g.back_depth} mm"),
            ("Leg Height",  f"{g.leg_height} mm"),
            ("Springs",     f"×{g.spring_count}"),
            ("Seat Belts",  f"×{g.belt_count}"),
            ("Back Belts",  f"×{g.back_belt_count}"),
            ("Clips",       f"×{g.clip_count}"),
        ]

        y_start = 0.67
        row_h   = 0.052
        for i, (label, value) in enumerate(rows):
            y = y_start - i * row_h
            if y < 0.02:
                break
            bg = "#eef2ff" if i % 2 == 0 else "#f8f8f0"
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.02, y - row_h * 0.8), 0.96, row_h * 0.85,
                transform=ax.transAxes,
                boxstyle="square,pad=0", fc=bg, ec="#ccc", lw=0.5
            ))
            ax.text(0.05, y - row_h * 0.35, label, transform=ax.transAxes,
                    ha="left", va="center", fontsize=6.5, color="#333")
            ax.text(0.97, y - row_h * 0.35, value, transform=ax.transAxes,
                    ha="right", va="center", fontsize=6.5, fontweight="bold", color="#1a3a6e")

        # Request ID footer
        ax.text(0.5, 0.005, f"ID: {self.request_id}  |  Scale: NTS  |  Units: mm",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.5, color="#777")

    def _draw_sheet_border(self, fig):
        """Draw a simple border around the whole figure."""
        fig.add_artist(mpatches.FancyBboxPatch(
            (0.005, 0.005), 0.990, 0.990,
            transform=fig.transFigure,
            boxstyle="square,pad=0",
            fc="none", ec="#1a3a6e", lw=2.5
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # DXF export — ezdxf
    # ─────────────────────────────────────────────────────────────────────────

    def export_dxf(self, output_dir: str) -> str:
        """
        Export a proper DXF file with three model-space layouts (Top, Front, Side).
        Each component is on its own named layer with appropriate colour.
        """
        out_dir  = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.request_id}.dxf"

        doc = ezdxf.new(dxfversion="R2010")
        doc.header["$INSUNITS"] = 4   # mm

        msp = doc.modelspace()
        g   = self.geo

        # ── Layer setup ──────────────────────────────────────────────────────
        layers = {
            "OUTLINE":      (dxf_colors.WHITE,      "Continuous"),
            "WOOD_FRAME":   (dxf_colors.BLUE,       "DASHED"),
            "PLYWOOD":      (dxf_colors.YELLOW,     "Continuous"),
            "FOAM_SEAT":    (dxf_colors.YELLOW,     "Continuous"),
            "FOAM_BACK":    (dxf_colors.YELLOW,     "Continuous"),
            "SPRINGS":      (dxf_colors.RED,        "Continuous"),
            "SEAT_BELTS":   (dxf_colors.GREEN,      "DASHDOT"),
            "BACK_BELTS":   (dxf_colors.GREEN,      "DASHDOT"),
            "CLIPS":        (dxf_colors.MAGENTA,    "Continuous"),
            "ARM_FRAME":    (dxf_colors.BLUE,       "DOTTED"),
            "LEGS":         (dxf_colors.GRAY,       "Continuous"),
            "DIMENSIONS":   (dxf_colors.CYAN,       "Continuous"),
        }
        for name, (colour, lt) in layers.items():
            doc.layers.add(name, color=colour)

        # ── OFFSET constants — arrange 3 views with spacing ──────────────────
        GAP = 200   # gap between views in mm
        TOP_OFFSET_X  = 0
        TOP_OFFSET_Y  = g.H + GAP   # Top view above Front
        SIDE_OFFSET_X = g.L + GAP   # Side view to the right of Front
        SIDE_OFFSET_Y = g.H + GAP

        # ── VIEW LABELS ───────────────────────────────────────────────────────
        for label, ox, oy in [
            ("TOP VIEW",      TOP_OFFSET_X  + g.L/2, TOP_OFFSET_Y  + g.W + 80),
            ("FRONT ELEV.",   g.L/2,                 -80),
            ("SIDE ELEV.",    SIDE_OFFSET_X + g.W/2, SIDE_OFFSET_Y - 80 + g.W + 80),
        ]:
            msp.add_text(
                label,
                dxfattribs={
                    "layer": "DIMENSIONS",
                    "height": 50,
                    "insert": (ox, oy),
                    "halign": 4,  # center
                }
            )

        # ────────────────────────────────────────────────────────────────────
        # Helper: add DXF rect
        # ────────────────────────────────────────────────────────────────────
        def dxf_rect(ox, oy, w, h, layer):
            pts = [
                (ox,     oy),
                (ox + w, oy),
                (ox + w, oy + h),
                (ox,     oy + h),
                (ox,     oy),
            ]
            msp.add_lwpolyline(pts, dxfattribs={"layer": layer, "closed": True})

        def dxf_line(x0, y0, x1, y1, layer):
            msp.add_line((x0, y0), (x1, y1), dxfattribs={"layer": layer})

        def dxf_circle(cx, cy, r, layer):
            msp.add_circle((cx, cy), r, dxfattribs={"layer": layer})

        def dxf_dim_h(x0, x1, y, text, layer="DIMENSIONS"):
            """Horizontal dimension line."""
            msp.add_linear_dim(
                base=(x0 + (x1 - x0)/2, y),
                p1=(x0, 0),
                p2=(x1, 0),
                dimstyle="EZDXF",
                override={"dimtxt": 35, "dimasz": 25},
                dxfattribs={"layer": layer},
            ).render()

        # ── TOP VIEW ─────────────────────────────────────────────────────────
        ox, oy = TOP_OFFSET_X, TOP_OFFSET_Y
        dxf_rect(ox, oy, g.L, g.W, "OUTLINE")
        dxf_rect(ox, oy, g.arm_width, g.W, "OUTLINE")
        dxf_rect(ox + g.L - g.arm_width, oy, g.arm_width, g.W, "OUTLINE")
        dxf_rect(ox + g.arm_width, oy + g.seat_depth, g.seat_w, g.back_depth, "OUTLINE")
        dxf_rect(ox + g.arm_width, oy, g.seat_w, g.seat_depth, "OUTLINE")

        fi = g.frame_inset
        dxf_rect(ox + fi, oy + fi, g.L - 2*fi, g.W - 2*fi, "WOOD_FRAME")

        seat_span = g.seat_w - 2 * fi
        if g.spring_count > 1:
            sxs = [ox + g.seat_x0 + fi + i * seat_span / (g.spring_count - 1)
                   for i in range(g.spring_count)]
        else:
            sxs = [ox + g.seat_x0 + g.seat_w / 2]

        for sx in sxs:
            dxf_circle(sx, oy + g.seat_depth * 0.5, 8, "SPRINGS")

        if g.belt_count > 1:
            bxs = [ox + g.seat_x0 + fi + i * seat_span / (g.belt_count - 1)
                   for i in range(g.belt_count)]
        else:
            bxs = [ox + g.seat_x0 + g.seat_w / 2]

        for bx in bxs:
            dxf_line(bx, oy + fi, bx, oy + g.seat_depth - fi, "SEAT_BELTS")

        # ── FRONT ELEVATION ───────────────────────────────────────────────────
        ox, oy = 0, 0
        dxf_rect(ox, oy, g.L, g.H, "OUTLINE")
        dxf_rect(ox, oy, g.arm_width, g.H - g.back_height * 0.4, "OUTLINE")
        dxf_rect(ox + g.L - g.arm_width, oy, g.arm_width, g.H - g.back_height * 0.4, "OUTLINE")
        dxf_rect(ox, oy + g.seat_height, g.L, g.back_height, "OUTLINE")
        dxf_rect(ox + g.arm_width, oy + g.leg_height,
                 g.seat_w, g.seat_height - g.leg_height, "OUTLINE")

        # Legs
        dxf_rect(ox + g.arm_width * 0.3, oy, g.leg_size, g.leg_height, "LEGS")
        dxf_rect(ox + g.L - g.arm_width * 0.3 - g.leg_size, oy,
                 g.leg_size, g.leg_height, "LEGS")

        # Wood frame
        dxf_rect(ox + fi, oy + g.leg_height + fi,
                 g.L - 2*fi, g.H - g.leg_height - 2*fi, "WOOD_FRAME")

        # Seat foam
        dxf_rect(ox + g.arm_width + fi, oy + g.leg_height + fi,
                 g.seat_w - 2*fi, g.seat_foam_t, "FOAM_SEAT")

        # Back foam
        dxf_rect(ox + fi, oy + g.seat_height + fi,
                 g.L - 2*fi, g.back_foam_t, "FOAM_BACK")

        # Plywood
        dxf_rect(ox + fi, oy + g.seat_height - 15, g.L - 2*fi, 15, "PLYWOOD")

        # Springs front
        spring_y0 = g.leg_height + fi + g.seat_foam_t
        spring_h  = max(g.seat_height - spring_y0 - fi, 20)
        sw = max(seat_span / (g.spring_count * 2), 8)
        for sx_raw in sxs:
            sx = sx_raw - ox  # remove top-view offset
            dxf_rect(sx - sw/2, spring_y0, sw, spring_h, "SPRINGS")

        # Seat belts front
        if g.belt_count > 1:
            belt_ys = [spring_y0 + spring_h * (i + 0.5) / g.belt_count
                       for i in range(g.belt_count)]
        else:
            belt_ys = [spring_y0 + spring_h / 2]

        for by in belt_ys:
            dxf_line(g.seat_x0 + fi, by, g.seat_x1 - fi, by, "SEAT_BELTS")

        # Clips
        clip_size = max(sw * 0.4, 5)
        n_clips = 0
        for sx_raw in sxs:
            sx = sx_raw - ox
            for by in belt_ys:
                if n_clips >= g.clip_count:
                    break
                dxf_rect(sx - clip_size/2, by - clip_size/2,
                         clip_size, clip_size, "CLIPS")
                n_clips += 1

        # Back belts front
        back_x0 = fi * 2
        back_x1 = g.L - fi * 2
        bk_span = back_x1 - back_x0
        if g.back_belt_count > 1:
            bk_xs = [back_x0 + i * bk_span / (g.back_belt_count - 1)
                     for i in range(g.back_belt_count)]
        else:
            bk_xs = [g.L / 2]

        for bkx in bk_xs:
            dxf_line(bkx, g.seat_height + fi + g.back_foam_t,
                     bkx, g.H - fi, "BACK_BELTS")

        # ── SIDE ELEVATION ────────────────────────────────────────────────────
        ox, oy = SIDE_OFFSET_X, SIDE_OFFSET_Y - g.H
        dxf_rect(ox, oy, g.W, g.H, "OUTLINE")
        dxf_rect(ox, oy, g.seat_depth, g.seat_height, "OUTLINE")
        dxf_rect(ox + g.seat_depth, oy, g.back_depth, g.H, "OUTLINE")
        dxf_rect(ox + g.seat_depth * 0.15, oy, g.leg_size, g.leg_height, "LEGS")

        dxf_rect(ox + fi, oy + g.leg_height + fi,
                 g.W - 2*fi, g.H - g.leg_height - 2*fi, "WOOD_FRAME")
        dxf_rect(ox + fi, oy + g.leg_height + fi,
                 g.seat_depth - fi, g.seat_foam_t, "FOAM_SEAT")
        dxf_rect(ox + g.seat_depth + fi, oy + fi,
                 g.back_foam_t, g.H - 2*fi, "FOAM_BACK")

        bk_y0 = oy + g.seat_height + fi
        bk_h  = g.H - (g.seat_height + fi) - fi
        if g.back_belt_count > 1:
            bk_ys_s = [bk_y0 + i * bk_h / (g.back_belt_count - 1)
                       for i in range(g.back_belt_count)]
        else:
            bk_ys_s = [bk_y0 + bk_h / 2]

        for bky in bk_ys_s:
            dxf_line(ox + g.seat_depth + fi, bky, ox + g.W - fi, bky, "BACK_BELTS")

        spring_mid_side = oy + (g.leg_height + fi + g.seat_foam_t + g.seat_height) / 2
        spring_r_side   = max((g.seat_height - g.leg_height - fi - g.seat_foam_t) * 0.15, 8)
        dxf_circle(ox + g.seat_depth * 0.5, spring_mid_side, spring_r_side, "SPRINGS")

        doc.saveas(str(out_path))
        return str(out_path)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _rect(ax, x, y, w, h, ec, fc, lw=1.0, ls="-", zorder=2,
          label=None, hatch=None):
    kw = dict(linewidth=lw, edgecolor=ec, facecolor=fc,
              linestyle=ls, zorder=zorder)
    if hatch:
        kw["hatch"] = hatch
    if label:
        kw["label"] = label
    patch = mpatches.Rectangle((x, y), w, h, **kw)
    ax.add_patch(patch)
    return patch


def _dim_arrow(ax, x0, y0, x1, y1, label, color=C_DIM,
               fontsize=7, vertical=False):
    """Draw a dimension line with arrow caps and a centred label."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="<->",
            color=color,
            lw=0.8,
            shrinkA=0, shrinkB=0,
        ),
        zorder=10,
    )
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    rot = 90 if vertical else 0
    offset = (-18, 0) if vertical else (0, -12)
    ax.annotate(
        label,
        xy=(mx, my),
        xytext=offset,
        textcoords="offset points",
        ha="center", va="center",
        fontsize=fontsize,
        color=color,
        rotation=rot,
        zorder=10,
        bbox=dict(fc="white", ec="none", pad=1, alpha=0.7),
    )
