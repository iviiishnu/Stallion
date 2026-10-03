import os
import math
import json
import pandas as pd


class SofaCostEngine:
    def __init__(self, base_dir=".."):
        """
        base_dir should point to sofa_project when running from src/

        Expected structure:
            sofa_project/
            ├── data/
            │   ├── master_template/
            │   │   ├── master_dimensions.csv
            │   │   └── master_template_spec.csv
            │   ├── pricing/
            │   │   └── cost_sheet.csv
            │   └── fusion_mapping/
            │       └── fusion_component_map.csv
            ├── outputs/
            └── src/
                └── cost_engine.py
        """
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), base_dir))

        # -----------------------------
        # Input data paths
        # -----------------------------
        self.master_dim_path = os.path.join(
            self.base_dir, "data", "master_template", "master_dimensions.csv"
        )
        self.master_bom_path = os.path.join(
            self.base_dir, "data", "master_template", "master_template_spec.csv"
        )
        self.cost_sheet_path = os.path.join(
            self.base_dir, "data", "pricing", "cost_sheet.csv"
        )
        self.fusion_map_path = os.path.join(
            self.base_dir, "data", "fusion_mapping", "fusion_component_map.csv"
        )

        # -----------------------------
        # Output folders
        # -----------------------------
        self.outputs_dir = os.path.join(self.base_dir, "outputs")
        self.bom_output_dir = os.path.join(self.outputs_dir, "bom_outputs")
        self.quote_output_dir = os.path.join(self.outputs_dir, "quotations")
        self.fusion_report_dir = os.path.join(self.outputs_dir, "fusion_reports")

        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.bom_output_dir, exist_ok=True)
        os.makedirs(self.quote_output_dir, exist_ok=True)
        os.makedirs(self.fusion_report_dir, exist_ok=True)

        # -----------------------------
        # Data holders
        # -----------------------------
        self.master_dimensions = None
        self.master_bom = None
        self.cost_sheet = None
        self.fusion_map = None
        
        # Scaling rules mapping (since new Excel removed this column)
        self.DEFAULT_SCALING_RULES = {
            "wood frame": "3d volume",
            "plywood": "area",
            "seat foam": "area/volume",
            "back foam": "area/volume",
            "handle foam": "area/volume",
            "fabric": "surface area",
            "springs": "count by length",
            "clips": "derived from springs",
            "seat belts": "count by length",
            "back rest belts": "count by height",
            "handle frame": "area/volume",
            "adhesive": "3d volume",
            "thread": "surface area",
            "legs": "fixed",
            "hardware": "count by length"
        }
        
        # Dynamic dimension baselines and limits
        self.TYPE_BASE_DIMS = {}
        self.TYPE_LIMITS = {}

    # ---------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------
    def load_data(self):
        self.master_dimensions = pd.read_csv(self.master_dim_path)
        self.master_bom = pd.read_csv(self.master_bom_path)
        self.cost_sheet = pd.read_csv(self.cost_sheet_path)

        if os.path.exists(self.fusion_map_path):
            self.fusion_map = pd.read_csv(self.fusion_map_path)
        else:
            self.fusion_map = None

        print("Loaded:")
        files = {
            "master_dimensions": self.master_dim_path,
            "master_template_spec": self.master_bom_path,
            "cost_sheet": self.cost_sheet_path
        }
        if self.fusion_map is not None:
            files["fusion_component_map"] = self.fusion_map_path
            
        print("Loaded:")
        for name, path in files.items():
            print(f"  {name} -> {path}")
            
        # Parse master_dimensions into our runtime dictionaries
        for _, row in self.master_dimensions.iterrows():
            variant = str(row["variant"]).strip().lower()
            if not variant:
                continue
            self.TYPE_BASE_DIMS[variant] = (
                float(row["base_length"]),
                float(row["base_width"]),
                float(row["base_height"])
            )
            self.TYPE_LIMITS[variant] = {
                "min_l": float(row["min_length"]), "max_l": float(row["max_length"]),
                "min_w": float(row["min_width"]), "max_w": float(row["max_width"]),
                "min_h": float(row["min_height"]), "max_h": float(row["max_height"])
            }

    # ---------------------------------------------------
    # 2. GET BASE DIMENSIONS
    # ---------------------------------------------------
    def get_base_dimensions(self, sofa_type="3-seater"):
        """
        Retrieves L0, W0, H0 for the requested sofa type.
        """
        sofa_type = str(sofa_type).lower().strip()
        if sofa_type in self.TYPE_BASE_DIMS:
            return self.TYPE_BASE_DIMS[sofa_type]
        else:
            print(f"Warning: Unknown sofa type '{sofa_type}', defaulting to 3-seater baselines.")
            return self.TYPE_BASE_DIMS.get("3-seater", (2100, 900, 850))

    # ---------------------------------------------------
    # 3. COMPUTE SCALE FACTORS
    # ---------------------------------------------------
    def compute_scale_factors(self, length_mm, width_mm, height_mm, sofa_type="3-seater"):
        for name, value in (("length_mm", length_mm), ("width_mm", width_mm), ("height_mm", height_mm)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"'{name}' must be a number, got {value!r}")
            if value <= 0:
                raise ValueError(f"'{name}' must be a positive number, got {value}")
                
        sofa_type = str(sofa_type).lower().strip()
        if sofa_type in self.TYPE_LIMITS:
            limits = self.TYPE_LIMITS[sofa_type]
            if not (limits["min_l"] <= length_mm <= limits["max_l"]):
                raise ValueError(f"Length {length_mm}mm out of bounds for {sofa_type}. Must be between {limits['min_l']} and {limits['max_l']}.")
            if not (limits["min_w"] <= width_mm <= limits["max_w"]):
                raise ValueError(f"Width {width_mm}mm out of bounds for {sofa_type}. Must be between {limits['min_w']} and {limits['max_w']}.")
            if not (limits["min_h"] <= height_mm <= limits["max_h"]):
                raise ValueError(f"Height {height_mm}mm out of bounds for {sofa_type}. Must be between {limits['min_h']} and {limits['max_h']}.")

        L0, W0, H0 = self.get_base_dimensions(sofa_type)

        SL = length_mm / L0
        SW = width_mm / W0
        SH = height_mm / H0

        return {
            "SL": SL,
            "SW": SW,
            "SH": SH,
            "L0": L0,
            "W0": W0,
            "H0": H0,
            "L1": length_mm,
            "W1": width_mm,
            "H1": height_mm,
        }

    # ---------------------------------------------------
    # 4. SURFACE AREA RATIO
    # ---------------------------------------------------
    @staticmethod
    def surface_area_ratio(L1, W1, H1, L0, W0, H0):
        """
        Approximate surface-area based scaling ratio.
        Used for fabric / upholstery type components.
        """
        num = (L1 * W1) + (L1 * H1) + (W1 * H1)
        den = (L0 * W0) + (L0 * H0) + (W0 * H0)
        return num / den

    # ---------------------------------------------------
    # 5. SCALE ONE COMPONENT
    # ---------------------------------------------------
    def scale_component(self, component_name, base_qty, scaling_rule, scales, springs_new=None):
        """
        Supported scaling rules:
            3D Volume
            Area
            Area/Volume
            Surface Area
            Count by Length
            Count by Height
            Derived from Springs
            Fixed
        """
        SL = scales["SL"]
        SW = scales["SW"]
        SH = scales["SH"]

        L0, W0, H0 = scales["L0"], scales["W0"], scales["H0"]
        L1, W1, H1 = scales["L1"], scales["W1"], scales["H1"]

        rule = str(scaling_rule).strip().lower()

        # Structural components that scale in all 3 dimensions
        if rule == "3d volume":
            return base_qty * SL * SW * SH

        # Flat panels / plywood / back panels etc.
        elif rule == "area":
            # using length-height scaling as a simple structural panel approximation
            return base_qty * SL * SH

        # Foam / handle frame / similar components
        elif rule == "area/volume":
            # general approximation for seat/back/arm components
            return base_qty * SL * SW

        # Upholstery / fabric
        elif rule == "surface area":
            ratio = self.surface_area_ratio(L1, W1, H1, L0, W0, H0)
            return base_qty * ratio

        # Count-based items that depend mainly on sofa length
        elif rule == "count by length":
            return math.ceil(base_qty * SL)

        # Count-based items that depend mainly on height / back size
        elif rule == "count by height":
            return math.ceil(base_qty * SH)

        # Clips derived from number of springs
        elif rule == "derived from springs":
            if springs_new is None:
                raise ValueError(
                    f"springs_new is required for component '{component_name}' with rule 'Derived from Springs'"
                )
            # Keep same clip-to-spring ratio as base sofa
            return math.ceil((base_qty / 11.0) * springs_new)

        # No scaling
        elif rule == "fixed":
            return base_qty

        else:
            raise ValueError(
                f"Unknown scaling rule '{scaling_rule}' for component '{component_name}'"
            )

    # ---------------------------------------------------
    # 6. GENERATE SCALED BOM
    # ---------------------------------------------------
    def generate_scaled_bom(self, length_mm, width_mm, height_mm, sofa_type="3-seater"):
        """
        master_template_spec.csv now has columns like '3-Seater - Qty', '1-Seater - Qty'
        """
        scales = self.compute_scale_factors(length_mm, width_mm, height_mm, sofa_type)

        scaled_rows = []
        springs_new = None
        
        type_key = str(sofa_type).lower().strip()
        col_map = {
            "1-seater": "1-Seater - Qty",
            "2-seater": "2-Seater - Qty",
            "3-seater": "3-Seater - Qty",
            "4-seater": "4-Seater - Qty",
            "l-shape": "L-Shape (Left) - Qty"
        }
        qty_col = col_map.get(type_key, "3-Seater - Qty")

        # First pass: scale everything except clip-derived rows
        for _, row in self.master_bom.iterrows():
            component = str(row["Component Name"]).strip()
            
            # Skip if this component doesn't have a quantity for this type
            if qty_col not in row or pd.isna(row[qty_col]):
                continue
                
            base_qty = float(row[qty_col])
            if base_qty == 0:
                continue
                
            unit = str(row["Unit of Measurement"]).strip()
            scaling_rule = self.DEFAULT_SCALING_RULES.get(component.lower(), "fixed")
            notes = ""

            if scaling_rule.strip().lower() == "derived from springs":
                continue

            new_qty = self.scale_component(
                component_name=component,
                base_qty=base_qty,
                scaling_rule=scaling_rule,
                scales=scales
            )

            if component.lower() == "springs":
                springs_new = new_qty

            scaled_rows.append({
                "component_group": component,
                "base_qty": base_qty,
                "unit": unit,
                "scaling_rule": scaling_rule,
                "new_qty": new_qty,
                "notes": notes
            })

        # Second pass: components derived from springs (clips)
        for _, row in self.master_bom.iterrows():
            component = str(row["Component Name"]).strip()
            if qty_col not in row or pd.isna(row[qty_col]):
                continue
            base_qty = float(row[qty_col])
            if base_qty == 0:
                continue
            unit = str(row["Unit of Measurement"]).strip()
            scaling_rule = self.DEFAULT_SCALING_RULES.get(component.lower(), "fixed")
            notes = ""

            if scaling_rule.strip().lower() == "derived from springs":
                new_qty = self.scale_component(
                    component_name=component,
                    base_qty=base_qty,
                    scaling_rule=scaling_rule,
                    scales=scales,
                    springs_new=springs_new
                )

                scaled_rows.append({
                    "component_group": component,
                    "base_qty": base_qty,
                    "unit": unit,
                    "scaling_rule": scaling_rule,
                    "new_qty": new_qty,
                    "notes": notes
                })

        bom_df = pd.DataFrame(scaled_rows)

        # Keep same order as master BOM
        component_order = self.master_bom["Component Name"].tolist()
        bom_df["component_group"] = pd.Categorical(
            bom_df["component_group"],
            categories=component_order,
            ordered=True
        )
        bom_df = bom_df.sort_values("component_group").reset_index(drop=True)

        return scales, bom_df

    # ---------------------------------------------------
    # 7. GENERATE FUSION-SCALED COMPONENT REPORT
    # ---------------------------------------------------
    def generate_fusion_scaled_components(self, scales):
        """
        fusion_component_map.csv expected columns:
            fusion_component_name,component_group,scale_mode,cost_group,notes
        """
        if self.fusion_map is None:
            return None

        fusion_rows = []
        springs_new = None

        # First pass: everything except clip-derived rows
        for _, row in self.fusion_map.iterrows():
            fusion_component = str(row["fusion_component_name"]).strip()
            component_group = str(row["component_group"]).strip()
            scale_mode = str(row["scale_mode"]).strip()
            cost_group = str(row["cost_group"]).strip()
            notes = row["notes"] if "notes" in row and pd.notna(row["notes"]) else ""

            if scale_mode.lower() == "derived from springs":
                continue

            match = self.master_bom[
                self.master_bom["component_group"].astype(str).str.strip().str.lower()
                == component_group.lower()
            ]

            if match.empty:
                raise ValueError(
                    f"No matching component_group '{component_group}' found in master_template_spec.csv"
                )

            base_qty = float(match.iloc[0]["base_qty"])

            new_qty = self.scale_component(
                component_name=component_group,
                base_qty=base_qty,
                scaling_rule=scale_mode,
                scales=scales
            )

            if component_group.lower() == "springs":
                springs_new = new_qty

            fusion_rows.append({
                "fusion_component_name": fusion_component,
                "component_group": component_group,
                "scale_mode": scale_mode,
                "cost_group": cost_group,
                "base_qty": base_qty,
                "scaled_qty": new_qty,
                "notes": notes
            })

        # Second pass: clip-derived rows
        for _, row in self.fusion_map.iterrows():
            fusion_component = str(row["fusion_component_name"]).strip()
            component_group = str(row["component_group"]).strip()
            scale_mode = str(row["scale_mode"]).strip()
            cost_group = str(row["cost_group"]).strip()
            notes = row["notes"] if "notes" in row and pd.notna(row["notes"]) else ""

            if scale_mode.lower() == "derived from springs":
                match = self.master_bom[
                    self.master_bom["component_group"].astype(str).str.strip().str.lower()
                    == component_group.lower()
                ]

                if match.empty:
                    raise ValueError(
                        f"No matching component_group '{component_group}' found in master_template_spec.csv"
                    )

                base_qty = float(match.iloc[0]["base_qty"])

                new_qty = self.scale_component(
                    component_name=component_group,
                    base_qty=base_qty,
                    scaling_rule=scale_mode,
                    scales=scales,
                    springs_new=springs_new
                )

                fusion_rows.append({
                    "fusion_component_name": fusion_component,
                    "component_group": component_group,
                    "scale_mode": scale_mode,
                    "cost_group": cost_group,
                    "base_qty": base_qty,
                    "scaled_qty": new_qty,
                    "notes": notes
                })

        fusion_df = pd.DataFrame(fusion_rows)
        return fusion_df

    # ---------------------------------------------------
    # 8. COMPUTE COST
    # ---------------------------------------------------
    def compute_cost(self, bom_df):
        """
        cost_sheet.csv new columns:
            Component Name, Unit of Measurement, Cost per Unit (INR), Rate Type (Flat/Hourly/%), Value (INR or %)
        """
        cost_map = {}
        
        labor_cost = 0.0
        finishing_cost = 0.0
        overhead_pct = 12.0
        profit_pct = 15.0

        for _, row in self.cost_sheet.iterrows():
            if pd.isna(row["Component Name"]):
                continue
            name = str(row["Component Name"]).strip().lower()
            
            if name == "materials":
                continue
                
            if name == "labour":
                labor_cost = float(row["Value (INR or %)"])
            elif name == "finishing":
                finishing_cost = float(row["Value (INR or %)"])
            elif name == "overhead":
                overhead_pct = float(row["Value (INR or %)"]) * 100
            elif name == "profit margin":
                profit_pct = float(row["Value (INR or %)"]) * 100
            else:
                cost_map[name] = float(row["Cost per Unit (INR)"])

        cost_rows = []

        for _, row in bom_df.iterrows():
            component = str(row["component_group"]).strip()
            qty = float(row["new_qty"])

            pricing_key = component.lower()
            if pricing_key not in cost_map:
                raise ValueError(
                    f"Cost not found for component '{component}' in cost_sheet.csv"
                )

            unit_cost = cost_map[pricing_key]
            total_cost = qty * unit_cost

            cost_rows.append({
                "component_group": component,
                "new_qty": qty,
                "unit_cost": unit_cost,
                "total_cost": total_cost
            })

        cost_df = pd.DataFrame(cost_rows)

        material_cost = cost_df["total_cost"].sum() if not cost_df.empty else 0.0

        subtotal = material_cost + labor_cost + finishing_cost
        overhead = subtotal * (overhead_pct / 100.0)
        cost_after_overhead = subtotal + overhead
        profit = cost_after_overhead * (profit_pct / 100.0)
        final_price = cost_after_overhead + profit

        summary = {
            "material_cost": material_cost,
            "labor_cost": labor_cost,
            "finishing_cost": finishing_cost,
            "subtotal": subtotal,
            "overhead": overhead,
            "cost_after_overhead": cost_after_overhead,
            "profit": profit,
            "final_quotation_price": final_price
        }

        return cost_df, summary

    # ---------------------------------------------------
    # 9. SAVE OUTPUTS
    # ---------------------------------------------------
    def save_outputs(self, bom_df, cost_df, summary, fusion_df=None, output_prefix="quotation_output"):
        bom_path = os.path.join(self.bom_output_dir, f"{output_prefix}_bom.csv")
        quote_csv_path = os.path.join(self.quote_output_dir, f"{output_prefix}_cost.csv")
        quote_json_path = os.path.join(self.quote_output_dir, f"{output_prefix}_summary.json")

        bom_df.to_csv(bom_path, index=False)
        cost_df.to_csv(quote_csv_path, index=False)

        with open(quote_json_path, "w") as f:
            json.dump(summary, f, indent=4)

        fusion_csv_path = None
        if fusion_df is not None:
            fusion_csv_path = os.path.join(
                self.fusion_report_dir,
                f"{output_prefix}_fusion_scaled_components.csv"
            )
            fusion_df.to_csv(fusion_csv_path, index=False)

        return bom_path, quote_csv_path, quote_json_path, fusion_csv_path

    # ---------------------------------------------------
    # 10. FULL PIPELINE
    # ---------------------------------------------------
    def generate_quote(self, length_mm, width_mm, height_mm, sofa_type="3-seater", output_prefix="quotation_output"):
        self.load_data()

        scales, bom_df = self.generate_scaled_bom(length_mm, width_mm, height_mm, sofa_type)
        fusion_df = None # self.generate_fusion_scaled_components(scales) -> Disabled temporarily as map needs update
        cost_df, summary = self.compute_cost(bom_df)

        bom_path, quote_csv_path, quote_json_path, fusion_csv_path = self.save_outputs(
            bom_df=bom_df,
            cost_df=cost_df,
            summary=summary,
            fusion_df=fusion_df,
            output_prefix=output_prefix
        )

        result = {
            "input_dimensions": {
                "length_mm": length_mm,
                "width_mm": width_mm,
                "height_mm": height_mm
            },
            "scale_factors": {
                "SL": scales["SL"],
                "SW": scales["SW"],
                "SH": scales["SH"]
            },
            "summary": summary,
            "output_files": {
                "bom_csv": bom_path,
                "cost_csv": quote_csv_path,
                "summary_json": quote_json_path,
                "fusion_component_report": fusion_csv_path
            }
        }

        return result


if __name__ == "__main__":
    engine = SofaCostEngine()

    # Example test input
    result = engine.generate_quote(
        length_mm=2400,
        width_mm=950,
        height_mm=900,
        output_prefix="sample_3seater_quote"
    )

    print("\n===== QUOTATION RESULT =====")
    print(json.dumps(result, indent=4))