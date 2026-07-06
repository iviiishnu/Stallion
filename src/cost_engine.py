import os
import math
import json
import pandas as pd


class SofaCostEngine:
    def __init__(self, base_dir=".."):
        """
        base_dir should point to sofa_project when running from src/

        Example structure:
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
        # project root
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

    # ---------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------
    def load_data(self):
        self.master_dimensions = pd.read_csv(self.master_dim_path)
        self.master_bom = pd.read_csv(self.master_bom_path)
        self.cost_sheet = pd.read_csv(self.cost_sheet_path)

        # Fusion mapping is optional for safety, but expected in your project now
        if os.path.exists(self.fusion_map_path):
            self.fusion_map = pd.read_csv(self.fusion_map_path)
        else:
            self.fusion_map = None

        print("Loaded:")
        print(f"  master_dimensions -> {self.master_dim_path}")
        print(f"  master_template_spec -> {self.master_bom_path}")
        print(f"  cost_sheet -> {self.cost_sheet_path}")
        if self.fusion_map is not None:
            print(f"  fusion_component_map -> {self.fusion_map_path}")
        else:
            print("  fusion_component_map -> NOT FOUND (Fusion report will be skipped)")

    # ---------------------------------------------------
    # 2. GET BASE DIMENSIONS
    # ---------------------------------------------------
    def get_base_dimensions(self):
        """
        Expected master_dimensions.csv format:
        parameter,symbol,value_mm
        sofa_length,L0,2100
        sofa_width,W0,900
        sofa_height,H0,850
        ...
        """
        dim_map = {}
        for _, row in self.master_dimensions.iterrows():
            dim_map[str(row["symbol"]).strip()] = float(row["value_mm"])

        L0 = dim_map["L0"]
        W0 = dim_map["W0"]
        H0 = dim_map["H0"]

        return L0, W0, H0

    # ---------------------------------------------------
    # 3. COMPUTE SCALE FACTORS
    # ---------------------------------------------------
    def compute_scale_factors(self, length_mm, width_mm, height_mm):
        L0, W0, H0 = self.get_base_dimensions()

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
    # 4. SURFACE AREA RATIO FOR FABRIC
    # ---------------------------------------------------
    @staticmethod
    def surface_area_ratio(L1, W1, H1, L0, W0, H0):
        num = (L1 * W1) + (L1 * H1) + (W1 * H1)
        den = (L0 * W0) + (L0 * H0) + (W0 * H0)
        return num / den

    # ---------------------------------------------------
    # 5. SCALE ONE COMPONENT
    # ---------------------------------------------------
    def scale_component(self, component_name, base_qty, scaling_rule, scales, springs_new=None):
        SL = scales["SL"]
        SW = scales["SW"]
        SH = scales["SH"]

        L0, W0, H0 = scales["L0"], scales["W0"], scales["H0"]
        L1, W1, H1 = scales["L1"], scales["W1"], scales["H1"]

        rule = str(scaling_rule).strip().lower()

        if rule == "sl*sw*sh":
            return base_qty * SL * SW * SH

        elif rule == "sl*sh":
            return base_qty * SL * SH

        elif rule == "sl*sw":
            return base_qty * SL * SW

        elif rule == "sw*sh":
            return base_qty * SW * SH

        elif rule == "surface_area_ratio":
            ratio = self.surface_area_ratio(L1, W1, H1, L0, W0, H0)
            return base_qty * ratio

        elif rule == "round(base*sl)":
            return math.ceil(base_qty * SL)

        elif rule == "round((45/11)*springs_new)":
            if springs_new is None:
                raise ValueError("springs_new must be provided for clip calculation")
            return math.ceil((45 / 11) * springs_new)

        else:
            raise ValueError(
                f"Unknown scaling rule '{scaling_rule}' for component '{component_name}'"
            )

    # ---------------------------------------------------
    # 6. GENERATE SCALED BOM
    # ---------------------------------------------------
    def generate_scaled_bom(self, length_mm, width_mm, height_mm):
        scales = self.compute_scale_factors(length_mm, width_mm, height_mm)

        scaled_rows = []
        springs_new = None

        # First pass: everything except clips
        for _, row in self.master_bom.iterrows():
            component = str(row["component_group"]).strip()
            base_qty = float(row["base_qty"])
            unit = row["unit"]
            scaling_rule = row["scaling_rule"]

            if str(scaling_rule).strip().lower() == "round((45/11)*springs_new)":
                continue

            new_qty = self.scale_component(component, base_qty, scaling_rule, scales)

            if component.lower() == "springs":
                springs_new = new_qty

            scaled_rows.append({
                "component_group": component,
                "base_qty": base_qty,
                "unit": unit,
                "scaling_rule": scaling_rule,
                "new_qty": new_qty
            })

        # Second pass: clips
        for _, row in self.master_bom.iterrows():
            component = str(row["component_group"]).strip()
            base_qty = float(row["base_qty"])
            unit = row["unit"]
            scaling_rule = row["scaling_rule"]

            if str(scaling_rule).strip().lower() == "round((45/11)*springs_new)":
                new_qty = self.scale_component(
                    component, base_qty, scaling_rule, scales, springs_new=springs_new
                )
                scaled_rows.append({
                    "component_group": component,
                    "base_qty": base_qty,
                    "unit": unit,
                    "scaling_rule": scaling_rule,
                    "new_qty": new_qty
                })

        bom_df = pd.DataFrame(scaled_rows)

        # keep same order as master BOM CSV
        component_order = self.master_bom["component_group"].tolist()
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
        Generate a Fusion-aware component scaling report using fusion_component_map.csv.

        Expected fusion_component_map.csv columns:
        fusion_component_name,component_group,scale_mode,cost_group,notes
        """
        if self.fusion_map is None:
            return None

        fusion_rows = []
        springs_new = None

        # First pass: compute everything except clip rows
        for _, row in self.fusion_map.iterrows():
            fusion_component = str(row["fusion_component_name"]).strip()
            component_group = str(row["component_group"]).strip()
            scale_mode = str(row["scale_mode"]).strip()
            cost_group = str(row["cost_group"]).strip()
            notes = row["notes"] if "notes" in row and pd.notna(row["notes"]) else ""

            # Find matching BOM row by component_group
            match = self.master_bom[
                self.master_bom["component_group"].astype(str).str.strip().str.lower()
                == component_group.lower()
            ]

            if match.empty:
                raise ValueError(
                    f"No base BOM component found for fusion component group '{component_group}'"
                )

            base_qty = float(match.iloc[0]["base_qty"])

            # Skip clip row for second pass
            if scale_mode.lower() == "round((45/11)*springs_new)":
                continue

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

        # Second pass: clip rows that depend on springs_new
        for _, row in self.fusion_map.iterrows():
            fusion_component = str(row["fusion_component_name"]).strip()
            component_group = str(row["component_group"]).strip()
            scale_mode = str(row["scale_mode"]).strip()
            cost_group = str(row["cost_group"]).strip()
            notes = row["notes"] if "notes" in row and pd.notna(row["notes"]) else ""

            if scale_mode.lower() == "round((45/11)*springs_new)":
                match = self.master_bom[
                    self.master_bom["component_group"].astype(str).str.strip().str.lower()
                    == component_group.lower()
                ]

                if match.empty:
                    raise ValueError(
                        f"No base BOM component found for fusion component group '{component_group}'"
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
        Expected cost_sheet.csv format:
        material_component,unit_cost,unit
        Wood Frame,5000,base unit
        ...
        Labor,4000,per sofa
        PVD / Finishing,2500,per sofa
        Overhead %,10,%
        Profit Margin %,20,%
        """
        cost_map = {}
        for _, row in self.cost_sheet.iterrows():
            name = str(row["material_component"]).strip().lower()
            cost_map[name] = float(row["unit_cost"])

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

        material_cost = cost_df["total_cost"].sum()
        labor_cost = cost_map["labor"]
        finishing_cost = cost_map["pvd / finishing"]
        overhead_pct = cost_map["overhead %"]
        profit_pct = cost_map["profit margin %"]

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
    def generate_quote(self, length_mm, width_mm, height_mm, output_prefix="quotation_output"):
        self.load_data()

        scales, bom_df = self.generate_scaled_bom(length_mm, width_mm, height_mm)
        fusion_df = self.generate_fusion_scaled_components(scales)
        cost_df, summary = self.compute_cost(bom_df)

        bom_path, quote_csv_path, quote_json_path, fusion_csv_path = self.save_outputs(
            bom_df,
            cost_df,
            summary,
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