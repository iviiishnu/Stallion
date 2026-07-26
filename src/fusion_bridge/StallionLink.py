import adsk.core
import adsk.fusion
import traceback
import json
import os

# --- CONFIGURATION ---
# Change these strings to match the EXACT names of the User Parameters
# inside your master Sofa Internal Structure.f3z model.
PARAM_NAME_LENGTH = "Length"
PARAM_NAME_WIDTH  = "Width"
PARAM_NAME_HEIGHT = "Height"
# ---------------------

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct

        if not design:
            ui.messageBox('No active Fusion design', 'Stallion AI Link')
            return

        # 1. Prompt user to select the input_request.json file
        fileDialog = ui.createFileDialog()
        fileDialog.isMultiSelectEnabled = False
        fileDialog.title = "Select Stallion JSON Request File"
        fileDialog.filter = "JSON files (*.json)"
        fileDialog.filterIndex = 0
        dialogResult = fileDialog.showOpen()
        
        if dialogResult == adsk.core.DialogResults.DialogOK:
            filename = fileDialog.filename
        else:
            return  # User canceled

        # 2. Read the JSON file
        with open(filename, 'r', encoding='utf-8') as f:
            request_data = json.load(f)
            
        if "dimensions_mm" not in request_data:
            ui.messageBox("Invalid JSON file. Missing 'dimensions_mm' key.", 'Stallion AI Link')
            return

        dims = request_data["dimensions_mm"]
        target_length_mm = float(dims.get("length", 0))
        target_width_mm  = float(dims.get("width", 0))
        target_height_mm = float(dims.get("height", 0))

        if target_length_mm == 0 or target_width_mm == 0 or target_height_mm == 0:
            ui.messageBox("Invalid dimensions in JSON file. Dimensions cannot be 0.", 'Stallion AI Link')
            return

        # 3. Update the parameters in the Fusion 360 model
        # Fusion 360 internally uses centimeters for all length units.
        # So we convert mm to cm (divide by 10) or just pass it as an expression with "mm".
        
        user_params = design.userParameters
        updated_count = 0
        missing_params = []

        # Helper function to update a parameter safely
        def update_param(param_name, value_mm):
            nonlocal updated_count
            param = user_params.itemByName(param_name)
            if param:
                # Set the expression as a string with the unit (e.g. "2100 mm")
                param.expression = f"{value_mm} mm"
                updated_count += 1
            else:
                missing_params.append(param_name)

        # Update Length, Width, Height
        update_param(PARAM_NAME_LENGTH, target_length_mm)
        update_param(PARAM_NAME_WIDTH, target_width_mm)
        update_param(PARAM_NAME_HEIGHT, target_height_mm)

        # 4. Display results
        msg = f"Successfully read: {os.path.basename(filename)}\n"
        msg += f"Target Dimensions: {target_length_mm}x{target_width_mm}x{target_height_mm} mm\n\n"
        
        if updated_count == 3:
            msg += "✅ Successfully updated the parametric model!"
        else:
            msg += f"⚠️ Updated {updated_count}/3 parameters.\n"
            msg += f"Could not find these parameters: {', '.join(missing_params)}\n"
            msg += "\nPlease check the Configuration section at the top of StallionLink.py to ensure the names match your model."

        ui.messageBox(msg, 'Stallion AI Link')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
