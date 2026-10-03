# StallionLink (Fusion 360 Script)

This folder contains a custom Fusion 360 Python script that automatically resizes your parametric `Sofa Internal Structure` CAD model based on the JSON output from the Stallion Web Application.

## Installation Instructions

1. Open Autodesk Fusion 360.
2. Go to **Design** > **UTILITIES** tab > **ADD-INS** > **Scripts and Add-Ins** (or press `Shift + S`).
3. Click the **Scripts** tab at the top.
4. Next to the "My Scripts" folder icon, click the **+ (Create)** button.
5. Choose **Python** as the language, and enter `StallionLink` as the script name. Click **Create**.
6. This will create a new default script. Right-click the newly created `StallionLink` in the list and select **Open File Location**.
7. Delete the default files in that folder and **copy and paste all the files from this directory (`StallionLink.py` and `StallionLink.manifest`) into that folder.**
8. Go back to Fusion 360. Your script is now installed!

## Usage

1. Open your master model (`Sofa Internal Structure.f3z`) in Fusion 360.
2. Press `Shift + S` to open Scripts and Add-Ins.
3. Select `StallionLink` under My Scripts and click **Run**.
4. A file dialog will appear. Navigate to your `Stallion/outputs/requests/` folder and select the `input_request.json` file generated for the specific quote.
5. The script will read the exact dimensions (`length`, `width`, `height`) and automatically update the User Parameters in your CAD model.

## Troubleshooting

If the script says it updated 0/3 parameters, you need to configure the parameter names. 
By default, the script looks for parameters named `Length`, `Width`, and `Height`. If your CAD file uses different names (e.g., `SofaLength`, `Depth`), open `StallionLink.py` in a text editor (like VS Code or Notepad) and change the configuration at the very top of the file:

```python
PARAM_NAME_LENGTH = "Length"
PARAM_NAME_WIDTH  = "Width"
PARAM_NAME_HEIGHT = "Height"
```
