# LeafVision — Leaf Disease Segmentation using Color Image Processing

Professional Streamlit UI for classical image processing with PlantVillage folder support.

## Features
- Overview dashboard matching the supplied UI reference.
- Sidebar navigation: Overview, Leaf Analysis, Segmentation Lab, Dataset & Model, Project Methodology.
- Upload a custom image or choose an image from the local PlantVillage dataset.
- HSV and CIELAB colour processing, leaf mask, yellow/brown candidate regions and morphology.
- Candidate-area estimate, masks, overlay, pipeline preview and CSV export.
- Local dataset counts and class browser.

## Run on Windows
1. Extract this ZIP.
2. Put PlantVillage class folders inside `dataset/`.
3. Double-click `START_PROJECT.bat`, or run in PowerShell:
   `py -m pip install -r requirements.txt`
   `py -m streamlit run app.py`

## Dataset structure
`dataset/ClassName/image.jpg`

The dataset is not included in the ZIP because it is large. Use your existing PlantVillage download.

## Method
RGB input → HSV/CIELAB conversion → approximate leaf mask → yellow/brown candidate thresholding → morphological cleanup → pixel-area estimate → overlay and masks.

## Limitation
This is colour-based candidate segmentation, not a confirmed diagnosis or a trained deep-learning model. PlantVillage generally has image-level class labels rather than pixel-level lesion masks. Do not report validated accuracy, Dice, IoU or disease severity without suitable ground-truth evaluation.
