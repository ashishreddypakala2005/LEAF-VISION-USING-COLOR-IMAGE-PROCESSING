# Leaf Disease Segmentation using Color Image Processing

LeafVision is a Python/OpenCV project that demonstrates classical colour-based leaf and candidate disease-region segmentation. The Streamlit interface lets you explore the included PlantVillage images or upload a JPG/PNG image, adjust analysis settings, inspect masks and overlays, and download a CSV summary.

## Run the project on Windows

1. Clone this repository or download it from GitHub using **Code → Download ZIP**, then extract it.
2. Install Python 3.10 or newer.
3. Double-click `START_PROJECT.bat`. It installs the packages in `requirements.txt` and starts the app.
4. Open the local URL printed in the terminal, usually [http://localhost:8501](http://localhost:8501).

Alternatively, open PowerShell in the project folder and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The GitHub repository contains the project code and PlantVillage dataset. The app runs on the computer where it is started: `localhost` is not a public website, so another person cannot open your local URL from their own computer. To let your teacher run it, share the repository and have them follow these steps on their computer.

## Features

- Overview, Leaf Analysis, Segmentation Lab, Dataset & Model, and Project Methodology pages.
- Select an image from the included dataset or upload a JPG/PNG image.
- HSV and CIELAB colour processing, approximate leaf masks, candidate-region masks, overlays, and pipeline views.
- Adjustable detection sensitivity and minimum component size.
- Downloadable CSV analysis summaries and dataset class counts.

## Dataset and example results

The `dataset/` directory contains the PlantVillage class folders and images used by the app. Its structure is:

```text
dataset/ClassName/image.jpg
```

The `results/` directory contains a committed example generated from a Tomato Target Spot image using the **HSV only** colour-space option:

- `sample_pipeline.png` — input, masks, and overlay in one image.
- `sample_overlay.png` — candidate regions highlighted on the input.
- `sample_leaf_mask.png` and `sample_candidate_mask.png` — example masks.
- `sample_summary.csv` — example area measurements and settings.

These are fixed reference files; analyses run in the app are displayed in the interface and can be downloaded as a CSV. They are not automatically written to `results/`.

## Method and limitations

The pipeline converts the input to HSV and CIELAB colour spaces, estimates a leaf mask, thresholds candidate colours, cleans regions with morphological operations, estimates candidate area as a percentage of the estimated leaf area, and visualizes the intermediate results.

This is a colour-based segmentation demonstration, not a confirmed diagnosis or a trained deep-learning model. Its candidate regions can include healthy tissue or miss diseased tissue, depending on lighting, background, and image content. PlantVillage generally provides image-level class labels rather than pixel-level lesion masks; do not interpret the area estimate as validated disease severity or report accuracy, Dice, or IoU without suitable ground-truth annotations.
