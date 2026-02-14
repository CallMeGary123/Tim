# Tim! (Art Classifier with Grad-Cam Explainability)

## Setup and Requirements
- 1- Create a virutal environment with the following command.
`python -m venv .venv`
- 2- Activatie your venv. `.venv/Scripts/Activate`
- 3- Install dependecies with pip. `pip install -r requirements.txt` 
- 4- Install [GraphViz](https://graphviz.org/download/). Make sure it's added to PATH.
- 5- Download the [dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) and put it in the `data` folder. **you might need to rename the zip archive to "dataset.zip" manually.**

## Order of Operation
This section regards result reproducibility.

Run scripts in the following order:
- 1- `source\preprocessing\prep.py`
- 2- `source\preprocessing\name_correction.py`
- 3- `source\preprocessing\resize.py`
- 4- `source\preprocessing\resize_padding.py`

Now you can run any of the scirpts in the `source\training` or `source\experiments` to train your desired model.

## Demo
To demo your traind models run the following notebook. `notebooks\interactive_demo.ipynb`

**For the demo to work you'll need the `data\classes.csv` and at least one model in the `models\` directory.**

## Trained models
Trained models can be found [here](https://drive.google.com/drive/folders/1kdbbSOqvKl_gjqTs6UscykaOPi98Naie).