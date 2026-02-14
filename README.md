# Tim! (Art Classifier with Grad-Cam Explainability)

## Setup and Requirements
- 1- Create a virutal environment with the following command.
`python -m venv .venv`
- 2- Activatie your venv. `.venv/Scripts/Activate`
- 3- Install dependecies with pip. `pip install -r requirements.txt` 
- 4- Install [GraphViz](https://graphviz.org/download/). Make sure it's added to PATH.
- 5- Download the [dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) and put it in the `data` folder. **you might need to rename the zip archive to "dataset.zip" manually.**
## File Structure
The following is the raw structure. Folders and files may be created as you run Tim.
```
├───data
├───documents
│   ├───phase1
│   └───phase2
├───models
├───notebooks
├───results
│   ├───charts
│   │   ├───convnext_tiny_finetuned
│   │   ├───efficientnet_b0_finetuned
│   │   ├───experiments
│   │   │   ├───efficientnet_b0_finetuned_larger_batch
│   │   │   ├───efficientnet_b0_finetuned_larger_batch_no_padding
│   │   │   └───efficientnet_b0_finetuned_no_padding
│   │   ├───resnet50_baseline
│   │   ├───resnet50_finetuned
│   │   └───structures
│   ├───convnext_tiny_finetuned
│   ├───efficientnet_b0_finetuned
│   ├───experiments
│   │   ├───efficientnet_b0_finetuned_larger_batch
│   │   ├───efficientnet_b0_finetuned_larger_batch_no_padding
│   │   └───efficientnet_b0_finetuned_no_padding
│   ├───resnet50_baseline
│   └───resnet50_finetuned
└───source
    ├───experiments
    ├───preprocessing
    └───training
```
## Order of Operation
This section regards result reproducibility.

Run scripts in the following order:
- 1- `source\preprocessing\prep.py`
- 2- `source\preprocessing\name_correction.py`
- 3- `source\preprocessing\resize.py`
- 4- `source\preprocessing\resize_padding.py`

Now you can run any of the scirpts in the `source\training` or `source\experiments` to train your desired model.

## Demo
To demo your trained models run the following notebook. `notebooks\interactive_demo.ipynb`

**For the demo to work you'll need `data\classes.csv` and at least one model in the `models\` directory.**
<img width="819" height="1079" alt="image" src="https://github.com/user-attachments/assets/0da7a963-8541-490f-bd90-e178a8f1e059" />

## Trained models
Trained models can be found [here](https://drive.google.com/drive/folders/1kdbbSOqvKl_gjqTs6UscykaOPi98Naie).
