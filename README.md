
# Grapevine Microbiome Machine Learning 🪴🌿🌱🍇

Code and workflows associated with:

**Machine Learning Reveals Scion and Rootstock Signatures in the Global Grapevine Soil Microbiome**

## Authors

Lakshay Anand¹, Thanos Gentimis², Allan B. Downie¹, Carlos M. Rodriguez Lopez¹*

### Affiliations

¹ Department of Horticulture, College of Agriculture, Food and Environment, University of Kentucky, Lexington, KY, USA

² Department of Soil and Crop Sciences, College of Agriculture and Life Sciences, Texas A&M University, College Station, Texas, USA

*Corresponding author

Email: carlos.rodriguez@uky.edu


## Contents

This repository provides code for all nine machine-learning algorithms described in the manuscript:

1. Random Forest
2. AdaBoost
3. Gradient Boosting Machine
4. Support Vector Machine, linear kernel
5. Support Vector Machine, radial basis function kernel
6. Gaussian Naive Bayes
7. Bernoulli Naive Bayes
8. k-Nearest Neighbors
9. Neural Network

It also contains Neural Network scripts for continent, country, scion cultivar, rootstock, and scion-rootstock combination classification, plus the SHAP interpretability workflow.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── data/
│   └── README.md
└── scripts/
    ├── analysis_config.py
    ├── data_utils.py
    ├── evaluation_utils.py
    ├── classical_models.py
    ├── nn_common.py
    ├── train_nn_approachA.py
    ├── train_nn_approachB_continent.py
    ├── train_nn_approachB_country.py
    ├── train_nn_approachB_scion.py
    ├── train_nn_approachB_rootstock.py
    ├── train_nn_approachB_scion_rootstock.py
    ├── shap_analysis.py
    └── original_code.py
```

## Input data

The scripts expect a pandas pickle file named:

```text
data/FeatureDataWoOut.pkl
```
The file can be downloaded from:

The file must contain CLR-normalized microbiome features followed by six metadata columns. The required target columns are `Country`, `Continent`, `Grape_variety`, `Rootstock`, and `comb`. The `comb` field stores scion-rootstock combinations.

The analysis does **not** apply additional feature scaling because the microbiome feature matrix is already centered log-ratio transformed.


## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate grapevine-microbiome-ml
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## TensorFlow compatibility

The Approach B workflow uses `tf.keras.losses.CategoricalFocalCrossentropy`. Use a TensorFlow release that provides this loss. The supplied environment uses TensorFlow 2.12. If reproducing the historical software environment from the manuscript with TensorFlow 2.8.0, this loss is not guaranteed to be available and a compatible focal-loss implementation may be required.

## Hardware note

The original analysis was performed with an NVIDIA Tesla V100-SXM2-32GB GPU in a high-performance computing environment. The CLR-normalized feature matrix is large, and the Neural Network and SHAP analyses may be slow or may exceed the memory available on a regular desktop computer. A GPU and high-memory compute node are strongly recommended.

## Running the analyses

The scripts intentionally use a simple editable configuration block instead of command-line arguments. Open the relevant file, review the constants near the top, and run it from the repository root.

### Classical models

Set `CLASS_LABEL` and `APPROACH` in `scripts/classical_models.py`, then run:

```bash
python scripts/classical_models.py
```

This script evaluates Random Forest, AdaBoost, Gradient Boosting Machine, linear SVM, radial SVM, Gaussian Naive Bayes, Bernoulli Naive Bayes, and k-Nearest Neighbors. It writes per-model metrics, classification reports, and normalized confusion matrices to `results/classical_<target>/`.

### Neural Network, Approach A

Set `CLASS_LABEL` in `scripts/train_nn_approachA.py`, then run:

```bash
python scripts/train_nn_approachA.py
```

### Neural Network, Approach B

Run the script matching the target:

```bash
python scripts/train_nn_approachB_continent.py
python scripts/train_nn_approachB_country.py
python scripts/train_nn_approachB_scion.py
python scripts/train_nn_approachB_rootstock.py
python scripts/train_nn_approachB_scion_rootstock.py
```

Each script reports the retained class distribution, fold-level F1-macro, F1-weighted, and balanced accuracy. It saves the fold metrics, classification report, normalized confusion matrix, class mapping, and final trained model in a target-specific results directory.

### SHAP analysis

After running the Approach B scion script, run:

```bash
python scripts/shap_analysis.py
```

The SHAP script reads the saved scion model and its saved background/evaluation matrices. It exports positive and negative mean SHAP rankings for every cultivar and a complete mean-SHAP table.

## Outputs

Typical outputs include:

- `metrics.csv`: fold-level or model-level performance
- `classification_report.csv`: precision, recall, and F1 results
- `confusion_matrix_normalized.png`: normalized confusion matrix
- `label_mapping.json`: integer-to-class mapping
- `model.keras`: saved Neural Network model
- `shap_mean_all_features.csv`: mean SHAP values for every feature and class
- `shap_top_positive.csv` and `shap_top_negative.csv`: ranked SHAP features

## Reproducibility notes

Random seeds are fixed where supported. The scripts report class counts after filtering and preserve the manuscript label names. Approach A retains the manuscript-specified classes with at least three samples. Approach B uses the manuscript-specified retained classes with at least 15 samples. Rootstock and scion-rootstock analyses remove unknown rootstocks and the own-rooted Shiraz category, following the original analysis code.

## Citation

Please cite the associated manuscript. Update `CITATION.cff` with the final journal, year, DOI, and version after acceptance.

## Original code preservation

The analysis workflows in this repository have been refactored with AI assistance to improve clarity and reproducibility. The original analysis code is preserved in [`scripts/original_code.py`](scripts/original_code.py).
