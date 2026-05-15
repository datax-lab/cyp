# cyp
# BIN-PU: Bacterial CYP Compound–Protein Interaction Prediction

This repository contains the implementation of **BIN-PU**, a Positive-Unlabeled learning framework for predicting bacterial cytochrome P450 compound–protein interactions using only known positive samples.

---


## Workflow

### Step 1: Generate Unlabeled Data

Generate unlabeled protein–compound pairs by creating all possible combinations between unique proteins and unique compounds.

Known positive interactions are removed from these combinations.

```text
All protein–compound combinations
− Known positive interactions
= Unlabeled samples
```

The unlabeled samples may contain both likely positive and likely negative interactions.

---

### Step 2: Data Preprocessing

Preprocess the bacterial CYP protein–compound interaction data.

Run the following notebooks:

```text
preprocessing_cpi_and_ssnet.ipynb
preprocessing_for_padding_data_for_transformercpi.ipynb
```

This step prepares protein sequences, compound SMILES, known positive CPI labels, and model-specific input files.

---


### Step 3: Training CPI backbone models for Generating Pseudo Labels 

Use the scripts in:

```text
generating_pseudo_labels/
```

BIN-PU divides unlabeled samples into multiple bins and trains CPI classifiers using known positive samples and unlabeled samples.

---

### Step 4: Final Training CPI Backbone Models with Known Positives, pseudo positives and pseudo negatives by using Weighted Positive Loss

The weighted positive loss function is available in:

```text
custom_loss_function/
```

---


## Requirements

Install the required Python packages before running the code.

```text
python
numpy
pandas
scikit-learn
torch
gensim
rdkit
matplotlib
```

Additional packages may be required depending on the selected CPI backbone model.

---

## Citation

If you use this repository, please cite:

```text
Kim, K.-H., Yaganapu, A., Kosaraju, S., Bhatt, A., Luo, Y. L.,
Parsa, S. P., Park, J., Lee, H., Lee, J. H., Oh, T.-J., and Kang, M.
Prediction of bacterial protein–compound interactions with only positive samples.
Bioinformatics, 2026.
```
