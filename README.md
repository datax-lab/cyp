
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

The preprocessing procedure is adapted from the respective CPI backbone model papers (TransformerCPI, SSNet, CPIprediction, MulinforCPI) to ensure that the input format is consistent with each model requirement.

This step prepares protein sequences, compound SMILES, known positive CPI labels, and model-specific input files.

---


### Step 3: Training CPI backbone models for Generating Pseudo Labels 

Use the scripts in:

```text
generating_pseudo_labels/
```

BIN-PU divides unlabeled samples and known positives into multiple bins and trains CPI classifiers using known positive samples and unlabeled samples.

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

