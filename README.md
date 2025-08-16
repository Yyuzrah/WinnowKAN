# WinnowKAN
![image](https://github.com/Yyuzrah/WinnowKAN/blob/main/doc/WKAN01.jpg)
# Winnow-KAN: Single-Cell RNA-seq Location Recovery with Small-Gene-Set Spatial Transcriptomics 

This repo is for paper of Winnow-KAN.


Winnow-KAN is a computational method designed to enhance the integration of single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics data. By leveraging a Kolmogorov-Arnold Network architecture, Winnow-KAN predicts the spatial information of cells from scRNA-seq data using a significantly reduced set of genes. 
The core of our method is to extract the weights of each nonlinear part in the KAN operators to obtain the nonlinearity weights of the variables, thereby selecting genes (variables). Specifically, the KAN analyzes the contribution of each gene to the prediction of spatial locations by calculating the weights with its nonlinearity. Genes with higher nonlinearity weights are prioritized, allowing Winnow-KAN to select a small, impactful gene set.

# System Requirements

Python (>3.8) support packages: torch>=1.8, pandas>=1.4, numpy>=1.20, scipy, tqdm, scanpy>=1.5, anndata, sklearn, scikit-image

# Installation

```
git clone git@github.com/Yyuzrah/WinnowKAN.git
cd WinnowKAN
pip install -r requirements.txt
```

# Usage
```
from WKAN import *
```

Then follow these steps:

1. Import and standardize spatial transcriptomics data from diverse formats for downstream processing.

2. Train a predictive model to get cell spatial location using the processed expression data.

3. Extract and select variables (genes) from the trained model.

4. Produce artificial spatial transcriptomic datasets for validation or further experimentation.

# Cite
```
@article{Zhang2025,
  author = {Zhang, Yuyang and Zhang, Qihuang},
  title = {Winnow-KAN: single-cell RNA-seq location recovery with small-gene-set spatial transcriptomics},
  journal = {BMC Bioinformatics},
  volume = {26},
  number = {1},
  pages = {209},
  year = {2025},
  month = aug,
  day = {12},
  doi = {10.1186/s12859-025-06243-9},
  url = {https://doi.org/10.1186/s12859-025-06243-9},
  issn = {1471-2105},
  note = {Received: 09 February 2025; Accepted: 29 July 2025; Published: 12 August 2025}
}
```

