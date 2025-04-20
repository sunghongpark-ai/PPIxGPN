---

# PPIxGPN

---

## Overview

### Title
PPIxGPN: plasma proteomic profiling of neurodegenerative biomarkers with protein-protein interaction-based explainable graph propagational network

### Abstract
Neurodegenerative diseases involve progressive neuronal dysfunction, requiring the identification of specific pathological features for accurate diagnosis. While cerebrospinal fluid analysis and neuroimaging are commonly used, their invasive nature and high costs limit clinical applicability. Recently advances in plasma proteomics offer a less invasive and cost-effective alternative, further enhanced by machine learning (ML). However, most ML-based studies overlook synergetic effects from protein-protein interactions (PPIs), which play a key role in disease mechanisms. Although graph convolutional network and its extensions can utilize PPIs, they rely on locality-based feature aggregation, overlooking essential components and emphasizing noisy interactions. Moreover, expanding those methods to cover broader PPIs results in complex model architectures that reduce explainability, which is crucial in medical ML models for clinical decision-making. To address these challenges, we propose Protein-Protein Interaction-based eXplainable Graph Propagational Network (PPIxGPN), a novel ML model designed for plasma proteomic profiling of neurodegenerative biomarkers. PPIxGPN captures synergetic effects between proteins by integrating PPIs with independent effects of proteins, leveraging globality-based feature aggregation to represent comprehensive PPI properties. This process is implemented using a single graph propagational layer, enabling PPIxGPN to be configured by shallow architecture, thereby PPIxGPN ensures high model explainability, enhancing clinical applicability by providing interpretable outputs. Experimental validation on the UK Biobank dataset demonstrated the superior performance of PPIxGPN in neurodegenerative risk prediction, outperforming comparison methods. Furthermore, the explainability of PPIxGPN facilitated detailed analyses of the discriminative significance of synergistic effects, the predictive importance of proteins, and the longitudinal changes in biomarker profiles, highlighting its clinical relevance.

<b>Keywords</b>: Neurodegenerative diseases; Blood-based biomarkers; Protein-protein interaction; Graph neural network; Explainable machine learning.

![Abstract](https://github.com/user-attachments/assets/827e62a4-f427-4b9e-804a-e4a6eed2be04)

---

## Data description

### Study participants
The dataset used in this study was sourced from the UK Biobank (UKB), which recruited over 500,000 participants aged 39–70 between 2006 and 2010, with long-term monitoring of their health outcomes (additional details can be at: https://biobank.ndph.ox.ac.uk/showcase/). From the UKB participants, 906 individuals with complete data on four neurodegenerative biomarkers (Aβ42/40, GFAP, NfL, and pTau181) were selected for the analytical cohort. Given that higher levels of GFAP, NfL, and pTau indicate increased risk, while lower levels of Aβ42/40 signify greater risk, the reciprocal of Aβ42/40 (Aβ40/42) was used in this study to maintain consistency in predictions. For each participant, the positivity and negativity for the biomarkers were determined using the ‘cutoff’ implemented in R (https://github.com/choisy/cutoff). Of the participants, 783 had follow-up data on the biomarkers, with an average follow-up duration of 3.25 years, enabling the assessment of longitudinal changes. The demographic characteristics of the study participants are presented as below.

|      Characteristics     | Total participants (N=906) | Follow-up participants (N=783) |            |          |
|:------------------------:|:--------------------------:|:------------------------------:|:----------:|:--------:|
|                          |                            |            Baseline            |  Follow-up | P-value* |
|      Female, No. (%)     |         491 (54.2)         |          422   (53.9)          |            |     –    |
|   Education, med. (IQR)  |          7 (3–17)          |           7   (3–17)           |            |     –    |
|    Age, med. (IQR), yr   |         58 (54–64)         |           58 (53–64)           | 61 (57–67) |  <0.0001 |
|  Aβ-positivity, No. (%)  |         192 (21.2)         |           162 (20.7)           | 265 (33.8) |  <0.0001 |
| GFAP-positivity, No. (%) |         175 (19.3)         |           155 (19.8)           | 205 (26.2) |  0.0027  |
|  NfL-positivity, No. (%) |         227 (25.1)         |           189 (24.1)           | 236 (30.1) |  0.0075  |
| pTau-positivity, No. (%) |         212 (23.4)         |           182 (23.2)           | 163 (20.8) |   0.247  |

### Plasma samples
The plasma samples of participants were profiled by the Olink Platform, where 1,463 proteins were totally assayed (further details can be found at: https://olink.com/). For the initial data for protein expression, proteins with a missing frequency greater than 5% were excluded, and missing values were estimated using the k-nearest neighbor method. Subsequently, protein expression levels were standardized through Z-score normalization and then scaled using the logistic function.

---

## Model implementation

### Overall process
Given that $d$ and $n$ are the numbers of target proteins and study participants, respectively, the data matrix for the independent effects, denoted by $X\in\mathbb{R}^{d×n}$, consists of the preprocessed expression data for the target proteins. PPIxGPN first extracts the synergetic effects of target proteins, denoted as $Z\in\mathbb{R}^{d×n}$, by propagating the independent effects onto the PPI network $W\in\mathbb{R}^{d×d}$. In this process, the propagation parameter $ϕ\in\mathbb{R}^{d}$ is applied to the proteins for individually controlling the intensity of propagation. Subsequently, the synergetic effect is applied to the estimation parameter, denoted as $Θ_{\*}\in\mathbb{R}^{d}$ (${\*}$: Aβ, GFAP, NfL, pTau), and the individual risks for neurodegenerative biomarkers, denoted as $P_{\*}\in\mathbb{R}^{n}$, are derived. The proposed method encompasses two-layered model architecture, including two parameter sets, $ϕ$ and $Θ_{\*}$, which are optimized by comparing the predicted risk $P_{\*}$ with the real diagnosis, denoted as $Y_{\*}\in\mathbb{R}^{n}$. Notations for PPIxGPN are summarized as below.

| Notation                  | Description                             |
| :-----------------------: | :-------------------------------------: |
| $d$                       | Number of study participants            |
| $n$                       | Number of target proteins               |
| $X\in\mathbb{R}^{d×n}$    | Independent effect of target proteins   |
| $Z\in\mathbb{R}^{d×n}$    | Synergetic effect of target proteins    |
| $ϕ\in\mathbb{R}^{d}$      | Propagation parameter set of PPIxGPN    |
| $Θ_{\*}\in\mathbb{R}^{d}$ | Estimation parameter set of PPIxGPN     |
| $P_{\*}\in\mathbb{R}^{n}$ | Predicted risk set for biomarkers       |
| $Y_{\*}\in\mathbb{R}^{n}$ | Real diagnosis label set for biomarkers |
| ${\*}$                    | Aβ, GFAP, NfL, pTau                     |

### Code description

<b>MATLAB (version R2024b)</b>

- <b>Data setting</b>
  - Xdata: Independent effect of target proteins
  - Ydata: Real diagnosis label set for biomarkers
  - Split two data into train/valid/test sets

- <b>Parameter setting</b>
  - Uppi: Propagation parameter
  - Babt, Bgfa, Bnfl, Btau: Estimation parameters
  - Babt, Bgfa, Bnfl, Btau: Estimation parameters

- <b>Model parameter</b>
  - epoch: maximum number of epoch
  - rate: learning rate
  - gamma: regularization coefficient

- <b>Model implementation</b>
  - Model training
  - Risk estimation

---
