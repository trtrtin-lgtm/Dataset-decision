# Open Quantum System and Classical Markov Decision Modeling of Human Deliberation Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Science Framework](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/)
[![Dataset: Cross-Paradigm](https://img.shields.io/badge/Data-3%2C733_Trials_|_59_Subjects-green.svg)](https://github.com/trtrtin-lgtm/Dataset-decision)

This repository hosts the official experimental datasets, preprocessing pipelines, numerical simulation routines, and computational cognitive modeling scripts for the study:

> **"Open Quantum System Models Show a Consistent Parsimony Advantage over Classical Markov Chains in Visual Deliberation Dynamics: A Cross-Paradigm Benchmark"**  
> *Authors:* Tran Trung Tin (First & Corresponding), Le Thuy Trang, Pham My Xuan, Do Tuong Phu, Le Nhat Tan  
> *Affiliations:*  
> 1. Faculty of Applied Science, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, Ho Chi Minh City, Vietnam  
> 2. Vietnam National University Ho Chi Minh City (VNU-HCM), Ho Chi Minh City, Vietnam  
> *Corresponding Author:* Tran Trung Tin (`trtrtin@hcmut.edu.vn`)

---

## 📌 Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Datasets Included](#2-datasets-included)
- [3. Repository Structure](#3-repository-structure)
- [4. Data Dictionary & Formats](#4-data-dictionary--formats)
- [5. Mathematical Models](#5-mathematical-models)
- [6. Installation & Quickstart](#6-installation--quickstart)
- [7. Reproducing Manuscript Results & Figures](#7-reproducing-manuscript-results--figures)
- [8. Key Findings & Empirical Benchmarks](#8-key-findings--empirical-benchmarks)
- [9. Ethics Statement & Governance](#9-ethics-statement--governance)
- [10. Citation](#10-citation)

---

## 1. Executive Summary

A fundamental challenge in cognitive science and mathematical psychology is characterizing the temporal dynamics of deliberation before a choice is made. While classical cognitive models typically represent evidence accumulation via continuous-time Markov chains (CTMC) or drift-diffusion processes (DDM), quantum-like cognitive models formalize hesitation as an open quantum system evolving within a complex Hilbert space governed by the Lindblad-Gorini-Kossakowski-Sudarshan (GKSL) master equation.

This project delivers an end-to-end computational and empirical evaluation comparing **Open Quantum Systems (OQS4, $k=3$)** against **Classical Continuous-Time Markov Chains (CTMC, $k=4$)**:
1. **In-House Moral Dilemmas Cohort ($N=20$ participants, 82 clean trials):** High-conflict binary ethical decisions recorded via a calibrated 60 Hz infrared eye-tracker across 15 response-locked deliberation bins ($3,000$ ms window prior to choice).
2. **Empirical Question-Order Testing ($N=173$ participants):** Independent survey evaluating question order effects and confirming the Quantum Question (QQ) equality invariant.
3. **External Cross-Paradigm Benchmark ($N=39$ participants, 3,651 trials):** Canonical food-choice dataset of Krajbich et al. (2010, *Nature Neuroscience*), validating generalizability from ethical conflict to rapid perceptual consumer purchasing.

**Primary Findings:**
- **Parsimony Advantage:** OQS4 decisively outperforms CTMC under the Bayesian Information Criterion (BIC) in **82.9%** of in-house trials and **80.4%** of external benchmark trials (winning across 39/39 individual participants).
- **Equivalent Descriptive Fit:** Unpenalized sum-of-squared errors (SSE) are statistically indistinguishable ($p = 0.384$, mean SSE ratio = 1.002), demonstrating that OQS4's victory is driven by **Occam's razor (algebraic coordinate compression saving 1 degree of freedom)** rather than superior curve-fitting.
- **Universality of the Overdamped Regime:** Across all 39 participants, the dissipation-to-tunneling ratio $\max(\gamma)/d \approx 7.32$ forces instantaneous decoherence ($\tau_{\text{dec}} \approx 0.14$ s), causing quantum dynamics to project onto an effectively classical dissipative manifold.
- **Boundary Parameter Artifacts:** Pervasive parameter boundary hits ($93.9\%$ in-house, $93.5\%$ Krajbich) were resolved via Empirical Bayes maximum a posteriori (MAP) shrinkage, regularizing 100% of parameters into the interior domain while retaining a 76.8% selection majority for OQS4.

---

## 2. Datasets Included

| Dataset | Sample Size | Paradigm | Temporal Structure | Format | Primary Files |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-House Dilemmas Cohort** | 20 subjects, 82 clean trials (159 raw) | Moral & Academic Dilemmas (Career, Integrity, Retake) | 15 uniform 200 ms response-locked bins ($t \in [-3000, 0]$ ms) | CSV, SQLite, Excel | `data/raw_eyetracking/`, `DATN(1).db`, `OQS_Results_Auto.xlsx` |
| **Question-Order Survey** | 173 participants (Group 1: 88, Group 2: 85) | Academic Integrity & Retake Scenarios | Two presentation orders (Order 1: A $\to$ B; Order 2: B $\to$ A) | CSV, XLSX | `data/survey/QOE_Survey_Data.xlsx` |
| **Krajbich et al. (2010) Benchmark** | 39 subjects, 3,651 valid trials (12,181 fixations) | Binary Food Snack Choice | 10 proportional deliberation intervals ($t \in [0, \text{RT}]$) | Stata (`.dta`), Excel | `data/krajbich/data_nature2010.dta`, `results/Krajbich_2010_Validation_Results.xlsx` |

---

## 3. Repository Structure

```text
Dataset-decision/
├── README.md                           <- Master repository documentation (this file)
├── requirements.txt                    <- Python dependency specification
├── LICENSE                             <- MIT Open Source License
├── history.md                          <- Chronological research log, audit trail, and methodological lessons
│
├── data/                               <- Raw and preprocessed experimental datasets
│   ├── raw_eyetracking/                <- Raw OGAMA coordinate streams (x, y, pupil, timestamp)
│   ├── processed_bins/                 <- 15 response-locked bin dwell ratios per trial
│   ├── survey/                         <- Question-order survey responses (N = 173)
│   └── krajbich_2010/                  <- Cleaned data from Krajbich et al. (2010, Nature Neuroscience)
│
├── GUI_App/                            <- Core computational engines and GUI tools
│   ├── modules/
│   │   ├── data_processor.py           <- I-VT segmentation, blink interpolation, 6-AOI mapping, Gate 2 gating
│   │   ├── oqs_models.py               <- Lindblad master equation solvers (OQS4, OQS6, Hybrid)
│   │   ├── ctmc_model.py               <- 4-parameter continuous-time Markov chain master equation
│   │   └── optimization.py             <- L-BFGS-B, Nelder-Mead, SSE loss, BIC/AIC penalty evaluation
│   ├── bayesian_informative_priors.py  <- Synthetic parameter recovery & Empirical Bayes MAP shrinkage
│   ├── krajbich_validation_pipeline.py <- Cross-paradigm benchmark ingestion and trial-by-trial fitting
│   ├── qq_equality_analysis.py         <- Fisher's exact tests and quantum question equality verification
│   └── visualization.py                <- Figure plotting routines (Figures 1-7)
│
├── results/                            <- Output tables, model selection logs, and high-res figures
│   ├── OQS_Results_Auto.xlsx           <- Fitting results across 82 in-house trials (BIC, AIC, SSE, parameters)
│   ├── Krajbich_2010_Validation_Results.xlsx <- 3,651 trial-level fits on the Krajbich benchmark
│   ├── Bayesian_Informative_Priors_Results.xlsx <- Parameter recovery distributions and MAP shrinkage logs
│   ├── fig1_apparatus_setup.png        <- Figure 1: Eyetech VT3 apparatus & ergonomic geometry
│   ├── fig2_data_pipeline.png          <- Figure 2: End-to-end calibration and quality control flowchart
│   ├── fig3_gaze_cascade.png           <- Figure 3: Empirical gaze cascade trajectories (15 bins)
│   ├── fig4_model_comparison.png       <- Figure 4: BIC win counts and SSE distributions
│   ├── fig5_order_effects.png          <- Figure 5: Joint probabilities & QQ equality diagonal
│   ├── fig6_parameter_recovery.png     <- Figure 6: Synthetic recovery under low and realistic noise
│   └── fig7_krajbich_validation.png    <- Figure 7: Benchmark trajectories and Delta-BIC distribution
│
├── manuscripts/                        <- Preprints and finalized submission documents
│   ├── Manuscript_Nature_English_Final.pdf  <- Complete English manuscript (Publication-ready, 26 pages)
│   ├── Manuscript_Nature_English_Final.docx <- Word version formatted with Nature styling
│   ├── BaoCao_Nature_TiengViet_Final.pdf    <- Complete Vietnamese report (26 pages)
│   └── BaoCao_Nature_TiengViet_Final.docx   <- Word version in Vietnamese
│
└── latex/                              <- Production LaTeX source bundle
    ├── main_en.tex                     <- English LaTeX manuscript
    ├── main_vi.tex                     <- Vietnamese LaTeX manuscript
    ├── references.bib                  <- BibTeX bibliography database
    ├── latex_manuscript.zip            <- Complete packaged zip archive for journal submission
    └── figures/                        <- High-resolution figure assets for LaTeX compilation
```

---

## 4. Data Dictionary & Formats

### 4.1 Eye-Tracking Raw Logs (`data/raw_eyetracking/`)
- `Subject`: Participant identifier (`P01` to `P20`).
- `Scenario`: Decision dilemma context (e.g., `Job_Choice`, `Academic_Integrity`, `Course_Retake`).
- `Time`: Elapsed trial time (ms) sampled at 60 Hz.
- `GazePosX`, `GazePosY`: Normalized gaze coordinate positions on the 1440 × 900 px active display.
- `PupilDia`: Baseline-corrected pupil diameter measurement.
- `Event`: Oculomotor classification via I-VT (`Fixation`, `Saccade`, `Blink`).
- `AOI`: Spatial Area of Interest:
  - `op1a`, `op1b`: Sub-regions corresponding to Option 1 text.
  - `op2a`, `op2b`: Sub-regions corresponding to Option 2 text.
  - `target1`, `target2`: Central scenario text and hesitation prompt areas.
  - `Out-of-AOI`: Gaze outside predefined bounding boxes.

### 4.2 Response-Locked Time Bins (`data/processed_bins/`)
- `Bin`: Index from 1 to 15 (uniform 200 ms windows covering $t \in [-3000, 0]$ ms prior to response).
- `P_Option1`: Dwell-time fraction allocated to Option 1 within bin $t$.
- `P_Option2`: Dwell-time fraction allocated to Option 2 within bin $t$.
- `P_Neutral`: Dwell-time fraction allocated to deliberative hesitation / scenario prompt.
- `Valid_Ratio`: Ratio of tracked samples relative to theoretical maximum ($T_{\text{AOI}} / T_{\text{bin}}$).
- `Trial_OK`: Boolean quality gate flag (`True` if $\text{Ratio}_{\text{AOI}} \ge 0.50$ and cumulative track loss $< 30\%$).

---

## 5. Mathematical Models

### 5.1 Open Quantum System (OQS4, $k = 3$)
Deliberation is represented on a 3-dimensional Hilbert space: $|1\rangle$ (Option 1), $|2\rangle$ (Hesitation), $|3\rangle$ (Option 2). State density matrix $\rho(t)$ evolves via the Lindblad equation:
$$\frac{d\rho}{dt} = -i[H, \rho] + \sum_{k=1}^2 \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$
- **Hamiltonian:** Coherent cognitive tunneling between states:
  $$H = \begin{pmatrix} 0 & d & 0 \\ d & 0 & d \\ 0 & d & 0 \end{pmatrix}$$
- **Jump Operators:** Non-reversible dissipation from hesitation into choices:
  $$L_1 = |1\rangle\langle 2|, \quad L_2 = |3\rangle\langle 2|$$
- **Parameters:** Free parameters $\theta = (d, \gamma_1, \gamma_2)$ with $k = 3$.

### 5.2 Classical Markov Chain (CTMC, $k = 4$)
A continuous-time master equation on identical 3-state probability vector $P(t) = [p_1(t), p_2(t), p_3(t)]$:
$$\frac{dP}{dt} = P \cdot Q, \quad Q = \begin{pmatrix} -q_{12} & q_{12} & 0 \\ q_{21} & -(q_{21} + q_{23}) & q_{23} \\ 0 & q_{32} & -q_{32} \end{pmatrix}$$
- **Parameters:** Free transition rates $\theta = (q_{12}, q_{21}, q_{23}, q_{32})$ with $k = 4$.

### 5.3 Model Selection Metrics
For $N = 30$ observations per trial (15 bins $\times$ 2 options):
$$\text{SSE} = \sum_{t=1}^{15} \left( [p_1(t) - \hat{p}_1(t)]^2 + [p_2(t) - \hat{p}_2(t)]^2 \right)$$
$$\text{BIC} = N \ln\left(\frac{\text{SSE}}{N}\right) + k \ln(N), \quad \Delta\text{BIC} = \text{BIC}_{\text{CTMC}} - \text{BIC}_{\text{OQS4}}$$
- Analytical 1-parameter complexity penalty: $\Delta\text{BIC}_{\text{penalty}} = \ln(30) \approx 3.401$.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux

### 1. Clone the Repository
```bash
git clone https://github.com/trtrtin-lgtm/Dataset-decision.git
cd Dataset-decision
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Reproducing Manuscript Results & Figures

### 1. Execute Full In-House Model Evaluation
```bash
python GUI_App/modules/optimization.py --dataset in_house --output results/OQS_Results_Auto.xlsx
```

### 2. Run Cross-Paradigm Validation on Krajbich Benchmark (3,651 Trials)
```bash
python GUI_App/krajbich_validation_pipeline.py --input data/krajbich/data_nature2010.dta --output results/Krajbich_2010_Validation_Results.xlsx
```

### 3. Run Parameter Recovery & Empirical Bayes Shrinkage
```bash
python GUI_App/bayesian_informative_priors.py --trials 100 --noise 0.04 0.316 --output results/Bayesian_Informative_Priors_Results.xlsx
```

### 4. Verify Question Order Effects & QQ Equality
```bash
python GUI_App/qq_equality_analysis.py --input data/survey/QOE_Survey_Data.xlsx
```

### 5. Re-generate High-Resolution Manuscript Figures (Figures 1 to 7)
```bash
python GUI_App/visualization.py --export-all --dpi 300 --outdir results/
```

---

## 8. Key Findings & Empirical Benchmarks

### Model Comparison Summary (Table 1 & Table 2)

| Dataset | Metric / Model | OQS4 ($k=3$) | CTMC ($k=4$) | OQS6 ($k=5$) | Statistical Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-House Dilemmas ($N=82$)** | **BIC Win Count (%)** | **68 / 82 (82.9%)** | 14 / 82 (17.1%) | 0 / 82 (0.0%) | Binomial $p < 10^{-8}$ |
| | **AIC Win Count (%)** | **62 / 82 (75.6%)** | 19 / 82 (23.2%) | 1 / 82 (1.2%) | Binomial $p < 10^{-5}$ |
| | **Median SSE (Unpenalized)** | 0.985 | 0.933 | 0.885 | Wilcoxon $p = 0.384$ |
| | **Median $\Delta\text{BIC}$** | **+2.57** [1.26, 3.56] | Baseline | --- | Positive evidence band |
| **Krajbich Benchmark ($N=3,651$)** | **BIC Win Count (%)** | **2,934 / 3,651 (80.4%)** | 717 / 3,651 (19.6%) | --- | Binomial $p < 10^{-100}$ |
| | **Subject Unanimity** | **39 / 39 (100.0%)** | 0 / 39 (0.0%) | --- | Wilcoxon $W = 780, p = 1.82 \times 10^{-12}$ |
| | **Mean SSE Ratio** | **1.002** | Baseline (1.000) | --- | Descriptively equivalent |
| | **Median $\Delta\text{BIC}$** | **+1.79** [0.50, 2.77] | Baseline | --- | Directionally unanimous |
| **Boundary Parameters** | In-House MLE Boundary Hit Rate | **93.9% (77 / 82)** | 100.0% (82 / 82) | --- | Misspecification marker |
| | Krajbich MLE Boundary Hit Rate | **93.5% (3,413 / 3,651)** | 98.8% (3,607 / 3,651) | --- | Replication $\Delta = 0.4\%$ |
| | **Empirical Bayes MAP ($\lambda = 0.50$)** | **0.0% Boundary Hits** | --- | --- | Regularizes 100% into interior |

---

## 9. Ethics Statement & Governance

Ethics Approval and Consent to Participate: All procedures performed in this study involving human participants were conducted in accordance with the ethical standards of the institutional and/or national research  ommittee and with the 1964 Declaration of Helsinki and its later amendments or comparable ethical standards. The investigation was an observational, non-interventional cognitive psychophysics study posing no more than minimal risk. Written informed consent was obtained from all individual participants prior to inclusion in the study. All data were fully anonymized and de-identified. 

**Conflict of Interest:** The authors declare no competing financial or non-financial interests.  
**Funding:** This study received no external financial funding. Supported by facilities at Ho Chi Minh City University of Technology (HCMUT), VNU-HCM.

---

## 10. Citation

If you find this repository, dataset, or computational code helpful in your research, please cite our manuscript:

```bibtex
@article{tin2026openquantum,
  title={Open Quantum System Models Show a Consistent Parsimony Advantage over Classical Markov Chains in Visual Deliberation Dynamics: A Cross-Paradigm Benchmark},
  author={Tin, Tran Trung and Trang, Le Thuy and Xuan, Pham My and Phu, Do Tuong and Tan, Le Nhat},
  journal={Preprint / Under Review},
  year={2026},
  publisher={Ho Chi Minh City University of Technology (HCMUT), VNU-HCM},
  url={https://github.com/trtrtin-lgtm/Dataset-decision}
}
```

For correspondence, inquiries, or bug reports, please contact:  
**Tran Trung Tin** — `trtrtin@hcmut.edu.vn`  
*Faculty of Applied Science, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM.*
