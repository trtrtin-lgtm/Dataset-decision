# Gaze Dynamics and Quantum-Like Cognitive Modeling

This repository provides open-access data and computational algorithms for modeling visual deliberation, decision conflict, and gaze dynamics using Open Quantum Systems (OQS) and Continuous-Time Markov Chains (CTMC).

## Repository Structure

```
├── data/
│   ├── RawData_from_DB_anonymized.csv         # 60 Hz gaze, pupil, and mouse time series (Codes S01-S20)
│   ├── SheetDecision1.xlsx                    # Question-order Survey (Order 2, N=85, Codes G1_P01-G1_P85)
│   ├── SheetDecision2.xlsx                    # Question-order Survey (Order 1, N=88, Codes G2_P01-G2_P88)
│   ├── trials.csv                             # Trial-level identifiers, choices, and duration
│   └── Krajbich_2010_Validation_Results.xlsx  # Independent public benchmark fits (Krajbich et al. 2010)
│
├── oqs_models.py                              # Lindblad master equation solver (OQS3, OQS5, Hybrid)
├── classical_models.py                        # Continuous-Time Markov Chain generators (CTMC4, 3R, 3E, 2A, 2S)
├── data_processing.py                         # I-VT classification, PFE pupil correction, and binning
├── model_fitting.py                           # L-BFGS-B parameter fitting, pseudo-BIC, and cross-validation
├── multimodal_metrics.py                      # Gaze switches, late pupil dilation, and DCI calculation
├── question_order.py                          # Quantum QQ equality and Fisher order testing
├── run_pipeline.py                            # Minimal end-to-end execution script
└── requirements.txt                           # Python dependencies
```

## Setup and Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Execute the test pipeline:

```bash
python run_pipeline.py
```

## License

This project is released under the MIT License for academic and non-commercial research purposes.
