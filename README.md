# 🌊 AEGIS: Anti-Erosion Grey-box Intelligence System

**Physics-Informed Machine Learning for DEC Pelton Turbine Sediment Erosion Prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-CFD-green.svg)](https://www.openfoam.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Project Overview
AEGIS is an advanced **Anti-Erosion Grey-box Intelligence System** designed to predict sediment erosion in large-capacity Pelton turbines. This project was developed for the *"Precision Anti-Wear Protection Technology for Key Flow Components of Large-Capacity Pelton Turbines"* competition hosted by **Dongfang Electric Machinery Co., Ltd. (DEC)**.

By combining high-fidelity **OpenFOAM** Computational Fluid Dynamics (CFD) simulations with data-driven machine learning models, AEGIS provides a robust, physics-informed solution (Grey-box model) to accurately forecast wear patterns and extend the lifespan of critical turbine components.

## 🌟 Key Features & Innovations

*   **Grey-Box Modeling:** Integrates physical laws (Navier-Stokes, particle tracking) with data-driven ML regression to overcome the limitations of pure black-box models.
*   **OpenFOAM Topology Alignment:** Custom scripts to precisely align Step30 collision data with native OpenFOAM bucket topology for accurate erosion mapping.
*   **Multi-Year Data Synthesis:** Leverages comprehensive structural and operational datasets (`DecPelton2024` and `DecPelton2026`) to capture long-term degradation trends.
*   **Industrial Impact:** Directly addresses a real-world engineering bottleneck in hydropower generation, reducing maintenance costs for mega-watt class turbines.
## System Architecture & Repository Structure

```text
AEGIS/
├── DecPelton2024/          # Historical dataset: 150MW Pelton turbine R&D data
├── DecPelton2026/          # Updated dataset: 2026 turbine geometry & operation data
├── workspace/              # Core step-by-step engineering pipeline
│   ├── step01-step09/      # Geometry, STL, Bezier, baseline validation
│   ├── step10-step19/      # CFD config, OpenFOAM, GPR, particle tracking
│   ├── step20-step29/      # Validation, degradation, CFD/GPR alignment
│   ├── step30/             # Competition core (100+ files, v1-v5, FEM, SI, collision)
│   ├── step31/             # Hydraulic performance, CFD monitor
│   ├── test_step30_*.py    # 50+ unit tests
│   ├── remote/             # DCU / GPU remote sbatch scripts
│   ├── output/             # JSON reports, erosion maps, sensitivity
│   ├── templates/          # Step30 protocol manifest templates
│   ├── schemas/            # JSON schema for Step30 protocol
├── report_build/           # Final report build artifacts
├── DecPelton competition brief.pdf  (original, SHA-identical across 7 branches)
├── STEP30_LATEST_BACKUP_20260829.md
├── MIGRATION_MANIFEST_20260821.txt
├── MIGRATION_README.md
├── PATH_PORTABILITY_POLICY.md
├── CHATGPT_HANDOFF_PROMPT.md
├── .gitignore / .gitattributes
├── README.md
```

**Branch consolidation (2026-11-06):** 6 auxiliary branches (`backup/*`,
`codex/*`, `fix/*`, `simulation`) consolidated into a single `main`.
The complete Step30 engineering pipeline (200+ Python scripts, 5 versions,
test suite, remote execution, report_build) came from the latest
`codex/step30-complete-relative-sync-20260913`. Raw datasets
`DecPelton2024/` and `DecPelton2026/` had identical SHA across all 7
branches (never modified). README.md text preserved from pre-existing `main`.

## 🔬 Methodology (The Grey-Box Approach)

1.  **Physics-Based Simulation (White-Box):** Utilizing OpenFOAM to simulate the two-phase flow (water + sediment particles) and extract impact velocities/angles on the Pelton buckets.
2.  **Data-Driven Surrogate (Black-Box):** Training machine learning models (e.g., Random Forest, Gradient Boosting) on the CFD-extracted impact data to rapidly predict erosion rates without running time-consuming simulations.
3.  **Grey-Box Integration:** Fusing the ML predictions with empirical erosion equations (e.g., Oka erosion model) to ensure physical consistency and generalization capability.

## 🔧 Quick Start

### 1. Environment Setup
```bash
conda create -n aegis python=3.10 -y
conda activate aegis
pip install -r workspace/requirements.txt
```

### 2. Running the Pipeline
```bash
# Extract collision data from OpenFOAM outputs
python workspace/cfd_preprocessing/align_topology.py

# Train the surrogate ML model
python workspace/ml_models/train_erosion_model.py --epochs 100 --batch_size 32
```

## 🏆 Competition & Acknowledgements
*   **Host:** Dongfang Electric Machinery Co., Ltd. (DEC)
*   **Project:** Key Flow Components Precision Anti-Wear Protection Technology Research
*   **Note:** The proprietary raw datasets (`DecPelton2024/2026`) are excluded from this public repository. The code provided demonstrates the data processing and modeling pipeline.

## 📄 License
MIT License
