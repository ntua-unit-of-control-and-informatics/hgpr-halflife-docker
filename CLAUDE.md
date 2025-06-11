# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a containerized Heteroscedastic Gaussian Process Regression (HGPR) model API for predicting PFAS compound half-life with uncertainty quantification. The service integrates with the Jaqpot platform and provides REST endpoints for molecular property prediction.

## Core Architecture

**FastAPI Service** (`main.py`): Async web service with `/infer` and `/health` endpoints
**HGPR Implementation** (`src/hgpr.py`): Custom Gaussian Process with heteroscedastic noise modeling and ARD kernel
**Model Service** (`src/model.py`): Molecular descriptor computation using RDKit, topological fingerprints, and Domain of Applicability analysis
**Logging** (`src/loggers/`): Structured JSON logging with request/response middleware

## Key Technical Details

- **Input**: SMILES strings and LogKa values → **Output**: Half-life predictions with epistemic/aleatoric uncertainty and DOA assessment
- **Features**: 6 molecular descriptors (LogKa, PEOE_VSA4, topological bits)
- **Pre-trained model**: `hgpr_model.pkl` (loaded at startup)
- **Training data**: `src/train_data.csv` contains PFAS compounds with half-life statistics

## Development Commands

**Run locally:**
```bash
python -m main --port 8000
```

**Build Docker image:**
```bash
docker build -t jaqpot-hgpr-halflife .
```

**Run container:**
```bash
docker run -p 8000:8000 jaqpot-hgpr-halflife
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Important Implementation Notes

- Model uses Bayesian optimization with L-BFGS-B and proper regularization priors
- Uncertainty decomposition separates model uncertainty from data noise
- Domain of Applicability prevents predictions outside model's reliable range
- All molecular features are automatically computed from SMILES notation
- Authentication token required in `etc/aau_token` for some operations