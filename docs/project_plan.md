# Project Plan: DermaMNIST Imbalance- and Calibration-Aware Baseline Prototype

## Problem statement
Skin-lesion classifiers are usually reported by **accuracy** on imbalanced data, rarely report **calibration**, and are rarely shipped with reproducible tracking or a serving layer. On HAM10000-style data, a model that always predicts "melanocytic nevi" already reaches ~67% accuracy. This prototype asks: *which CPU-friendly classifiers give the best macro-F1 on a small, imbalanced 7-class dermatoscopic dataset, how well calibrated are they, and can the whole loop be made reproducible and servable?*

This is a scaled-down, runnable slice of the DERMAFUSION research design (see `D:\Download\pdf\...markdown.md`). It keeps the methodology (macro-F1 headline, calibration, leakage awareness, honest reporting) and swaps SCIN/deep backbones for a dataset and models that run on a laptop in minutes.

## Objectives
1. Reproducible environment and one-command pipeline.
2. Compare ≥1 trivial, 2 baseline and 2 advanced models on macro-F1, balanced accuracy, ROC-AUC, accuracy and ECE.
3. Track every run in MLflow (params, metrics, artifacts, model, registry version).
4. Serve the registered model through FastAPI with health, JSON and image-upload inference.
5. Document everything in a notebook, README and final report.

## Scope
**In:** DermaMNIST (28×28), classical ML on pixel + colour-histogram features, class weighting, ECE, MLflow, FastAPI.
**Out (documented as next steps):** CNN/ViT backbones, 224 px images, metadata fusion, skin-tone fairness (DermaMNIST has neither metadata nor skin-tone labels), lesion-grouped splits, clinical claims.

## Architecture
```
Zenodo (dermamnist.npz)
   │ src/data.py  download → load official splits → features (pixels + colour hist)
   ▼
src/train.py  5 sklearn pipelines ──► metrics (macro-F1, bal-acc, AUC, ECE) + plots
   │                                   │
   ▼                                   ▼
MLflow (sqlite:///mlflow.db) ◄── params / metrics / artifacts / model; best → Model Registry
   │ models:/dermamnist-classifier/<v>
   ▼
src/api.py (FastAPI)  GET /health · POST /predict (28×28×3 JSON) · POST /predict/image (upload)
```
Configuration lives in `config.yaml`. `MLFLOW_TRACKING_URI` and `MODEL_URI` environment variables override it at serving time.

## Multi-agent team (how the work was split)
| Agent | Responsibility | Output |
|---|---|---|
| Literature review | Find and download 13 open-access papers, summarize them, identify gaps | `papers/`, `docs/literature_review.md` |
| Dataset engineering | Download, splits, EDA, features | `src/data.py`, notebook §2–4 |
| Model development | Baselines + advanced models, metrics | `src/train.py` |
| Experiment tracking | MLflow runs, artifacts, registry | `mlflow.db`, `mlruns/` (artifacts) |
| Backend/API | FastAPI service, validation, errors | `src/api.py`, `tests/test_api.py` |
| Documentation | README, plan, comparison, report | `README.md`, `docs/` |
| Discussion/review | Critique results, recommend fixes, loop until coherent | Review log in `docs/final_report.md` §7 |

## Milestones
| # | Milestone | Done when |
|---|---|---|
| M1 | Lit review | ≥10 PDFs verified, gaps written |
| M2 | Env + data | `pip install -r requirements.txt` works; data downloads; EDA done |
| M3 | Models | 5 models trained, comparison CSV/plots produced |
| M4 | Tracking | Runs visible in MLflow UI; best model registered |
| M5 | API | `/health`, `/predict`, `/predict/image` pass `tests/test_api.py` |
| M6 | Notebook | Executes top-to-bottom with no errors |
| M7 | Review loop | Reviewer findings addressed; final report written |

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Accuracy oversold under imbalance | Macro-F1 is the headline metric; a dummy baseline is shown |
| Test-set selection bias | The best model is picked on validation only |
| Image-level leakage (several images per lesion in HAM10000) | Stated as a limitation; the lesion-grouped split is a next step |
| Python 3.14 wheel availability (torch etc.) | Pure sklearn stack; no GPU or torch dependency |
| MLflow file store deprecated | SQLite backend |
| Unsafe model deserialisation | Only self-trained sklearn types are trusted in skops |
| Misuse as a diagnostic tool | Disclaimer in API responses, docs and notebook |

## Setup requirements and dependencies
Python ≥3.10 (tested on 3.14.6, Windows 11) and about 200 MB of disk. CPU only; full training takes about 1 minute. The packages are listed in `requirements.txt`: numpy, pandas, scikit-learn, matplotlib, mlflow, fastapi, uvicorn, httpx, pyyaml, python-multipart, pillow, jupyter, nbconvert, ipykernel.

## Evaluation criteria
- **Primary:** test macro-F1 for the model selected on validation macro-F1.
- **Secondary:** balanced accuracy, macro one-vs-rest ROC-AUC, ECE (15 bins), accuracy (context only), fit time.
- **Engineering:** the pipeline re-runs to identical metrics with the same seed, the notebook executes cleanly, the API tests pass, and every run appears in MLflow with a registered model version.
