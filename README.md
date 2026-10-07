# DermaMNIST Mini Research Prototype

An end-to-end, CPU-only research prototype for imbalanced skin-lesion classification on DermaMNIST (HAM10000 at 28×28 px, 7 classes). It covers the literature review, the data, 5 models, MLflow tracking and registry, a FastAPI service, a notebook and a report.

> Research and education only. **Not a diagnostic device.**

## Repository structure
```
config.yaml                   seed, data URL, MLflow URI, experiment/model names
requirements.txt              (requirements-lock.txt = exact tested versions)
src/data.py                   download, official splits, features
src/train.py                  train 5 models, metrics, plots, MLflow logging, registry
src/api.py                    FastAPI: /health, /predict, /predict/image
tests/test_api.py             API smoke test (incl. error cases)
notebooks/build_notebook.py   generates the notebook
notebooks/dermafusion_workflow.ipynb   full executed workflow
papers/                       13 downloaded open-access PDFs
docs/literature_review.md     paper table, summaries, gaps
docs/project_plan.md          problem, objectives, scope, architecture, milestones, risks
docs/algorithm_comparison.md  per-phase algorithm comparison + results
docs/final_report.md          final report
results/                      comparison CSV/PNG, confusion matrices, best_model.json
data/, mlflow.db, mlruns/     generated (data, tracking DB, artifacts)
```

## Installation
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
(On Linux/macOS: `source .venv/bin/activate`.)
```bash
pip install -r requirements.txt
```

## Run the pipeline
This downloads the data (19 MB), trains all models (~1 min), logs to MLflow and registers the best model.
```bash
python -m src.train
```

## MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open http://127.0.0.1:5000. Experiment: `dermamnist-baselines`. Registered model: `dermamnist-classifier`.

## FastAPI service
```bash
uvicorn src.api:app --port 8000
```
The interactive docs are at http://127.0.0.1:8000/docs. The service loads the model recorded in `results/best_model.json`. Override it with the `MODEL_URI` environment variable (e.g. `models:/dermamnist-classifier/1`) and the tracking store with `MLFLOW_TRACKING_URI`.

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | none | `status` (`ok`/`degraded`), `model_loaded`, model info, error |
| `POST /predict` | `{"pixels": 28×28×3 uint8 nested list}` | label, class index, 7 probabilities, disclaimer |
| `POST /predict/image` | multipart `file` (any image; resized to 28×28) | same as above |

Errors: `422` for a wrong shape, ragged lists or out-of-range pixels, `400` for an undecodable image, `413` for uploads over 5 MB or 16 MP, `503` if no model is loaded (run training first).

```bash
curl http://127.0.0.1:8000/health
```
```bash
curl -X POST http://127.0.0.1:8000/predict/image -F "file=@lesion.jpg"
```
Python:
```python
import requests
from src.data import load_splits
img = load_splits()["test"][0][0]
print(requests.post("http://127.0.0.1:8000/predict", json={"pixels": img.tolist()}).json())
```

## Tests and notebook
```bash
python -m tests.test_api
```
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/dermafusion_workflow.ipynb
```
Or open it interactively with `jupyter lab`.

## Results (seed 42, test set; the model is selected on validation macro-F1)
| Model | Macro-F1 [95% CI] | Bal. acc | Acc | ROC-AUC | ECE raw → temp-scaled |
|---|---|---|---|---|---|
| MLP (registered) | **0.427** [0.374, 0.470] | 0.394 | 0.727 | 0.898 | 0.124 → 0.023 |
| HistGradientBoosting | 0.419 [0.379, 0.457] | 0.431 | 0.665 | 0.871 | 0.042 → 0.036 |
| Logistic regression | 0.363 [0.335, 0.392] | **0.493** | 0.545 | 0.846 | 0.093 → 0.028 |
| Random forest | 0.259 [0.235, 0.283] | 0.241 | 0.710 | 0.897 | 0.035 → 0.037 |
| Dummy (prior) | 0.115 | 0.143 | 0.669 | 0.500 | n/a |

The MLP and HistGB are tied on macro-F1 (overlapping CIs). The API serves temperature-scaled probabilities (T fitted on validation).

See `docs/final_report.md` for the discussion, limitations and next steps.

## Next steps
1. Add a CNN (ResNet18/EfficientNet-B0) on 224 px HAM10000 with a lesion-grouped split.
2. Oversample minority classes for the MLP so every model is class-balanced.
3. Use 5 seeds (CIs currently cover test-set sampling only, not training variance), and run a lesion-duplicate check across splits.
4. Move to PAD-UFES-20/SCIN for metadata fusion and skin-tone subgroup evaluation.
5. Containerise the API (Dockerfile) once deployment is needed.
