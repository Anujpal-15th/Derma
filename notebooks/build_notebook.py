"""Generates notebooks/dermafusion_workflow.ipynb. Then execute:
jupyter nbconvert --to notebook --execute --inplace notebooks/dermafusion_workflow.ipynb"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# DermaMNIST Mini Research Prototype: Full Workflow
**Question:** On a small, heavily imbalanced skin-lesion dataset, which CPU-friendly classifiers give the best *macro-F1* (not accuracy), and how well calibrated are they?

Workflow: environment → data → EDA → preprocessing → training (5 models) → evaluation & calibration → MLflow → FastAPI → conclusions.

> Research/education prototype. **Not a diagnostic device.**"""),
md("""## 1. Environment setup
Create the env once from the repo root:
```bash
python -m venv .venv
.venv\\Scripts\\activate        # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```"""),
code("""import os, sys, warnings
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.getcwd())
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn, mlflow
from src.data import load_config, load_splits, to_features, CLASSES
cfg = load_config()
print('python', sys.version.split()[0], '| sklearn', sklearn.__version__, '| mlflow', mlflow.__version__)
cfg"""),
md("## 2. Data loading\nDermaMNIST (MedMNIST v2) is HAM10000 dermatoscopic images resized to 28×28 RGB, 7 classes, with the official train/val/test split. It is downloaded from Zenodo on first use."),
code("""splits = load_splits(cfg)
for s, (Xs, ys) in splits.items():
    print(f'{s:5s} images={Xs.shape} labels={ys.shape} dtype={Xs.dtype}')"""),
md("## 3. Exploratory data analysis"),
code("""counts = pd.DataFrame({s: pd.Series(ys).value_counts().sort_index() for s, (_, ys) in splits.items()})
counts.index = CLASSES
counts['train_%'] = (100 * counts['train'] / counts['train'].sum()).round(1)
counts"""),
code("""fig, ax = plt.subplots(figsize=(8, 3.5))
counts['train'].sort_values().plot.barh(ax=ax, color='#4C72B0')
ax.set_title('Training class distribution (heavy imbalance: nevi dominate)'); ax.set_xlabel('images')
plt.tight_layout(); plt.show()"""),
code("""Xtr, ytr = splits['train']
fig, axes = plt.subplots(7, 8, figsize=(9, 8))
for c in range(7):
    for j, i in enumerate(np.where(ytr == c)[0][:8]):
        axes[c, j].imshow(Xtr[i])
    for a in axes[c]:
        a.axis('off')
    axes[c, 0].set_title(CLASSES[c][:22], fontsize=8, loc='left')
plt.tight_layout(); plt.show()"""),
code("""# Per-class mean colour: a cheap signal the colour-histogram features exploit
means = pd.DataFrame([Xtr[ytr == c].reshape(-1, 3).mean(0) for c in range(7)], index=CLASSES, columns=['R', 'G', 'B']).round(1)
print('pixel range:', Xtr.min(), Xtr.max(), '| no missing values in uint8 arrays')
means"""),
md("## 4. Preprocessing\nFeatures are 2,352 scaled pixels plus 48 colour-histogram bins. Model pipelines then add StandardScaler + PCA(100) where useful. Imbalance is handled with `class_weight='balanced'`. The splits are the official ones, and model selection uses **validation** only."),
code("""X = {s: to_features(x) for s, (x, _) in splits.items()}
y = {s: lab for s, (_, lab) in splits.items()}
{s: v.shape for s, v in X.items()}"""),
md("## 5. Model training + MLflow tracking\n`src/train.py` trains 5 models (dummy prior, logistic regression, random forest, HistGradientBoosting, MLP). For each it logs params, val/test metrics, the confusion matrix and the model to MLflow. It then registers the best model, chosen on validation macro-F1, in the MLflow Model Registry."),
code("""from src import train
train.main()"""),
md("## 6. Evaluation and comparison"),
code("""res = pd.read_csv('results/model_comparison.csv')
print(open('results/significance.json').read())
res[['model', 'val_macro_f1', 'test_macro_f1', 'test_macro_f1_ci_low', 'test_macro_f1_ci_high', 'test_balanced_acc', 'test_accuracy', 'test_roc_auc_ovr', 'test_ece', 'test_ece_temp_scaled', 'temperature']]"""),
code("""from IPython.display import Image as Img, display
display(Img('results/model_comparison.png'))
display(Img(f'results/confusion_{res.iloc[0].model}.png'))"""),
md("### Calibration: reliability diagram for the registered model"),
code("""import json, mlflow.sklearn
mlflow.set_tracking_uri(cfg['mlflow_uri'])
info = json.load(open('results/best_model.json'))
model = mlflow.sklearn.load_model(info['uri'])
proba = model.predict_proba(X['test']); conf, pred = proba.max(1), proba.argmax(1)
edges = np.linspace(0, 1, 11); mids, accs = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.sum() > 10:
        mids.append(conf[m].mean()); accs.append((pred[m] == y['test'][m]).mean())
plt.figure(figsize=(4.5, 4.5)); plt.plot([0, 1], [0, 1], 'k--', label='perfect')
plt.plot(mids, accs, 'o-', label=info['model']); plt.xlabel('confidence'); plt.ylabel('accuracy')
plt.title(f"Reliability (test ECE raw={train.ece(y['test'], proba):.3f}, temp-scaled={train.ece(y['test'], train.apply_temperature(proba, info['temperature'])):.3f})", fontsize=9); plt.legend(); plt.show()"""),
code("""from sklearn.metrics import classification_report
print(classification_report(y['test'], pred, target_names=CLASSES, digits=3))"""),
md("## 7. Inspect MLflow runs and the model registry\nUI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`, then open http://127.0.0.1:5000"),
code("""runs = mlflow.search_runs(experiment_names=[cfg['experiment']])
runs[['tags.mlflow.runName', 'metrics.val_macro_f1', 'metrics.test_macro_f1', 'metrics.test_ece']].sort_values('metrics.val_macro_f1', ascending=False).head(5)"""),
code("""client = mlflow.MlflowClient()
[(v.name, v.version, v.run_id) for v in client.search_model_versions(f"name='{cfg['registered_model']}'")]"""),
md("## 8. FastAPI usage\nStart the server with `uvicorn src.api:app --port 8000`. Swagger docs are at http://127.0.0.1:8000/docs. Here we call the same app in-process with `TestClient`. Against a live server, use `requests.post('http://127.0.0.1:8000/predict', json=...)`."),
code("""from fastapi.testclient import TestClient
from src.api import app
with TestClient(app) as client:
    print(client.get('/health').json())
    r = client.post('/predict', json={'pixels': splits['test'][0][0].tolist()})
    print(r.status_code, r.json()['label'], '| true:', CLASSES[y['test'][0]])
    bad = client.post('/predict', json={'pixels': [[[1, 2, 3]]]})
    print('bad input ->', bad.status_code, bad.json()['detail'])"""),
md("""## 9. Conclusions
* **Accuracy is misleading here.** The dummy prior classifier reaches ~67% accuracy with macro-F1 ≈ 0.11. All comparisons therefore lead with macro-F1 and balanced accuracy.
* **Best model** is picked on validation macro-F1 (the MLP). Its test macro-F1 is 0.427 [95% CI 0.374–0.470], statistically tied with HistGB (0.419 [0.379–0.457]). The MLP is significantly more *accurate* (McNemar p<0.0001), but HistGB has better balanced accuracy. Minority classes (dermatofibroma, vascular, actinic keratoses) stay weak for every model.
* **Calibration:** the MLP is over-confident (raw ECE 0.124). A temperature fitted on validation (T≈1.8) cuts test ECE to 0.023, and the API serves these scaled probabilities.
* **Trade-offs:** logistic regression has the best balanced accuracy (class weighting), and random forest has a high AUC but collapses on minority classes, so neither accuracy nor AUC alone is a safe headline.
* **Limitations:** 28×28 resolution, no lesion-level split (HAM10000 has several images per lesion, so leakage is possible), a single seed, no metadata or skin-tone labels, and no CNN.
* **Next:** a CNN/transfer-learning backbone at 224 px, a lesion-grouped split, multi-seed runs, and metadata fusion on PAD-UFES-20/SCIN (see `docs/final_report.md`)."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/dermafusion_workflow.ipynb")
print("written notebooks/dermafusion_workflow.ipynb")
