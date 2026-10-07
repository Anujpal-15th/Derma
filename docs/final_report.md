# Final Report: DermaMNIST Mini Research Prototype

*Date: 2026-10-08 · Environment: Python 3.14.6, scikit-learn 1.9.1, MLflow 3.17.0, FastAPI 0.142.2, Windows 11, CPU only.*

> Research and education prototype. Not a diagnostic device and not clinically validated.

## 1. Summary
We built a reproducible, end-to-end prototype for **imbalanced 7-class skin-lesion classification** on DermaMNIST (HAM10000 at 28×28). It trains five models, tracks every run in MLflow (params, metrics, confusion matrices, models, data hash, library versions), registers the validation-selected model, and serves it through FastAPI with **temperature-calibrated probabilities**.

The best model is an MLP: test **macro-F1 0.427 [95% CI 0.374–0.470]**, ROC-AUC 0.898, and ECE 0.023 after temperature scaling. It is statistically tied with HistGradientBoosting on macro-F1.

The prototype implements the methodology recommended in the DERMAFUSION research package (macro-F1 headline, calibration, CIs, leakage awareness, honest reporting) at laptop scale. It does **not** yet implement metadata fusion or skin-tone fairness, because DermaMNIST has neither metadata nor skin-tone labels.

## 2. Downloaded papers (13; PDFs in `papers/`)
Full summaries are in [`literature_review.md`](literature_review.md). All files were checked to be real PDFs.

| # | Paper | Year / venue | Link | Contribution in one line |
|---|---|---|---|---|
| 1 | MedMNIST v2 (Yang et al.) | 2023, Sci. Data | https://arxiv.org/abs/2110.14795 | Standardised 28×28 biomedical benchmarks, incl. DermaMNIST (7,007/1,003/2,005 split), with ResNet/AutoML baselines |
| 2 | HAM10000 (Tschandl et al.) | 2018, Sci. Data | https://arxiv.org/abs/1803.10417 | 10,015 dermatoscopic images, 7 classes; the source of DermaMNIST |
| 3 | SCIN (Ward et al.) | 2024, arXiv | https://arxiv.org/abs/2402.18545 | Crowdsourced real-world smartphone images of common conditions, with self-reported metadata and skin tone |
| 4 | DDI (Daneshjou et al.) | 2022, Sci. Adv. | https://arxiv.org/abs/2111.08006 | Biopsy-proven, skin-tone-balanced test set; large AUC drops for dark skin |
| 5 | Fitzpatrick17k (Groh et al.) | 2021, CVPR-W | https://arxiv.org/abs/2104.09957 | 16,577 clinical images with Fitzpatrick labels; quantifies tone-related generalisation gaps |
| 6 | DermaCon-IN | 2025, NeurIPS D&B | https://arxiv.org/abs/2506.06099 | 5,450 Indian smartphone images with Fitzpatrick + Monk skin-tone labels |
| 7 | Gessert et al. (ISIC 2019 winner) | 2020, MethodsX | https://arxiv.org/abs/1910.03910 | Multi-resolution EfficientNet ensembles + metadata; 74.2% bal-acc in CV but 63.6% on the official test set |
| 8 | de Lima & Krohling | 2022, arXiv | https://arxiv.org/abs/2205.15442 | CNN vs transformers and MetaBlock fusion on the small PAD-UFES-20 dataset |
| 9 | Guo et al., calibration | 2017, ICML | https://arxiv.org/abs/1706.04599 | Modern nets are over-confident; temperature scaling fixes it cheaply (**used here**) |
| 10 | Focal loss (Lin et al.) | 2017, ICCV | https://arxiv.org/abs/1708.02002 | Loss that down-weights easy examples for class imbalance |
| 11 | EfficientNet (Tan & Le) | 2019, ICML | https://arxiv.org/abs/1905.11946 | Compound-scaled CNN family; standard dermatology backbone |
| 12 | ViT (Dosovitskiy et al.) | 2021, ICLR | https://arxiv.org/abs/2010.11929 | Transformers for images; data-hungry, so marginal at small scale |
| 13 | Grad-CAM (Selvaraju et al.) | 2017, ICCV | https://arxiv.org/abs/1610.02391 | Gradient-based class activation maps for CNN explanations |

**Gaps this prototype addresses** (from the lit review):
1. Accuracy reported as the headline under imbalance. Here macro-F1 is the headline, with a dummy floor and CIs.
2. Calibration is rarely reported. Here we report ECE and apply temperature scaling.
3. Optimistic validation. Here selection is on validation, test results are reported once, and the Gessert CV→test drop is the motivating example.
4. Missing reproducible tracking and serving. Here MLflow logs everything, a registry holds the model, and FastAPI serves it.

**Gaps it does not yet address:** metadata fusion, skin-tone fairness, deep backbones, and explainability.

## 3. Algorithm comparison
Per-phase rationale is in [`algorithm_comparison.md`](algorithm_comparison.md). In short:
- **Features:** pixels plus colour histograms. PCA(100) is fitted inside each pipeline on train only.
- **Imbalance:** class weights (LR, RF, HGB).
- **Models:** a dummy floor, two baselines (LR, RF) and two advanced models (HGB, MLP).
- **Metrics:** macro-F1 (primary), balanced accuracy, macro OvR AUC, ECE, accuracy (context only).
- **Statistics:** 1,000× bootstrap CIs and an exact McNemar test.
- **Tracking:** MLflow with a SQLite backend and the Model Registry.
- **Serving:** FastAPI.

## 4. Prototype results (test set n = 2,005, seed 42; bit-identical on re-run)
| Model | Val macro-F1 | Test macro-F1 [95% CI] | Bal. acc | Acc | ROC-AUC | ECE raw → scaled |
|---|---|---|---|---|---|---|
| **MLP (registered v1)** | **0.475** | **0.427** [0.374, 0.470] | 0.394 | **0.727** | **0.898** | 0.124 → **0.023** |
| HistGB | 0.406 | 0.419 [0.379, 0.457] | 0.431 | 0.665 | 0.871 | 0.042 → 0.036 |
| Logistic regression | 0.343 | 0.363 [0.335, 0.392] | **0.493** | 0.545 | 0.846 | 0.093 → 0.028 |
| Random forest | 0.248 | 0.259 [0.235, 0.283] | 0.241 | 0.710 | 0.897 | 0.035 → 0.037 |
| Dummy (prior) | 0.115 | 0.115 | 0.143 | 0.669 | 0.500 | n/a |

**Findings**
1. **Accuracy hides failure.** The dummy model reaches 66.9% accuracy with macro-F1 0.115. Random forest has 71% accuracy and AUC 0.897 but macro-F1 0.26.
2. **The MLP and HistGB are tied on macro-F1** (the CIs overlap). The MLP is significantly more accurate (McNemar p < 0.0001), mostly on the majority class. HistGB and LR recover more minority-class recall.
3. **Temperature scaling works as Guo et al. predict.** The MLP's ECE falls from 0.124 to 0.023 with a single parameter (T = 1.80). The served probabilities are the scaled ones.
4. **The val→test macro-F1 drop for the selected model** (0.475 → 0.427) is a small instance of the selection optimism the literature warns about.

Artifacts: `results/model_comparison.{csv,png}`, `results/confusion_*.png`, `results/significance.json`, `results/best_model.json`, MLflow runs in `mlflow.db`/`mlruns/`, and the notebook `notebooks/dermafusion_workflow.ipynb`.

## 5. Limitations
- **28×28 resolution.** Most diagnostic texture is lost, so the results are a lower bound for these models.
- **Image-level official split.** HAM10000 has several images per lesion, so duplicates can leak across splits and inflate every score. This was not measured.
- **A single training seed.** The CIs reflect test-set sampling, not training variance.
- **Unequal class balancing.** The MLP has no class-weight option, so the comparison is not perfectly like-for-like.
- **No metadata or skin-tone labels.** Fusion and fairness, the core DERMAFUSION questions, are untested here.
- **Dermatoscopic, not smartphone, images.** The `/predict/image` endpoint squashes arbitrary photos to 28×28, which is far out of distribution for clinical photos.
- **Not a clinical tool**, and not validated on any external population.

## 6. Future improvements (in priority order)
1. Run the lesion-duplicate check (perceptual hash across splits), then train on full-resolution HAM10000 with a lesion-grouped split.
2. Add a CNN baseline (ResNet18/EfficientNet-B0, ImageNet-pretrained, 224 px) with focal loss vs weighted CE. Report the same metric suite.
3. Use 5 seeds and report mean ± std alongside the bootstrap CIs.
4. Port the pipeline to **SCIN (train) → SkinDisNet/DermaCon-IN (external test)** on overlapping classes. Add metadata fusion ablations and skin-tone subgroup gaps, following the DERMAFUSION plan.
5. Add Grad-CAM panels once there is a CNN, a Dockerfile for the API, and CI that runs `tests/test_api.py`.

## 7. Multi-agent process and review log
| Agent | Work done |
|---|---|
| Literature | Downloaded and verified 13 PDFs. It replaced one wrong arXiv ID (2004.13902 turned out to be a physics paper) with de Lima & Krohling 2022. It wrote the review and gaps, marking unverified figures. |
| Data / Model / Tracking / API / Docs | Built `src/`, MLflow logging and registry, FastAPI, the notebook and the docs. |
| Discussion / Review | Loop 1 critique, summarised below. |

**Review loop 1 (strict ML reviewer). It confirmed there is no preprocessing leakage and that selection uses validation only.**

| # | Finding | Action |
|---|---|---|
| 1 | Ragged pixel lists cause a 500 | **Fixed:** returns 422, test added |
| 2 | No upload size or pixel limit (memory DoS) | **Fixed:** 5 MB / 16 MP limits return 413, test added |
| 3 | Exception text leaks to clients | **Fixed:** generic messages; details go to the server log |
| 4 | Docs overclaimed the MLP win and calibration ranking | **Fixed:** CIs and McNemar added, claims softened to "tied" |
| 5 | The MLP is not class-balanced | **Documented;** oversampling listed as future work |
| 6 | Calibration not corrected | **Fixed:** temperature fitted on val, logged, applied in the API |
| 7 | Reproducibility (missing report, no data hash or versions) | **Fixed:** this report; `data_md5` and library-version tags on every run; `requirements-lock.txt` |
| 8 | Unmeasured lesion leakage | **Documented** (limitation, future item 1) |
| 9 | The image endpoint claimed to mirror DermaMNIST preprocessing | **Fixed:** the docstring now says it is lossy |
| 10 | No-op `np.random.seed` | **Removed** |

**Loop 2 (verification).** After the fixes, training was re-run, `python -m tests.test_api` passes (9 checks, including the ragged, wrong-type and oversized cases), and the notebook re-executes with 0 errors. The docs were cross-checked against `results/model_comparison.csv`. The project is coherent, reproducible (identical metrics on re-run) and validated within the stated limitations.
