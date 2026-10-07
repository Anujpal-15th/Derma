"""Fine-tune an ImageNet-pretrained ResNet-18 on DermaMNIST-128 images, trained at SIZE px on CPU. Run: python -m src.cnn [epochs]"""
import json
import sys
import time
import urllib.request

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet18_Weights, resnet18

from src.data import CLASSES, ROOT, load_config
from src.train import apply_temperature, bootstrap_ci, ece, fit_temperature, metrics

SIZE = 96  # ponytail: CPU budget (~6 min/epoch); use 128+ on a GPU
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_128(cfg):
    path = ROOT / "data" / "dermamnist_128.npz"
    if not path.exists():
        urllib.request.urlretrieve(cfg["data_url"].replace("dermamnist.npz", "dermamnist_128.npz"), path)
    d = np.load(path)
    return {s: (d[f"{s}_images"], d[f"{s}_labels"].ravel()) for s in ("train", "val", "test")}


def to_tensor(images):
    """uint8 (N,H,W,3) of any size -> normalised float tensor (N,3,128,128)."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != SIZE:
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


@torch.no_grad()
def predict_proba(model, images, bs=256):
    model.eval()
    return np.concatenate([F.softmax(model(to_tensor(images[i:i + bs])), 1).numpy()
                           for i in range(0, len(images), bs)])


def augment(x):
    """Random flips + 90-degree rotations (lesions have no canonical orientation)."""
    if np.random.rand() < 0.5:
        x = x.flip(3)
    if np.random.rand() < 0.5:
        x = x.flip(2)
    return torch.rot90(x, np.random.randint(4), (2, 3))


def main(epochs=8):
    cfg = load_config()
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.set_num_threads(torch.get_num_threads())
    data = load_128(cfg)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    # sqrt-inverse-frequency class weights: helps minority recall without wrecking majority precision
    counts = np.bincount(ytr, minlength=len(CLASSES))
    w = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xtr) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)

    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment(cfg["experiment"])
    best_f1, best_state = -1, None
    with mlflow.start_run(run_name="resnet18_96px") as run:
        mlflow.log_params({"model": "resnet18_96px", "pretrained": "IMAGENET1K_V1", "epochs": epochs,
                           "batch_size": 64, "lr": 1e-3, "loss": "CE + sqrt class weights",
                           "augment": "flips+rot90", "img_size": SIZE, "seed": cfg["seed"]})
        for ep in range(epochs):
            model.train()
            t0, perm, tot = time.time(), np.random.permutation(len(Xtr)), 0.0
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                x, y = augment(to_tensor(Xtr[idx])), torch.from_numpy(ytr[idx]).long()
                loss = F.cross_entropy(model(x), y, weight=w, label_smoothing=0.05)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * len(idx)
            m = metrics(yva, predict_proba(model, Xva))
            mlflow.log_metrics({"train_loss": tot / len(Xtr), **{f"val_{k}": v for k, v in m.items()}}, step=ep)
            print(f"epoch {ep + 1}/{epochs} loss={tot / len(Xtr):.3f} val_macroF1={m['macro_f1']:.3f} "
                  f"val_acc={m['accuracy']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if m["macro_f1"] > best_f1:  # select on validation only
                best_f1, best_state = m["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        t = fit_temperature(yva, predict_proba(model, Xva))
        p_test = predict_proba(model, Xte)
        m = metrics(yte, p_test)
        lo, hi = bootstrap_ci(yte, p_test.argmax(1))
        final = {**{f"test_{k}": v for k, v in m.items()}, "best_val_macro_f1": best_f1, "temperature": t,
                 "test_ece_temp_scaled": ece(yte, apply_temperature(p_test, t)),
                 "test_macro_f1_ci_low": lo, "test_macro_f1_ci_high": hi}
        mlflow.log_metrics(final)
        mlflow.pytorch.log_model(model, name="model")
        print(json.dumps({k: round(v, 4) for k, v in final.items()}, indent=1))
        print(confusion_matrix(yte, p_test.argmax(1)))
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", cfg["registered_model"])

    out = ROOT / cfg["out_dir"]
    np.save(out / "cnn_test_proba.npy", p_test)
    (out / "cnn_metrics.json").write_text(json.dumps({k: round(float(v), 4) for k, v in final.items()}, indent=2))
    (out / "best_model.json").write_text(json.dumps(
        {"model": "resnet18_96px", "flavor": "pytorch", "run_id": run.info.run_id, "version": mv.version,
         "temperature": t, "uri": f"models:/{cfg['registered_model']}/{mv.version}"}, indent=2))
    print(f"registered v{mv.version}")
    update_comparison(cfg)


def update_comparison(cfg=None):
    """Add/replace the CNN row in results/model_comparison.csv (sorted by val macro-F1)."""
    import pandas as pd
    cfg = cfg or load_config()
    out = ROOT / cfg["out_dir"]
    info, m = json.loads((out / "best_model.json").read_text()), json.loads((out / "cnn_metrics.json").read_text())
    df = pd.read_csv(out / "model_comparison.csv")
    row = {"model": info["model"], "run_id": info["run_id"], "val_macro_f1": m["best_val_macro_f1"],
           **{k: v for k, v in m.items() if k in df.columns}}
    df = pd.concat([df[~df.model.str.startswith("resnet")], pd.DataFrame([row])])
    df.sort_values("val_macro_f1", ascending=False).to_csv(out / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
