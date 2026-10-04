"""Alternância entre SVM e atualização do extrator por pares informados pelo SVM."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader

from .data import ImageDataset
from .manifest import read_manifest, validate_splits
from .model import FeatureCNN
from .svm import fit_svm, predict_svm, squared_distances


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 128
    features: int = 16
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    svm_c: float = 1.0
    margin: float = 1.0
    opposite_weight: float = 0.5
    seed: int = 42

    def validate(self) -> None:
        if self.image_size < 16 or self.features < 2 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("image_size >= 16, features >= 2, batch_size >= 1 e epochs >= 1")
        if self.learning_rate <= 0 or self.svm_c <= 0 or self.margin <= 0 or self.opposite_weight < 0:
            raise ValueError("learning_rate, svm_c e margin devem ser positivos; opposite_weight >= 0")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    chunks, labels = [], []
    with torch.no_grad():
        for images, targets, _ in loader:
            chunks.append(model(images.to(device)).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(chunks), np.concatenate(labels)


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classes": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def guidance_pairs(features: np.ndarray, labels: np.ndarray, classifier, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Usa vetores de suporte e erros para escolher vizinhos da mesma classe e da oposta."""
    predicted = predict_svm(classifier, features, features, gamma)
    anchors = np.union1d(classifier.support_, np.flatnonzero(predicted != labels))
    distances = squared_distances(features, features)
    same, opposite, valid = [], [], []
    for anchor in anchors:
        own = np.flatnonzero(labels == labels[anchor])
        own = own[own != anchor]
        other = np.flatnonzero(labels != labels[anchor])
        if len(own) == 0 or len(other) == 0:
            continue
        same.append(own[np.argmin(distances[anchor, own])])
        opposite.append(other[np.argmin(distances[anchor, other])])
        valid.append(anchor)
    return np.asarray(valid, dtype=np.int64), np.asarray(same, dtype=np.int64), np.asarray(opposite, dtype=np.int64)


def update_features(model: nn.Module, dataset: ImageDataset, features: np.ndarray, pairs: tuple,
                    optimizer: torch.optim.Optimizer, config: TrainConfig, device: torch.device) -> float:
    anchors, same, opposite = pairs
    if not len(anchors):
        return 0.0
    model.train()
    losses = []
    order = np.random.permutation(len(anchors))
    for start in range(0, len(order), config.batch_size):
        selected = order[start:start + config.batch_size]
        images = torch.stack([dataset[int(anchors[i])][0] for i in selected]).to(device)
        same_targets = torch.from_numpy(features[same[selected]]).to(device)
        opposite_targets = torch.from_numpy(features[opposite[selected]]).to(device)
        current = model(images)
        pull = (current - same_targets).square().sum(dim=1).mean()
        opposite_distance = torch.linalg.vector_norm(current - opposite_targets, dim=1)
        push = torch.relu(config.margin - opposite_distance).square().mean()
        loss = pull + config.opposite_weight * push
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train(manifest_dir: Path, output_dir: Path, config: TrainConfig) -> dict:
    config.validate()
    train_rows = read_manifest(manifest_dir / "train.txt")
    val_rows = read_manifest(manifest_dir / "val.txt")
    test_rows = read_manifest(manifest_dir / "test.txt")
    validate_splits(train_rows, val_rows, test_rows)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device()
    datasets = [ImageDataset(rows, config.image_size) for rows in (train_rows, val_rows, test_rows)]
    loaders = [DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0) for ds in datasets]
    model = FeatureCNN(config.features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best = None
    history = []

    for epoch in range(config.epochs):
        train_features, train_labels = extract(model, loaders[0], device)
        classifier, gamma = fit_svm(train_features, train_labels, config.svm_c)
        val_features, val_labels = extract(model, loaders[1], device)
        val_pred = predict_svm(classifier, val_features, train_features, gamma)
        val_metrics = report(val_labels, val_pred)
        train_pred = predict_svm(classifier, train_features, train_features, gamma)
        train_metrics = report(train_labels, train_pred)
        record = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics,
                  "support_vectors": int(len(classifier.support_))}
        history.append(record)
        score = val_metrics["balanced_accuracy"]
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch + 1,
                    "state": copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()}),
                    "classifier": classifier, "reference": train_features.copy(), "gamma": gamma}
        print(f"Época {epoch + 1}/{config.epochs}: treino={train_metrics['balanced_accuracy']:.4f} "
              f"validação={score:.4f} suportes={len(classifier.support_)}", flush=True)
        if epoch + 1 < config.epochs:
            pairs = guidance_pairs(train_features, train_labels, classifier, gamma)
            record["guidance_loss"] = update_features(model, datasets[0], train_features, pairs, optimizer, config, device)

    model.load_state_dict(best["state"])
    test_features, test_labels = extract(model, loaders[2], device)
    test_pred = predict_svm(best["classifier"], test_features, best["reference"], best["gamma"])
    result = {"best_epoch": best["epoch"], "device": str(device), "config": asdict(config),
              "counts": {name: len(rows) for name, rows in zip(("train", "val", "test"), (train_rows, val_rows, test_rows))},
              "history": history, "test": report(test_labels, test_pred)}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], output_dir / "cnn.pt")
    joblib.dump(best["classifier"], output_dir / "svm.joblib")
    np.savez_compressed(output_dir / "reference.npz", features=best["reference"], gamma=best["gamma"])
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Teste: acurácia balanceada={result['test']['balanced_accuracy']:.4f}")
    return result
