"""Compara MobileNetV2 + DBSCAN e avalia o setting2 no test.

O DBSCAN é aplicado apenas ao treino depois da divisão 80/20 estratificada. A
avaliação externa oficial usa o mesmo setting2 do baseline, para isolar o efeito
do DBSCAN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from comparar_mobilenetv2_settings import (
    DATASET_ROOT, DEFAULT_BATCH, DEFAULT_EPOCHS, DEVICE, EVAL_TRANSFORM, IMAGE_EXTENSIONS,
    OUTPUT_DIR, TRAIN_TRANSFORM, create_split, evaluate, image_files, link_or_copy, loader,
    new_model, seed_everything,
)


SEED = 42
BASE_OUTPUT = OUTPUT_DIR / "dbscan"


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return EVAL_TRANSFORM(image.convert("RGB"))


def split_paths(setting: Path, class_names: list[str]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    train, validation = {}, {}
    for class_index, class_name in enumerate(class_names):
        images = image_files(setting / class_name)
        order = np.random.default_rng(SEED + class_index).permutation(len(images))
        validation_count = min(len(images) - 1, max(1, round(len(images) * 0.2)))
        validation[class_name] = [images[index] for index in order[:validation_count]]
        train[class_name] = [images[index] for index in order[validation_count:]]
    return train, validation


@torch.inference_mode()
def embeddings(paths: list[Path], batch: int) -> np.ndarray:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features.to(DEVICE).eval()
    pool = nn.AdaptiveAvgPool2d((1, 1))
    values = []
    for images in DataLoader(ImagePathDataset(paths), batch_size=batch, shuffle=False, num_workers=0):
        values.append(pool(model(images.to(DEVICE))).flatten(1).cpu().numpy())
    return np.concatenate(values)


def dbscan_filter(vectors: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    count = len(vectors)
    scaled = StandardScaler().fit_transform(vectors)
    components = min(32, count - 1, scaled.shape[1])
    reduced = PCA(n_components=components, random_state=SEED).fit_transform(scaled)
    min_samples = max(5, min(20, math.ceil(count * 0.02)))
    distances = NearestNeighbors(n_neighbors=min_samples).fit(reduced).kneighbors(reduced)[0][:, -1]
    eps = float(np.quantile(distances, 0.90))
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(reduced)
    keep = labels != -1
    if not keep.any():
        keep[:] = True
    return keep, {"removidas": int((~keep).sum()), "min_samples": min_samples, "eps": eps, "componentes_pca": components}


def stage(destination: Path, train: dict[str, list[Path]], validation: dict[str, list[Path]]) -> None:
    for split_name, groups in (("train", train), ("val", validation)):
        for class_name, paths in groups.items():
            for index, path in enumerate(paths):
                link_or_copy(path, destination / split_name / class_name / f"{index:05d}_{path.name}")


def train_one(setting: Path, epochs: int, batch: int) -> dict[str, object]:
    seed_everything()
    class_names = sorted(path.name for path in setting.iterdir() if path.is_dir())
    train, validation = split_paths(setting, class_names)
    filtered, details = {}, {}
    for class_name in class_names:
        mask, result = dbscan_filter(embeddings(train[class_name], batch))
        filtered[class_name] = [path for path, keep in zip(train[class_name], mask) if keep]
        details[class_name] = {"imagens_antes": len(train[class_name]), "imagens_mantidas": len(filtered[class_name]), **result}

    output = BASE_OUTPUT / setting.name
    artifacts, internal, train_dir = output / "artefatos", output / "dataset_interno", output / "treino"
    artifacts.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    (artifacts / "dbscan.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    stage(internal, filtered, validation)
    train_loader = loader(internal / "train", TRAIN_TRANSFORM, batch, shuffle=True)
    validation_loader = loader(internal / "val", EVAL_TRANSFORM, batch, shuffle=False)
    model = new_model(len(class_names))
    optimizer, criterion = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4), nn.CrossEntropyLoss()
    weights, best_f1, history = train_dir / "best.pt", -1.0, []
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(); loss = criterion(model(images.to(DEVICE)), labels.to(DEVICE)); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics = evaluate(model, validation_loader)
        history.append({"epoca": epoch, "loss_treino": round(total_loss / len(train_loader.dataset), 6), **metrics})
        if float(metrics["f1_macro"]) > best_f1:
            best_f1 = float(metrics["f1_macro"]); torch.save(model.state_dict(), weights)
        print(f"{setting.name} | época {epoch}/{epochs} | F1 validação={metrics['f1_macro']:.6f}", flush=True)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    validation_metrics = evaluate(model, validation_loader)
    (train_dir / "historico.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts / "configuracao.json").write_text(json.dumps({"metodo": "MobileNetV2 + DBSCAN", "semente": SEED, "epocas": epochs, "batch": batch, "dispositivo": str(DEVICE), "dbscan": details, "protocolo": "DBSCAN somente no treino; validação e teste não filtrados."}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"setting": setting.name, "classes": class_names, "pesos": str(weights), "removidas_dbscan": sum(int(item["removidas"]) for item in details.values()), **{f"validacao_{key}": value for key, value in validation_metrics.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--avaliar-setting", metavar="SETTING",
        help="Avalia no test o checkpoint DBSCAN já treinado desse setting, sem retreinar.",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    if args.avaliar_setting:
        setting = args.avaliar_setting
        weights = BASE_OUTPUT / setting / "treino" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Não há checkpoint DBSCAN pronto para {setting}: {weights}")
        class_names = sorted(path.name for path in (DATASET_ROOT / setting).iterdir() if path.is_dir())
        model = new_model(len(class_names))
        model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
        test_metrics = evaluate(model, loader(DATASET_ROOT / "test", EVAL_TRANSFORM, args.batch, shuffle=False))
        payload = {
            "setting": setting,
            "protocolo": "Avaliação controlada: checkpoint MobileNetV2 + DBSCAN do mesmo setting no test externo, sem retreino.",
            "pesos": str(weights),
            "pesos_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "metricas_teste": test_metrics,
        }
        destination = BASE_OUTPUT / setting / "metricas_teste_setting2_controlado.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return
    rows = [train_one(setting, args.epochs, args.batch) for setting in sorted(DATASET_ROOT.glob("setting*")) if setting.is_dir()]
    ranked = sorted(rows, key=lambda row: (float(row["validacao_f1_macro"]), float(row["validacao_acuracia_balanceada"]), float(row["validacao_acuracia"])), reverse=True)
    selected = next((row for row in rows if row["setting"] == "setting2"), None)
    if selected is None:
        raise FileNotFoundError("O setting2 é obrigatório para a avaliação comparativa.")
    model = new_model(len(selected["classes"]))
    weights = Path(str(selected["pesos"])); model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    test_metrics = evaluate(model, loader(DATASET_ROOT / "test", EVAL_TRANSFORM, args.batch, shuffle=False))
    summary = BASE_OUTPUT / "comparacao_settings"; summary.mkdir(parents=True, exist_ok=True)
    (summary / "metricas_validacao.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {"ranking_por_validacao": [row["setting"] for row in ranked], "setting_avaliado_no_teste": selected["setting"], "criterio": "comparação controlada com o baseline: mesmo setting2, com e sem DBSCAN", "metricas_validacao": {key: value for key, value in selected.items() if key.startswith("validacao_")}, "metricas_teste": test_metrics, "pesos": str(weights), "pesos_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (summary / "setting2_controlado.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE_OUTPUT / selected["setting"] / "metricas_teste_setting2_controlado.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# MobileNetV2 + DBSCAN", "", "DBSCAN aplicado somente ao treino. O mesmo `setting2` do baseline foi avaliado no test.", "", "| Setting | Removidas DBSCAN | F1 macro validação |", "|---|---:|---:|"]
    report += ["| {setting} | {removidas_dbscan} | {validacao_f1_macro:.6f} |".format(**row) for row in ranked]
    report += ["", f"Avaliação controlada no teste: **{selected['setting']}**.", "", "| Acurácia | F1 macro | Recall macro |", "|---:|---:|---:|", "| {acuracia:.4f} | {f1_macro:.4f} | {recall_macro:.4f} |".format(**test_metrics), ""]
    (summary / "RELATORIO.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
