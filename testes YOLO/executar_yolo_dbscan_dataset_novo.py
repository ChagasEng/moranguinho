r"""Treina e avalia YOLO11-cls + DBSCAN para um setting do datasetmorangonovo.

O DBSCAN é aplicado exclusivamente às imagens de treino após a divisão
estratificada 80/20. Validação e teste externo nunca são filtrados.

Uso:
    .\.venv-yolo\Scripts\python.exe "testes YOLO\executar_yolo_dbscan_dataset_novo.py" --setting setting2
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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO


SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent.parent.parent / "datasetmorangonovo"
BASE_OUTPUT = DATASET_ROOT / "resultados_yolo_settings" / "dbscan"
PRETRAINED_WEIGHTS = SCRIPT_DIR.parent / "yolo11n-cls.pt"


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stratified_split(setting_root: Path, class_names: list[str]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    train, validation = {}, {}
    for class_index, class_name in enumerate(class_names):
        images = image_files(setting_root / class_name)
        if len(images) < 2:
            raise ValueError(f"A classe {class_name} precisa ter ao menos duas imagens.")
        permutation = np.random.default_rng(SEED + class_index).permutation(len(images))
        validation_count = min(len(images) - 1, max(1, round(len(images) * 0.2)))
        validation[class_name] = [images[index] for index in permutation[:validation_count]]
        train[class_name] = [images[index] for index in permutation[validation_count:]]
    return train, validation


def extract_embeddings(model: YOLO, images: list[Path], batch: int) -> np.ndarray:
    batches = []
    for start in range(0, len(images), batch):
        vectors = model.embed(
            [str(path) for path in images[start : start + batch]], imgsz=224, batch=batch, device="cpu", verbose=False
        )
        batches.append(np.stack([vector.cpu().numpy().reshape(-1) for vector in vectors]))
    return np.concatenate(batches)


def dbscan_filter(vectors: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    count = len(vectors)
    if count < 12:
        return np.ones(count, dtype=bool), {"removidas": 0, "min_samples": 0, "eps": 0.0, "componentes_pca": 0}
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
    return keep, {
        "removidas": int((~keep).sum()),
        "min_samples": min_samples,
        "eps": eps,
        "componentes_pca": components,
    }


def stage_dataset(destination: Path, train: dict[str, list[Path]], validation: dict[str, list[Path]]) -> None:
    for split_name, groups in (("train", train), ("val", validation)):
        for class_name, images in groups.items():
            for index, image in enumerate(images):
                link_or_copy(image, destination / split_name / class_name / f"{index:05d}_{image.name}")


def evaluate(weights: Path, data_root: Path, class_names: list[str], batch: int) -> dict[str, float | int]:
    expected, sources = [], []
    for label, class_name in enumerate(class_names):
        images = image_files(data_root / class_name)
        expected.extend([label] * len(images))
        sources.extend(str(image) for image in images)
    model = YOLO(str(weights))
    predicted = [
        int(result.probs.top1)
        for result in model.predict(source=sources, imgsz=224, batch=batch, device="cpu", verbose=False)
    ]
    return {
        "n_imagens": len(expected),
        "acuracia": round(float(accuracy_score(expected, predicted)), 6),
        "f1_macro": round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(expected, predicted, average="macro", zero_division=0)), 6),
        "acuracia_balanceada": round(float(balanced_accuracy_score(expected, predicted)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="setting2", choices=[f"setting{number}" for number in range(1, 6)])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("--epochs e --batch devem ser positivos.")

    setting_root = DATASET_ROOT / args.setting
    if not setting_root.is_dir() or not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError("Setting ou pesos pré-treinados não encontrados.")
    class_names = sorted(path.name for path in setting_root.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise ValueError("São necessárias ao menos duas classes.")

    train, validation = stratified_split(setting_root, class_names)
    extractor = YOLO(str(PRETRAINED_WEIGHTS))
    filtered, dbscan_details = {}, {}
    for class_name in class_names:
        mask, details = dbscan_filter(extract_embeddings(extractor, train[class_name], args.batch))
        filtered[class_name] = [image for image, selected in zip(train[class_name], mask) if selected]
        dbscan_details[class_name] = {
            "imagens_antes": len(train[class_name]),
            "imagens_mantidas": len(filtered[class_name]),
            **details,
        }

    output = BASE_OUTPUT / args.setting
    artifacts = output / "artefatos"
    metrics_dir = output / "metricas"
    internal_dataset = output / "dataset_interno"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (artifacts / "dbscan.json").write_text(json.dumps(dbscan_details, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_dataset(internal_dataset, filtered, validation)

    model = YOLO(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(internal_dataset), epochs=args.epochs, imgsz=224, batch=args.batch,
        device="cpu", workers=0, seed=SEED, deterministic=True,
        project=str(output), name="treino", exist_ok=True, verbose=False,
    )
    weights = output / "treino" / "weights" / "best.pt"
    checkpoint_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    validation_metrics = evaluate(weights, internal_dataset / "val", class_names, args.batch)
    test_metrics = evaluate(weights, DATASET_ROOT / "test", class_names, args.batch)
    metadata = {
        "metodo": "YOLO11n-cls + DBSCAN",
        "setting": args.setting,
        "classes": class_names,
        "semente": SEED,
        "epocas": args.epochs,
        "batch": args.batch,
        "imgsz": 224,
        "dispositivo": "cpu",
        "protocolo": "DBSCAN aplicado somente ao treino após divisão estratificada 80/20; validação e teste não foram filtrados.",
        "pesos_iniciais": str(PRETRAINED_WEIGHTS),
        "pesos_avaliados": str(weights),
        "pesos_avaliados_sha256": checkpoint_sha256,
        "dbscan": dbscan_details,
    }
    (artifacts / "configuracao.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "validacao.json").write_text(json.dumps(validation_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "teste.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validacao": validation_metrics, "teste": test_metrics, "saida": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
