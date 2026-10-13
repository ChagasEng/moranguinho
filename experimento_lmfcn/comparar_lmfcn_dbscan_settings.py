"""Compara LMFCN + DBSCAN e avalia o segundo melhor setting no teste externo."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from morango_lmfcn.data import ImageDataset
from morango_lmfcn.model import FeatureCNN
from morango_lmfcn.svm import fit_svm, predict_svm
from morango_lmfcn.training import TrainConfig, extract, guidance_pairs, update_features


SEED = 42
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT.parent.parent.parent / "datasetmorangonovo"
OUTPUT_ROOT = DATASET_ROOT / "resultados_lmfcn_settings" / "dbscan"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = torch.device("cpu")


def files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)


def seed_everything() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def split(setting: Path, classes: list[str]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    train, validation = {}, {}
    for class_index, class_name in enumerate(classes):
        paths = files(setting / class_name)
        order = np.random.default_rng(SEED + class_index).permutation(len(paths))
        validation_count = min(len(paths) - 1, max(1, round(len(paths) * 0.2)))
        validation[class_name] = [paths[index] for index in order[:validation_count]]
        train[class_name] = [paths[index] for index in order[validation_count:]]
    return train, validation


def rows(groups: dict[str, list[Path]], classes: list[str]) -> list[tuple[int, Path]]:
    return [(label, path) for label, name in enumerate(classes) for path in groups[name]]


def metricas(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n_imagens": int(len(expected)),
        "acuracia": round(float(accuracy_score(expected, predicted)), 6),
        "f1_macro": round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(expected, predicted, average="macro", zero_division=0)), 6),
        "acuracia_balanceada": round(float(balanced_accuracy_score(expected, predicted)), 6),
    }


def filter_dbscan(groups: dict[str, list[Path]], classes: list[str], config: TrainConfig) -> tuple[dict[str, list[Path]], dict]:
    seed_everything()
    extractor = FeatureCNN(config.features).to(DEVICE)
    filtered, info = {}, {}
    for label, class_name in enumerate(classes):
        dataset = ImageDataset([(label, path) for path in groups[class_name]], config.image_size)
        vectors, _ = extract(extractor, DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0), DEVICE)
        scaled = StandardScaler().fit_transform(vectors)
        components = min(32, len(vectors) - 1, scaled.shape[1])
        reduced = PCA(n_components=components, random_state=SEED).fit_transform(scaled)
        min_samples = max(5, min(20, math.ceil(len(vectors) * 0.02)))
        distances = NearestNeighbors(n_neighbors=min_samples).fit(reduced).kneighbors(reduced)[0][:, -1]
        eps = float(np.quantile(distances, 0.90))
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(reduced)
        keep = labels != -1
        if not keep.any(): keep[:] = True
        filtered[class_name] = [path for path, selected in zip(groups[class_name], keep) if selected]
        info[class_name] = {"imagens_antes": len(groups[class_name]), "imagens_mantidas": len(filtered[class_name]), "removidas": int((~keep).sum()), "min_samples": min_samples, "eps": eps, "componentes_pca": components}
    return filtered, info


def fit(train_rows: list[tuple[int, Path]], validation_rows: list[tuple[int, Path]], config: TrainConfig, output: Path) -> tuple[dict, dict]:
    seed_everything()
    train_data, validation_data = ImageDataset(train_rows, config.image_size), ImageDataset(validation_rows, config.image_size)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = FeatureCNN(config.features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best, history = None, []
    for epoch in range(1, config.epochs + 1):
        train_features, train_labels = extract(model, train_loader, DEVICE)
        classifier, gamma = fit_svm(train_features, train_labels, config.svm_c)
        val_features, val_labels = extract(model, validation_loader, DEVICE)
        val_pred = predict_svm(classifier, val_features, train_features, gamma)
        validation_metrics = metricas(val_labels, val_pred)
        history.append({"epoca": epoch, "validacao": validation_metrics, "vetores_suporte": int(len(classifier.support_))})
        if best is None or float(validation_metrics["f1_macro"]) > best["f1_macro"]:
            best = {"f1_macro": float(validation_metrics["f1_macro"]), "epoca": epoch, "state": copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()}), "classifier": classifier, "reference": train_features.copy(), "gamma": gamma, "metricas": validation_metrics}
        if epoch < config.epochs:
            update_features(model, train_data, train_features, guidance_pairs(train_features, train_labels, classifier, gamma), optimizer, config, DEVICE)
        print(f"validação época {epoch}/{config.epochs}: F1={validation_metrics['f1_macro']:.6f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], output / "cnn.pt"); joblib.dump(best["classifier"], output / "svm.joblib")
    np.savez_compressed(output / "reference.npz", features=best["reference"], gamma=best["gamma"])
    (output / "historico.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return best, {"historico": history}


def evaluate_checkpoint(best: dict, test_rows: list[tuple[int, Path]], config: TrainConfig) -> dict:
    model = FeatureCNN(config.features).to(DEVICE); model.load_state_dict(best["state"])
    features, expected = extract(model, DataLoader(ImageDataset(test_rows, config.image_size), batch_size=config.batch_size, shuffle=False, num_workers=0), DEVICE)
    return metricas(expected, predict_svm(best["classifier"], features, best["reference"], best["gamma"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128); parser.add_argument("--features", type=int, default=16); parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    config = TrainConfig(image_size=args.image_size, features=args.features, batch_size=args.batch, epochs=args.epochs, seed=SEED)
    all_rows = []
    for setting in sorted(path for path in DATASET_ROOT.glob("setting*") if path.is_dir()):
        classes = sorted(path.name for path in setting.iterdir() if path.is_dir()); train, validation = split(setting, classes)
        filtered, dbscan = filter_dbscan(train, classes, config)
        output = OUTPUT_ROOT / setting.name; artifacts = output / "artefatos"; artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "dbscan.json").write_text(json.dumps(dbscan, ensure_ascii=False, indent=2), encoding="utf-8")
        best, _ = fit(rows(filtered, classes), rows(validation, classes), config, output / "treino")
        (artifacts / "configuracao.json").write_text(json.dumps({"metodo": "LMFCN + DBSCAN", "configuracao_lmfcn": config.__dict__, "dbscan": dbscan, "protocolo": "DBSCAN no treino; validação e test não filtrados."}, ensure_ascii=False, indent=2), encoding="utf-8")
        all_rows.append({"setting": setting.name, "classes": classes, "pesos": str(output / "treino" / "cnn.pt"), "removidas_dbscan": sum(item["removidas"] for item in dbscan.values()), **{f"validacao_{key}": value for key, value in best["metricas"].items()}, "melhor_epoca": best["epoca"], "best": best})
    ranked = sorted(all_rows, key=lambda item: (item["validacao_f1_macro"], item["validacao_acuracia_balanceada"], item["validacao_acuracia"]), reverse=True)
    selected = ranked[1]; test = evaluate_checkpoint(selected["best"], rows({name: files(DATASET_ROOT / "test" / name) for name in selected["classes"]}, selected["classes"]), config)
    summary = OUTPUT_ROOT / "comparacao_settings"; summary.mkdir(parents=True, exist_ok=True)
    serializable_rows = [{key: value for key, value in item.items() if key != "best"} for item in all_rows]
    (summary / "metricas_validacao.json").write_text(json.dumps(serializable_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = Path(str(selected["pesos"])); selection = {"ranking_por_validacao": [item["setting"] for item in ranked], "setting_avaliado_no_teste": selected["setting"], "posicao": 2, "criterio": "segundo maior F1 macro de validação", "metricas_teste": test, "pesos": str(checkpoint), "pesos_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (summary / "segundo_melhor_setting.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__": main()
