"""Compara os cinco settings de datasetmorangonovo com MobileNetV2.

O conjunto test não participa da escolha: cada setting é dividido em 80% treino
e 20% validação estratificada. Apenas o melhor F1 macro de validação é avaliado
no teste externo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent.parent.parent / "datasetmorangonovo"
OUTPUT_DIR = DATASET_ROOT / "resultados_mobilenetv2_settings"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


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


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_split(setting: Path, destination: Path) -> tuple[list[str], dict[str, int]]:
    class_names = sorted(path.name for path in setting.iterdir() if path.is_dir())
    counts: dict[str, int] = {}
    for class_index, class_name in enumerate(class_names):
        images = image_files(setting / class_name)
        if len(images) < 2:
            raise ValueError(f"A classe {class_name} tem menos de duas imagens.")
        order = np.random.default_rng(SEED + class_index).permutation(len(images))
        validation_count = min(len(images) - 1, max(1, round(len(images) * 0.2)))
        partitions = {"val": order[:validation_count], "train": order[validation_count:]}
        counts[f"{class_name}_treino"] = len(partitions["train"])
        counts[f"{class_name}_validacao"] = len(partitions["val"])
        for split_name, indexes in partitions.items():
            for output_index, index in enumerate(indexes):
                image = images[index]
                link_or_copy(image, destination / split_name / class_name / f"{output_index:05d}_{image.name}")
    return class_names, counts


def loader(root: Path, transform: transforms.Compose, batch: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(
        datasets.ImageFolder(root, transform=transform), batch_size=batch, shuffle=shuffle,
        num_workers=0, generator=generator,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, data: DataLoader) -> dict[str, float | int]:
    model.eval()
    expected, predicted = [], []
    for images, labels in data:
        logits = model(images.to(DEVICE))
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        expected.extend(labels.tolist())
    return {
        "n_imagens": len(expected),
        "acuracia": round(float(accuracy_score(expected, predicted)), 6),
        "f1_macro": round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(expected, predicted, average="macro", zero_division=0)), 6),
        "acuracia_balanceada": round(float(balanced_accuracy_score(expected, predicted)), 6),
    }


def new_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    return model.to(DEVICE)


def train_one_setting(setting: Path, epochs: int, batch: int) -> dict[str, object]:
    seed_everything()
    internal_root = OUTPUT_DIR / "datasets_internos" / setting.name
    class_names, split_counts = create_split(setting, internal_root)
    train_loader = loader(internal_root / "train", TRAIN_TRANSFORM, batch, shuffle=True)
    validation_loader = loader(internal_root / "val", EVAL_TRANSFORM, batch, shuffle=False)
    model = new_model(len(class_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_dir = OUTPUT_DIR / "treinos" / setting.name
    train_dir.mkdir(parents=True, exist_ok=True)
    weights = train_dir / "best.pt"
    history, best_f1 = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics = evaluate(model, validation_loader)
        history.append({"epoca": epoch, "loss_treino": round(total_loss / len(train_loader.dataset), 6), **metrics})
        if float(metrics["f1_macro"]) > best_f1:
            best_f1 = float(metrics["f1_macro"])
            torch.save(model.state_dict(), weights)
        print(f"{setting.name} | época {epoch}/{epochs} | F1 validação={metrics['f1_macro']:.6f}", flush=True)

    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    validation_metrics = evaluate(model, validation_loader)
    (train_dir / "historico.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (train_dir / "configuracao.json").write_text(
        json.dumps({
            "modelo": "MobileNetV2 pré-treinada (ImageNet)", "classes": class_names,
            "epocas": epochs, "batch": batch, "imgsz": 224, "semente": SEED,
            "dispositivo": str(DEVICE), "otimizador": "AdamW", "learning_rate": 1e-4,
            "weight_decay": 1e-4, "transformacao_treino": "Resize(256), RandomResizedCrop(224), RandomHorizontalFlip, ImageNet normalization",
            "transformacao_avaliacao": "Resize(256), CenterCrop(224), ImageNet normalization",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "setting": setting.name, "classes": class_names, "epocas": epochs, "batch": batch,
        **split_counts, **{f"validacao_{key}": value for key, value in validation_metrics.items()}, "pesos": str(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--threads", type=int, default=4, help="Threads de CPU do PyTorch (padrão: 4).")
    parser.add_argument("--avaliar-setting", choices=[f"setting{number}" for number in range(1, 6)], help="Avalia um checkpoint já treinado no test, sem retreinar.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.threads < 1:
        raise ValueError("--epochs, --batch e --threads devem ser positivos.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    if args.avaliar_setting:
        setting = args.avaliar_setting
        weights = OUTPUT_DIR / "treinos" / setting / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {weights}")
        seed_everything()
        classes = sorted(path.name for path in (DATASET_ROOT / setting).iterdir() if path.is_dir())
        model = new_model(len(classes))
        model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
        metrics = evaluate(model, loader(DATASET_ROOT / "test", EVAL_TRANSFORM, args.batch, shuffle=False))
        result = {
            "setting": setting,
            "origem": "avaliação manual no teste externo",
            "classes": classes,
            "pesos": str(weights),
            "pesos_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            **metrics,
        }
        output = OUTPUT_DIR / f"metricas_teste_{setting}.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return

    settings = sorted(path for path in DATASET_ROOT.glob("setting*") if path.is_dir())
    rows = [train_one_setting(setting, args.epochs, args.batch) for setting in settings]
    best = max(rows, key=lambda row: (float(row["validacao_f1_macro"]), float(row["validacao_acuracia_balanceada"]), float(row["validacao_acuracia"])))

    test_loader = loader(DATASET_ROOT / "test", EVAL_TRANSFORM, args.batch, shuffle=False)
    model = new_model(len(best["classes"]))
    model.load_state_dict(torch.load(str(best["pesos"]), map_location=DEVICE, weights_only=True))
    test_metrics = evaluate(model, test_loader)
    best_hash = hashlib.sha256(Path(str(best["pesos"])).read_bytes()).hexdigest()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metricas_validacao.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "metricas_validacao.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "melhor_setting": best["setting"],
        "criterio": "maior F1 macro de validação; desempate por acurácia balanceada e acurácia",
        "metricas_validacao": {key: value for key, value in best.items() if key.startswith("validacao_")},
        "metricas_teste": test_metrics, "pesos": best["pesos"], "pesos_sha256": best_hash,
    }
    (OUTPUT_DIR / "melhor_setting.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# MobileNetV2 — comparação dos settings", "",
        "A seleção usa apenas validação estratificada de 20%. O `test` foi avaliado uma única vez no vencedor.", "",
        "| Setting | Imagens | Acurácia | F1 macro | Recall macro | Acurácia balanceada |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append("| {setting} | {validacao_n_imagens} | {validacao_acuracia:.4f} | {validacao_f1_macro:.4f} | {validacao_recall_macro:.4f} | {validacao_acuracia_balanceada:.4f} |".format(**row))
    report += ["", f"Vencedor por F1 macro de validação: **{best['setting']}**.", "", "## Teste externo", "", "| Imagens | Acurácia | F1 macro | Recall macro | Acurácia balanceada |", "|---:|---:|---:|---:|---:|", "| {n_imagens} | {acuracia:.4f} | {f1_macro:.4f} | {recall_macro:.4f} | {acuracia_balanceada:.4f} |".format(**test_metrics), ""]
    (OUTPUT_DIR / "RELATORIO.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
