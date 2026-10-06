"""Treina YOLO (classificação) no Cenário 1 e avalia uma vez no Teste."""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, recall_score
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "Nusrat Sultana"
OUTPUT = ROOT / "resultados_yolo_cenario1_5epocas"
CLASSES = ("FreshStrawberry", "RottenStrawberry")

def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try: os.link(source, target)
        except OSError: shutil.copy2(source, target)

def prepare_train_validation() -> Path:
    """Cria treino/validação estratificados sem modificar o dataset-fonte."""
    data = OUTPUT / "dados_internos"
    for position, label in enumerate(CLASSES):
        source = images(BASE / "Cenario1" / label)
        order = np.random.default_rng(42 + position).permutation(len(source))
        validation_count = round(len(source) * .2)
        for split, indexes in (("val", order[:validation_count]), ("train", order[validation_count:])):
            for index in indexes: link(source[index], data / split / label / f"{label}__{source[index].name}")
    return data

def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    data = prepare_train_validation()
    model = YOLO("yolo11n-cls.pt")
    print("Treino iniciado: Cenário 1, YOLO, 5 épocas.", flush=True)
    model.train(data=str(data), epochs=5, imgsz=224, batch=32, device="cpu", workers=0, seed=42,
                deterministic=True, project=str(OUTPUT / "treinos"), name="yolo_cenario1", exist_ok=True, verbose=False)
    weights = OUTPUT / "treinos" / "yolo_cenario1" / "weights" / "best.pt"
    with (OUTPUT / "treinos" / "yolo_cenario1" / "results.csv").open(newline="", encoding="utf-8") as history_file:
        history = list(csv.DictReader(history_file))
    epoch_table = "\n".join(
        "| {epoch} | {accuracy:.4f} | {train_loss:.4f} | {val_loss:.4f} |".format(
            epoch=row["epoch"], accuracy=float(row["metrics/accuracy_top1"]),
            train_loss=float(row["train/loss"]), val_loss=float(row["val/loss"])
        ) for row in history
    )
    test_paths = images(BASE / "Teste" / CLASSES[0]) + images(BASE / "Teste" / CLASSES[1])
    expected = [0] * len(images(BASE / "Teste" / CLASSES[0])) + [1] * len(images(BASE / "Teste" / CLASSES[1]))
    results = YOLO(str(weights)).predict(source=[str(path) for path in test_paths], imgsz=224, batch=32, device="cpu", verbose=False)
    predicted = [int(item.probs.top1) for item in results]
    metrics = {
        "cenario": "Cenario1", "modelo": "YOLO11n-cls", "epocas": 5,
        "teste_sadios": expected.count(0), "teste_danificados": expected.count(1),
        "acuracia": round(sum(a == b for a, b in zip(expected, predicted)) / len(expected), 6),
        "f1_macro": round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(expected, predicted, average="macro", zero_division=0)), 6),
        "pesos": str(weights.relative_to(ROOT)),
    }
    (OUTPUT / "metricas.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "metricas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics)); writer.writeheader(); writer.writerow(metrics)
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Saudável", "Danificado"]).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Matriz de confusão — YOLO, Cenário 1")
    figure.tight_layout(); figure.savefig(OUTPUT / "matriz_confusao.png", dpi=180); plt.close(figure)
    report = f"""# Relatório — YOLO, Cenário 1 (5 épocas)

## Protocolo

- Treino: `Nusrat Sultana/Cenario1` (80% treino interno e 20% validação interna estratificada).
- Modelo: YOLO11n para classificação, sem DBSCAN, 5 épocas e seed 42.
- Teste final: `Nusrat Sultana/Teste`, usado somente após o treino (40 saudáveis e 40 danificados).

## Métricas no teste final

| Acurácia | F1 macro | Recall macro |
|---:|---:|---:|
| {metrics['acuracia']:.4f} | {metrics['f1_macro']:.4f} | {metrics['recall_macro']:.4f} |

## Evolução por época — validação interna

| Época | Acurácia | Loss de treino | Loss de validação |
|---:|---:|---:|---:|
{epoch_table}

## Arquivos gerados

- `metricas.json` e `metricas.csv`: valores brutos das métricas;
- `matriz_confusao.png`: matriz de confusão do teste;
- `treinos/yolo_cenario1/weights/best.pt`: melhor modelo selecionado pela validação interna.
"""
    (OUTPUT / "RELATORIO.md").write_text(report, encoding="utf-8")
    print("Teste final concluído. Relatório, métricas e matriz de confusão foram criados.", flush=True)

if __name__ == "__main__":
    main()
