"""Compara os settings do datasetmorangonovo com YOLO11-cls.

O teste externo nunca participa da escolha do setting: cada setting e separado em
80% treino e 20% validacao, de forma estratificada. Depois, o melhor modelo de
validacao e avaliado uma unica vez em ``datasetmorangonovo/test``.

Uso:
    .\\.venv-yolo\\Scripts\\python.exe "testes YOLO\\executar_settings_dataset_novo.py"
    .\\.venv-yolo\\Scripts\\python.exe "testes YOLO\\executar_settings_dataset_novo.py" --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from ultralytics import YOLO


# Altere este valor, ou use --epochs, para rodar mais epocas.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 32
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
# O projeto fica em Documents/artigo/moranguinho e o dataset em Documents/.
DATASET_ROOT = SCRIPT_DIR.parent.parent.parent / "datasetmorangonovo"
OUTPUT_DIR = DATASET_ROOT / "resultados_yolo_settings"
PRETRAINED_WEIGHTS = SCRIPT_DIR.parent / "yolo11n-cls.pt"


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_or_copy(source: Path, destination: Path) -> None:
    """Evita duplicar imagens quando o sistema suporta hard links."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def create_split(setting: Path, split_root: Path, validation_fraction: float = 0.2) -> tuple[list[str], dict[str, int]]:
    """Cria train/val estratificados para o formato de classificacao do YOLO."""
    class_names = sorted(directory.name for directory in setting.iterdir() if directory.is_dir())
    if len(class_names) < 2:
        raise ValueError(f"{setting} precisa conter pelo menos duas classes.")

    counts: dict[str, int] = {}
    for class_index, class_name in enumerate(class_names):
        images = image_files(setting / class_name)
        if len(images) < 2:
            raise ValueError(f"A classe {class_name} de {setting.name} tem menos de duas imagens.")
        shuffled = [images[index] for index in np.random.default_rng(SEED + class_index).permutation(len(images))]
        validation_count = min(len(images) - 1, max(1, round(len(images) * validation_fraction)))
        partitions = {"val": shuffled[:validation_count], "train": shuffled[validation_count:]}
        counts[f"{class_name}_treino"] = len(partitions["train"])
        counts[f"{class_name}_validacao"] = len(partitions["val"])
        for split_name, paths in partitions.items():
            for image_index, image in enumerate(paths):
                # O indice tambem previne colisao se houver nomes repetidos em subpastas.
                link_or_copy(image, split_root / split_name / class_name / f"{image_index:05d}_{image.name}")
    return class_names, counts


def evaluate(weights: Path, data_root: Path, class_names: list[str], batch: int) -> dict[str, float | int]:
    expected: list[int] = []
    sources: list[str] = []
    for label, class_name in enumerate(class_names):
        images = image_files(data_root / class_name)
        expected.extend([label] * len(images))
        sources.extend(str(image) for image in images)
    if not sources:
        raise ValueError(f"Nenhuma imagem encontrada para avaliar em {data_root}")

    model = YOLO(str(weights))
    predictions = [
        int(result.probs.top1)
        for result in model.predict(source=sources, imgsz=224, batch=batch, device="cpu", verbose=False)
    ]
    return {
        "n_imagens": len(expected),
        "acuracia": round(float(accuracy_score(expected, predictions)), 6),
        "f1_macro": round(float(f1_score(expected, predictions, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(expected, predictions, average="macro", zero_division=0)), 6),
        "acuracia_balanceada": round(float(balanced_accuracy_score(expected, predictions)), 6),
    }


def train_one_setting(setting: Path, epochs: int, batch: int) -> dict[str, object]:
    split_root = OUTPUT_DIR / "datasets_internos" / setting.name
    class_names, split_counts = create_split(setting, split_root)
    model = YOLO(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(split_root),
        epochs=epochs,
        imgsz=224,
        batch=batch,
        device="cpu",
        workers=0,
        seed=SEED,
        deterministic=True,
        project=str(OUTPUT_DIR / "treinos"),
        name=setting.name,
        exist_ok=True,
        verbose=False,
    )
    weights = OUTPUT_DIR / "treinos" / setting.name / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Pesos nao encontrados apos o treino: {weights}")
    validation = evaluate(weights, split_root / "val", class_names, batch)
    return {
        "setting": setting.name,
        "classes": class_names,
        "epocas": epochs,
        "batch": batch,
        **split_counts,
        **{f"validacao_{key}": value for key, value in validation.items()},
        "pesos": str(weights),
    }


def save_report(rows: list[dict[str, object]], best: dict[str, object], test_metrics: dict[str, float | int]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metricas_validacao.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "metricas_validacao.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selection = {
        "melhor_setting": best["setting"],
        "criterio": "maior F1 macro de validacao; desempate por acuracia balanceada e acuracia",
        "metricas_validacao": {key: value for key, value in best.items() if key.startswith("validacao_")},
        "metricas_teste": test_metrics,
        "pesos": best["pesos"],
    }
    (OUTPUT_DIR / "melhor_setting.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = [
        "# Comparacao YOLO11-cls — datasetmorangonovo",
        "",
        f"Foram treinados os cinco settings por **{best['epocas']} epocas** (semente {SEED}). "
        "A selecao usa apenas a validacao estratificada de 20%; o diretorio `test` nao foi usado nessa decisao.",
        "",
        "## Validacao interna",
        "",
        "| Setting | Imagens | Acuracia | F1 macro | Recall macro | Acuracia balanceada |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {setting} | {validacao_n_imagens} | {validacao_acuracia:.4f} | {validacao_f1_macro:.4f} | "
            "{validacao_recall_macro:.4f} | {validacao_acuracia_balanceada:.4f} |".format(**row)
        )
    table += [
        "",
        "## Recomendacao",
        "",
        f"Use **{best['setting']}**. Ele obteve o maior F1 macro na validacao interna "
        f"({best['validacao_f1_macro']:.4f}).",
        "",
        "## Teste externo (avaliado uma unica vez no melhor modelo)",
        "",
        "| Imagens | Acuracia | F1 macro | Recall macro | Acuracia balanceada |",
        "|---:|---:|---:|---:|---:|",
        "| {n_imagens} | {acuracia:.4f} | {f1_macro:.4f} | {recall_macro:.4f} | {acuracia_balanceada:.4f} |".format(**test_metrics),
        "",
        "Observacao: em classificacao binaria, recall macro e acuracia balanceada sao equivalentes quando "
        "calculados sem ajuste; ambos foram reportados porque foram solicitados.",
    ]
    (OUTPUT_DIR / "RELATORIO.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epocas por setting (padrao: 3).")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size (padrao: 32).")
    parser.add_argument(
        "--avaliar-setting",
        choices=[f"setting{number}" for number in range(1, 6)],
        help="Avalia somente um setting no diretorio test, sem treinar nem alterar a selecao.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("--epochs e --batch devem ser maiores que zero.")
    if not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError(f"Pesos YOLO nao encontrados: {PRETRAINED_WEIGHTS}")

    if args.avaliar_setting:
        setting = args.avaliar_setting
        weights = OUTPUT_DIR / "treinos" / setting / "weights" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Pesos nao encontrados: {weights}")
        class_names = sorted(path.name for path in (DATASET_ROOT / "test").iterdir() if path.is_dir())
        metrics = evaluate(weights, DATASET_ROOT / "test", class_names, args.batch)
        output = OUTPUT_DIR / f"metricas_teste_{setting}.json"
        output.write_text(
            json.dumps(
                {
                    "setting": setting,
                    "classes": class_names,
                    "pesos": str(weights),
                    "diretorio_teste": str(DATASET_ROOT / "test"),
                    **metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False))
        print(f"Métricas salvas em: {output}")
        return

    settings = sorted(path for path in DATASET_ROOT.glob("setting*") if path.is_dir())
    if not settings:
        raise FileNotFoundError(f"Nenhum setting encontrado em {DATASET_ROOT}")
    rows = []
    for setting in settings:
        print(f"Treinando {setting.name} por {args.epochs} epocas...", flush=True)
        rows.append(train_one_setting(setting, args.epochs, args.batch))

    best = max(
        rows,
        key=lambda row: (
            float(row["validacao_f1_macro"]),
            float(row["validacao_acuracia_balanceada"]),
            float(row["validacao_acuracia"]),
        ),
    )
    class_names = list(best["classes"])
    print(f"Avaliando {best['setting']} no teste externo...", flush=True)
    test_metrics = evaluate(Path(str(best["pesos"])), DATASET_ROOT / "test", class_names, args.batch)
    save_report(rows, best, test_metrics)
    print(f"Concluido. Melhor setting: {best['setting']}. Resultados: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
