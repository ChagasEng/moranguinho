"""Executa o LMFCN original de Jonathan de Matos no dataset de morangos.

Este arquivo adapta somente a entrada de dados, a selecao do dispositivo e a
persistencia dos artefatos. A arquitetura, o otimizador, as perdas e o ciclo de
treinamento permanecem nos arquivos originais em ``codigo_jonathan/lmfcn``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch


ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "codigo_jonathan" / "lmfcn"
DEFAULT_DATASET = ROOT.parent.parent / "datasetmorangonovo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {"fresh": 0, "rotten": 1}


@dataclass(frozen=True)
class OriginalProtocol:
    epochs: int = 5
    gamma: float = 1.0
    relax: float = 1.0
    learning_rate: float = 0.1
    features: int = 2
    shake: int = 0
    svclose: int = 5
    shclose: int = 0
    wrclose: int = 1
    width: int = 350
    height: int = 230
    augmentation: int = 1
    normalization: int = 0
    seed: int = 0
    batch_size: int = 256


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_setting(
    setting_dir: Path,
    split_seed: int,
    limit_per_class: int,
) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    train: list[tuple[int, Path]] = []
    validation: list[tuple[int, Path]] = []
    for class_index, (class_name, label) in enumerate(CLASS_TO_LABEL.items()):
        paths = image_files(setting_dir / class_name)
        if limit_per_class:
            paths = paths[:limit_per_class]
        if len(paths) < 2:
            raise ValueError(f"A classe {class_name!r} precisa ter pelo menos duas imagens")
        order = np.random.default_rng(split_seed + class_index).permutation(len(paths))
        validation_count = min(len(paths) - 1, max(1, round(len(paths) * 0.2)))
        validation.extend((label, paths[index]) for index in order[:validation_count])
        train.extend((label, paths[index]) for index in order[validation_count:])
    return train, validation


def complete_setting(setting_dir: Path, limit_per_class: int) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        paths = image_files(setting_dir / class_name)
        if limit_per_class:
            paths = paths[:limit_per_class]
        if not paths:
            raise ValueError(f"Nenhuma imagem encontrada para {setting_dir.name}/{class_name}")
        rows.extend((label, path) for path in paths)
    return rows


def external_test(test_dir: Path, limit_per_class: int) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        paths = image_files(test_dir / class_name)
        if limit_per_class:
            paths = paths[:limit_per_class]
        if not paths:
            raise ValueError(f"Nenhuma imagem encontrada para test/{class_name}")
        rows.extend((label, path) for path in paths)
    return rows


def write_manifest(path: Path, rows: list[tuple[int, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # O carregador original remove sempre o ultimo caractere de cada linha;
    # por isso inclusive a ultima linha precisa terminar com quebra de linha.
    content = "".join(f"{label};{image_path}\n" for label, image_path in rows)
    path.write_text(content, encoding="utf-8")


def counts(rows: list[tuple[int, Path]]) -> dict[str, int]:
    return {
        class_name: sum(label == expected for label, _ in rows)
        for class_name, expected in CLASS_TO_LABEL.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    defaults = OriginalProtocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--setting", default="setting1")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--data-protocol",
        choices=("external", "holdout"),
        default="external",
        help="external treina com todo o setting e valida em test; holdout preserva a divisao 80/20",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-per-class", type=int, default=0, help="Somente para smoke test; zero usa todas as imagens")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--relax", type=float, default=defaults.relax)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--features", type=int, default=defaults.features)
    parser.add_argument("--shake", type=int, default=defaults.shake)
    parser.add_argument("--svclose", type=int, default=defaults.svclose)
    parser.add_argument("--shclose", type=int, default=defaults.shclose)
    parser.add_argument("--wrclose", type=int, default=defaults.wrclose)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--aug", type=int, default=defaults.augmentation)
    parser.add_argument("--norm", type=int, default=defaults.normalization)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--batch", type=int, default=defaults.batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (args.output or (dataset / "resultados_lmfcn_original" / f"{args.setting}_{args.data_protocol}")).resolve()
    setting_dir = dataset / args.setting
    test_dir = dataset / "test"
    if not setting_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Dataset invalido: esperados {setting_dir} e {test_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nao esta disponivel")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)

    external_rows = external_test(test_dir, args.limit_per_class)
    if args.data_protocol == "external":
        train_rows = complete_setting(setting_dir, args.limit_per_class)
        validation_rows = external_rows
        test_rows: list[tuple[int, Path]] = []
    else:
        train_rows, validation_rows = split_setting(setting_dir, args.split_seed, args.limit_per_class)
        test_rows = external_rows
    manifests = output / "manifests"
    train_manifest = manifests / "train.txt"
    write_manifest(train_manifest, train_rows)
    validation_manifest = manifests / "val.txt"
    write_manifest(validation_manifest, validation_rows)
    test_manifest = manifests / "test.txt"
    if test_rows:
        write_manifest(test_manifest, test_rows)

    # O modulo original usa uma variavel global chamada `cuda`. Redireciona-la
    # e a unica adaptacao necessaria para executar o mesmo algoritmo em CPU.
    sys.path.insert(0, str(ORIGINAL_DIR))
    import LMFCN as original_module  # pylint: disable=import-outside-toplevel,import-error
    from LoadImagesGeneral import LoadImages  # pylint: disable=import-outside-toplevel,import-error

    original_module.cuda = device
    train_loader = LoadImages(
        imglist=str(train_manifest), batch_size=args.batch, aug=args.aug,
        width=args.width, height=args.height,
    )
    validation_loader = LoadImages(
        imglist=str(validation_manifest), batch_size=args.batch, aug=1,
        width=args.width, height=args.height,
    )
    train_loader.generate_batches()
    validation_loader.generate_batches()
    test_loader = None
    if test_rows:
        test_loader = LoadImages(
            imglist=str(test_manifest), batch_size=args.batch, aug=1,
            width=args.width, height=args.height,
        )
        test_loader.generate_batches()

    model = original_module.LMFCN(
        feature=args.features,
        width=args.width,
        height=args.height,
        epoch=args.epochs,
        gamma=args.gamma,
        relax=args.relax,
        lr=args.lr,
        shake=args.shake,
        shclose=args.shclose,
        svclose=args.svclose,
        wrclose=args.wrclose,
        aug=args.aug,
        norm=args.norm,
        seed=args.seed,
        outdir=str(output),
        outfile="lmfcn_original",
        batch=args.batch,
        i=None,
    )
    model.li_tr = train_loader
    model.li_val = validation_loader
    model.li_ts = test_loader

    history = model.fit()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "modelo_final.pt"
    torch.save(model.model.state_dict(), checkpoint)
    joblib.dump(model.clf, output / "svm_final.joblib")
    np.savez_compressed(
        output / "referencia_final.npz",
        features=model.X.detach().cpu().numpy(),
        gamma=float(model.gamma_auto),
    )

    best_index = max(range(len(history)), key=lambda index: history[index].get("val balanced acc", float("-inf")))
    protocol = OriginalProtocol(
        epochs=args.epochs,
        gamma=args.gamma,
        relax=args.relax,
        learning_rate=args.lr,
        features=args.features,
        shake=args.shake,
        svclose=args.svclose,
        shclose=args.shclose,
        wrclose=args.wrclose,
        width=args.width,
        height=args.height,
        augmentation=args.aug,
        normalization=args.norm,
        seed=args.seed,
        batch_size=args.batch,
    )
    if args.data_protocol == "external":
        dataset_counts = {
            "train_setting": counts(train_rows),
            "dataset_test_used_for_validation": counts(validation_rows),
        }
    else:
        dataset_counts = {
            "train": counts(train_rows),
            "validation": counts(validation_rows),
            "external_test": counts(test_rows),
        }

    result = {
        "implementation": "codigo_jonathan/lmfcn (original, sem alteracoes)",
        "compatibility_adapter": "device, manifests and artifact persistence only",
        "setting": args.setting,
        "data_protocol": args.data_protocol,
        "validation_source": "dataset/test" if args.data_protocol == "external" else "20% of setting",
        "split_seed": args.split_seed,
        "counts": dataset_counts,
        "protocol": asdict(protocol),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
            "threads": args.threads,
        },
        "history": history,
        "best_validation_epoch": best_index + 1,
        "best_validation_metrics": history[best_index],
        "final_checkpoint_sha256": sha256(checkpoint),
    }
    (output / "resultados.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
