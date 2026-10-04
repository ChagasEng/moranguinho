"""Preparação e validação dos manifestos no formato usado por Jonathan."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
CLASS_DIRS = {0: "FreshStrawberry", 1: "RottenStrawberry"}


def prepare(dataset: Path, output: Path, seed: int = 42) -> dict[str, Counter]:
    source = dataset.resolve() / "Nusrat Sultana" / "Original"
    if not source.is_dir():
        raise ValueError(f"Diretório de imagens originais não encontrado: {source}")
    if output.exists() and any((output / f"{name}.txt").exists() for name in ("train", "val", "test")):
        raise ValueError(f"Manifestos já existem em {output}; escolha outra pasta de saída")

    rng = random.Random(seed)
    splits: dict[str, list[tuple[int, Path]]] = {"train": [], "val": [], "test": []}
    for label, directory in CLASS_DIRS.items():
        image_dir = source / directory
        if not image_dir.is_dir():
            raise ValueError(f"Classe ausente: {image_dir}")
        files = sorted(p.resolve() for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) < 3:
            raise ValueError(f"São necessárias ao menos 3 imagens na classe {directory}")
        rng.shuffle(files)
        n_train = int(len(files) * 0.7)
        n_val = int(len(files) * 0.15)
        for name, subset in (("train", files[:n_train]), ("val", files[n_train:n_train+n_val]), ("test", files[n_train+n_val:])):
            splits[name].extend((label, path) for path in subset)

    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in splits.items():
        rng.shuffle(rows)
        (output / f"{name}.txt").write_text(
            "".join(f"{label};{path}\n" for label, path in rows), encoding="utf-8"
        )
        counts[name] = Counter(label for label, _ in rows)
    return counts


def read_manifest(path: Path) -> list[tuple[int, Path]]:
    if not path.is_file():
        raise ValueError(f"Manifesto não encontrado: {path}")
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            label_text, image_text = raw.split(";", 1)
            label = int(label_text)
            image = Path(image_text.strip())
        except ValueError as exc:
            raise ValueError(f"Linha inválida em {path}:{line_number}: {raw!r}") from exc
        if label not in CLASS_DIRS or not image.is_file():
            raise ValueError(f"Classe ou imagem inválida em {path}:{line_number}: {raw!r}")
        rows.append((label, image.resolve()))
    if not rows:
        raise ValueError(f"Manifesto vazio: {path}")
    return rows


def validate_splits(train: list, val: list, test: list) -> None:
    for name, rows in (("treino", train), ("validação", val), ("teste", test)):
        counts = Counter(label for label, _ in rows)
        if set(counts) != set(CLASS_DIRS):
            raise ValueError(f"A divisão de {name} precisa conter as classes 0 e 1")
    paths = [{path for _, path in rows} for rows in (train, val, test)]
    if any(paths[i] & paths[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Há imagens repetidas entre treino, validação e teste")
