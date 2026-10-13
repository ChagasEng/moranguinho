"""Compara LMFCN sem DBSCAN e avalia o setting escolhido no teste externo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from comparar_lmfcn_dbscan_settings import (
    DATASET_ROOT, SEED, evaluate_checkpoint, files, fit, rows, seed_everything, split,
)
from morango_lmfcn.training import TrainConfig


OUTPUT_ROOT = DATASET_ROOT / "resultados_lmfcn_settings" / "sem_dbscan"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128); parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4); parser.add_argument("--avaliar-setting", default="setting2", choices=[f"setting{number}" for number in range(1, 6)])
    args = parser.parse_args(); torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    config = TrainConfig(image_size=args.image_size, features=args.features, batch_size=args.batch, epochs=args.epochs, seed=SEED)
    results, checkpoints = [], {}
    for setting in sorted(path for path in DATASET_ROOT.glob("setting*") if path.is_dir()):
        classes = sorted(path.name for path in setting.iterdir() if path.is_dir()); train, validation = split(setting, classes)
        best, _ = fit(rows(train, classes), rows(validation, classes), config, OUTPUT_ROOT / setting.name / "treino")
        checkpoints[setting.name] = (best, classes)
        results.append({"setting": setting.name, "classes": classes, "pesos": str(OUTPUT_ROOT / setting.name / "treino" / "cnn.pt"), "melhor_epoca": best["epoca"], **{f"validacao_{key}": value for key, value in best["metricas"].items()}})
    selected_best, selected_classes = checkpoints[args.avaliar_setting]
    test_rows = rows({name: files(DATASET_ROOT / "test" / name) for name in selected_classes}, selected_classes)
    test_metrics = evaluate_checkpoint(selected_best, test_rows, config)
    checkpoint = OUTPUT_ROOT / args.avaliar_setting / "treino" / "cnn.pt"; summary = OUTPUT_ROOT / "comparacao_settings"; summary.mkdir(parents=True, exist_ok=True)
    (summary / "metricas_validacao.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {"setting_avaliado_no_teste": args.avaliar_setting, "origem": "avaliação manual do LMFCN sem DBSCAN", "metricas_teste": test_metrics, "pesos": str(checkpoint), "pesos_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (summary / "setting_avaliado_teste.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__": main()
