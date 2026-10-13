"""Compara settings YOLO11-cls + DBSCAN e avalia o setting2 no teste.

O resultado comparativo oficial mantém o mesmo setting2 do baseline. Cada
setting pode continuar sendo treinado para registrar validação, mas somente o
checkpoint DBSCAN do setting2 é aplicado ao teste externo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from executar_yolo_dbscan_dataset_novo import (
    BASE_OUTPUT,
    DATASET_ROOT,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    PRETRAINED_WEIGHTS,
    SEED,
    dbscan_filter,
    evaluate,
    extract_embeddings,
    stage_dataset,
    stratified_split,
)


def train_and_validate(setting_name: str, epochs: int, batch: int) -> dict[str, object]:
    setting_root = DATASET_ROOT / setting_name
    class_names = sorted(path.name for path in setting_root.iterdir() if path.is_dir())
    train, validation = stratified_split(setting_root, class_names)
    extractor = YOLO(str(PRETRAINED_WEIGHTS))
    filtered, dbscan = {}, {}
    for class_name in class_names:
        mask, details = dbscan_filter(extract_embeddings(extractor, train[class_name], batch))
        filtered[class_name] = [image for image, selected in zip(train[class_name], mask) if selected]
        dbscan[class_name] = {
            "imagens_antes": len(train[class_name]),
            "imagens_mantidas": len(filtered[class_name]),
            **details,
        }

    output = BASE_OUTPUT / setting_name
    artifacts, metrics_dir = output / "artefatos", output / "metricas"
    internal_dataset = output / "dataset_interno"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (artifacts / "dbscan.json").write_text(json.dumps(dbscan, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_dataset(internal_dataset, filtered, validation)

    model = YOLO(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(internal_dataset), epochs=epochs, imgsz=224, batch=batch,
        device="cpu", workers=0, seed=SEED, deterministic=True,
        project=str(output), name="treino", exist_ok=True, verbose=False,
    )
    weights = output / "treino" / "weights" / "best.pt"
    validation_metrics = evaluate(weights, internal_dataset / "val", class_names, batch)
    (metrics_dir / "validacao.json").write_text(
        json.dumps(validation_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "metodo": "YOLO11n-cls + DBSCAN",
        "setting": setting_name,
        "classes": class_names,
        "semente": SEED,
        "epocas": epochs,
        "batch": batch,
        "imgsz": 224,
        "dispositivo": "cpu",
        "protocolo": "DBSCAN apenas no treino depois da divisão estratificada 80/20; validação e teste não foram filtrados.",
        "pesos_iniciais": str(PRETRAINED_WEIGHTS),
        "pesos_avaliados": str(weights),
        "dbscan": dbscan,
    }
    (artifacts / "configuracao.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "setting": setting_name,
        "classes": class_names,
        "epocas": epochs,
        "batch": batch,
        "pesos": str(weights),
        "removidas_dbscan": sum(int(details["removidas"]) for details in dbscan.values()),
        **{f"validacao_{name}": value for name, value in validation_metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--avaliar-setting", metavar="SETTING",
        help="Avalia no test o checkpoint DBSCAN já treinado desse setting, sem retreinar.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("--epochs e --batch devem ser positivos.")

    if args.avaliar_setting:
        setting = args.avaliar_setting
        weights = BASE_OUTPUT / setting / "treino" / "weights" / "best.pt"
        config = BASE_OUTPUT / setting / "artefatos" / "configuracao.json"
        if not weights.is_file() or not config.is_file():
            raise FileNotFoundError(f"Não há checkpoint DBSCAN pronto para {setting}: {weights}")
        metadata = json.loads(config.read_text(encoding="utf-8"))
        test_metrics = evaluate(weights, DATASET_ROOT / "test", list(metadata["classes"]), args.batch)
        payload = {
            "setting": setting,
            "protocolo": "Avaliação controlada: checkpoint YOLO + DBSCAN do mesmo setting no test externo, sem retreino.",
            "pesos": str(weights),
            "metricas_teste": test_metrics,
        }
        destination = BASE_OUTPUT / setting / "metricas" / "teste_setting2_controlado.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return

    settings = sorted(path.name for path in DATASET_ROOT.glob("setting*") if path.is_dir())
    rows = []
    for setting in settings:
        print(f"Treinando YOLO + DBSCAN: {setting}", flush=True)
        rows.append(train_and_validate(setting, args.epochs, args.batch))

    selected = next((row for row in rows if row["setting"] == "setting2"), None)
    if selected is None:
        raise FileNotFoundError("O setting2 é obrigatório para a avaliação comparativa.")
    print("Avaliando o setting2 + DBSCAN no teste externo...", flush=True)
    test_metrics = evaluate(Path(str(selected["pesos"])), DATASET_ROOT / "test", list(selected["classes"]), args.batch)
    winner_metrics = BASE_OUTPUT / "setting2" / "metricas" / "teste_setting2_controlado.json"
    winner_metrics.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_dir = BASE_OUTPUT / "comparacao_settings"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "metricas_validacao.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {
        "setting_avaliado_no_teste": "setting2",
        "criterio": "comparação controlada com o baseline: mesmo setting2, com e sem DBSCAN",
        "metricas_validacao": {key: value for key, value in selected.items() if key.startswith("validacao_")},
        "metricas_teste": test_metrics,
        "pesos": selected["pesos"],
    }
    (summary_dir / "melhor_setting.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# YOLO11-cls + DBSCAN — seleção dos settings",
        "",
        "Cada setting foi dividido de forma estratificada em 80% treino e 20% validação. "
        "O DBSCAN filtrou somente o treino. O `test` foi avaliado no mesmo `setting2` do baseline.",
        "",
        "| Setting | Removidas DBSCAN | Imagens validação | Acurácia | F1 macro | Recall macro | Acurácia balanceada |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {setting} | {removidas_dbscan} | {validacao_n_imagens} | {validacao_acuracia:.4f} | "
            "{validacao_f1_macro:.4f} | {validacao_recall_macro:.4f} | {validacao_acuracia_balanceada:.4f} |".format(**row)
        )
    report += [
        "",
        "## Teste externo",
        "",
        "Avaliação controlada: **setting2**.",
        "",
        "| Imagens | Acurácia | F1 macro | Recall macro | Acurácia balanceada |",
        "|---:|---:|---:|---:|---:|",
        "| {n_imagens} | {acuracia:.4f} | {f1_macro:.4f} | {recall_macro:.4f} | {acuracia_balanceada:.4f} |".format(**test_metrics),
        "",
    ]
    (summary_dir / "RELATORIO.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
