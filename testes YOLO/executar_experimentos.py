"""Protocolo sem vazamento: validação interna e teste final único."""
from __future__ import annotations
import argparse, csv, json, math, os, shutil
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "Dataset" / "Nusrat Sultana"
ARTIFACTS, RESULTS = ROOT / "artefatos_validacao_5epocas", ROOT / "resultados_validacao_5epocas"
CLASSES, SCENARIOS = ("FreshStrawberry", "RottenStrawberry"), tuple(f"Cenario{i}" for i in range(1, 6))

def files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try: os.link(source, target)
        except OSError: shutil.copy2(source, target)

def split_stratified(data: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    train, validation = {}, {}
    for pos, label in enumerate(CLASSES):
        images, order = data[label], np.random.default_rng(42 + pos).permutation(len(data[label]))
        count = max(1, round(len(images) * .2))
        validation[label] = [images[i] for i in order[:count]]
        train[label] = [images[i] for i in order[count:]]
    return train, validation

def prepare(name: str, train: dict[str, list[Path]], validation: dict[str, list[Path]]) -> Path:
    folder = ARTIFACTS / name
    for split, groups in (("train", train), ("val", validation)):
        for label, images in groups.items():
            for image in images: link(image, folder / split / label / f"{label}__{image.name}")
    return folder

def embeddings(model: YOLO, images: list[Path], batch: int) -> np.ndarray:
    output = []
    for start in range(0, len(images), batch):
        values = model.embed([str(p) for p in images[start:start + batch]], imgsz=224, batch=batch, device="cpu", verbose=False)
        output.append(np.stack([v.cpu().numpy().reshape(-1) for v in values]))
    return np.concatenate(output)

def dbscan_filter(vectors: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    n = len(vectors)
    if n < 12: return np.ones(n, dtype=bool), {"removidas": 0}
    scaled = StandardScaler().fit_transform(vectors)
    reduced = PCA(n_components=min(32, n - 1, scaled.shape[1]), random_state=42).fit_transform(scaled)
    minimum = max(5, min(20, math.ceil(n * .02)))
    dist = NearestNeighbors(n_neighbors=minimum).fit(reduced).kneighbors(reduced)[0][:, -1]
    labels = DBSCAN(eps=float(np.quantile(dist, .90)), min_samples=minimum, n_jobs=-1).fit_predict(reduced)
    keep = labels != -1
    if not keep.any(): keep[:] = True
    return keep, {"removidas": int((~keep).sum()), "min_samples": minimum}

def evaluate(weights: Path, groups: dict[str, list[Path]], batch: int) -> dict[str, float]:
    fresh, rotten = groups[CLASSES[0]], groups[CLASSES[1]]
    expected = [0] * len(fresh) + [1] * len(rotten)
    model = YOLO(str(weights))
    predicted = [int(x.probs.top1) for x in model.predict(source=[str(x) for x in fresh + rotten], imgsz=224, batch=batch, device="cpu", verbose=False)]
    return {"acuracia_top1": round(sum(a == b for a, b in zip(expected, predicted)) / len(expected), 6),
            "f1_macro": round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 6),
            "recall_macro": round(float(recall_score(expected, predicted, average="macro", zero_division=0)), 6)}

def fit(name: str, train_groups: dict[str, list[Path]], validation: dict[str, list[Path]], epochs: int, batch: int) -> tuple[Path, dict[str, float]]:
    model = YOLO("yolo11n-cls.pt")
    model.train(data=str(prepare(name, train_groups, validation)), epochs=epochs, imgsz=224, batch=batch, device="cpu", workers=0, seed=42, deterministic=True, project=str(RESULTS / "treinos"), name=name, exist_ok=True, verbose=False)
    weights = RESULTS / "treinos" / name / "weights" / "best.pt"
    return weights, evaluate(weights, validation, batch)

def report(rows: list[dict[str, object]], best: dict[str, object] | None = None, final: dict[str, float] | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metricas_validacao.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "metricas_validacao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else []); w.writeheader() if rows else None; w.writerows(rows)
    lines = ["# Relatório — seleção por validação e teste final", "", "`Teste` não participou de treino ou seleção. Cada cenário usa 80% treino e 20% validação estratificada.", "", "## Validação", "", "| Cenário | Método | Sadios treino | Danificados treino | Removidas DBSCAN | Acurácia | F1 macro | Recall macro |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += ["| {cenario} | {metodo} | {sadios_treino} | {danificados_treino} | {removidas_dbscan} | {acuracia_top1:.4f} | {f1_macro:.4f} | {recall_macro:.4f} |".format(**r) for r in rows]
    if best and final:
        lines += ["", "## Teste final (uso único)", "", f"Melhor configuração por F1 macro de validação: **{best['cenario']} — {best['metodo']}**.", "", "| Acurácia | F1 macro | Recall macro |", "|---:|---:|---:|", "| {acuracia_top1:.4f} | {f1_macro:.4f} | {recall_macro:.4f} |".format(**final)]
    (ROOT / "RELATORIO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch", type=int, default=32); args = parser.parse_args()
    rows, saved, extractor = [], {}, YOLO("yolo11n-cls.pt")
    for scenario in SCENARIOS:
        original = {c: files(DATASET / scenario / c) for c in CLASSES}; base_train, validation = split_stratified(original)
        weights, metrics = fit(f"{scenario}_yolo", base_train, validation, args.epochs, args.batch)
        rows.append({"cenario": scenario, "metodo": "YOLO", "sadios_treino": len(base_train[CLASSES[0]]), "danificados_treino": len(base_train[CLASSES[1]]), "removidas_dbscan": 0, **metrics}); saved[(scenario, "YOLO")] = (base_train, validation)
        filtered, details = {}, {}
        for label, images in base_train.items():
            mask, info = dbscan_filter(embeddings(extractor, images, args.batch)); filtered[label] = [x for x, selected in zip(images, mask) if selected]; details[label] = info
        ARTIFACTS.mkdir(exist_ok=True); (ARTIFACTS / f"{scenario}_dbscan.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        weights, metrics = fit(f"{scenario}_yolo_dbscan", filtered, validation, args.epochs, args.batch)
        rows.append({"cenario": scenario, "metodo": "YOLO + DBSCAN", "sadios_treino": len(filtered[CLASSES[0]]), "danificados_treino": len(filtered[CLASSES[1]]), "removidas_dbscan": sum(len(base_train[c])-len(filtered[c]) for c in CLASSES), **metrics}); saved[(scenario, "YOLO + DBSCAN")] = (filtered, validation)
        report(rows); print(f"Concluído: {scenario}", flush=True)
    best = max(rows, key=lambda r: (r["f1_macro"], r["recall_macro"], r["acuracia_top1"]))
    groups, validation = saved[(best["cenario"], best["metodo"])]
    final_weights, _ = fit("modelo_final", groups, validation, args.epochs, args.batch)
    final = evaluate(final_weights, {c: files(DATASET / "Teste" / c) for c in CLASSES}, args.batch)
    report(rows, best, final); print("Teste final concluído.", flush=True)

if __name__ == "__main__": main()
