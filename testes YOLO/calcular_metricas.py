"""Atualiza F1 macro e recall macro sem retreinar os modelos salvos."""
import json

from sklearn.metrics import f1_score, recall_score

import executar_experimentos as experiment


rows = json.loads((experiment.RESULTS / "metricas.json").read_text(encoding="utf-8"))
fresh = experiment.files(experiment.DATASET / "Teste" / experiment.CLASSES[0])
rotten = experiment.files(experiment.DATASET / "Teste" / experiment.CLASSES[1])
expected = [0] * len(fresh) + [1] * len(rotten)
sources = [str(image) for image in fresh + rotten]

for row in rows:
    model = experiment.YOLO(str(experiment.ROOT / row["modelo"]))
    predicted = [int(item.probs.top1) for item in model.predict(source=sources, imgsz=224, batch=32, device="cpu", verbose=False)]
    row["f1_macro"] = round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 6)
    row["recall_macro"] = round(float(recall_score(expected, predicted, average="macro", zero_division=0)), 6)
    row.pop("fitness", None)
    row.pop("acuracia_top5", None)

experiment.report(rows, 5)
print(f"Métricas macro atualizadas para {len(rows)} experimentos.")
