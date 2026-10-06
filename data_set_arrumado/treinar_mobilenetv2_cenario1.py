"""Treina MobileNetV2 no Cenário 1 e avalia uma vez no conjunto Teste."""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "Nusrat Sultana"
OUTPUT = ROOT / "resultados_mobilenetv2_cenario1_5epocas"
CLASSES = ("FreshStrawberry", "RottenStrawberry")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_TF = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
EVAL_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try: os.link(source, target)
        except OSError: shutil.copy2(source, target)

def prepare() -> Path:
    data = OUTPUT / "dados_internos"
    for position, label in enumerate(CLASSES):
        source = images(BASE / "Cenario1" / label); order = np.random.default_rng(42 + position).permutation(len(source)); val_count = round(len(source)*.2)
        for split, indices in (("val",order[:val_count]),("train",order[val_count:])):
            for index in indices: link(source[index], data / split / label / f"{label}__{source[index].name}")
    return data

@torch.inference_mode()
def evaluate(model: nn.Module, data: DataLoader) -> tuple[dict[str,float],list[int],list[int]]:
    model.eval(); expected, predicted = [], []
    for batch, labels in data:
        predicted += model(batch.to(DEVICE)).argmax(1).cpu().tolist(); expected += labels.tolist()
    return {"acuracia":round(sum(a==b for a,b in zip(expected,predicted))/len(expected),6),"f1_macro":round(float(f1_score(expected,predicted,average="macro",zero_division=0)),6),"recall_macro":round(float(recall_score(expected,predicted,average="macro",zero_division=0)),6)}, expected, predicted

def main() -> None:
    random.seed(42); np.random.seed(42); torch.manual_seed(42); OUTPUT.mkdir(exist_ok=True)
    root = prepare(); train = DataLoader(datasets.ImageFolder(root/"train",transform=TRAIN_TF),batch_size=32,shuffle=True,num_workers=0); validation = DataLoader(datasets.ImageFolder(root/"val",transform=EVAL_TF),batch_size=32,num_workers=0)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT); model.classifier[1] = nn.Linear(model.classifier[1].in_features,2); model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); criterion = nn.CrossEntropyLoss(); weights = OUTPUT / "melhor.pt"; best = -1.0; history = []
    print("Treino iniciado: Cenário 1, MobileNetV2, 5 épocas.",flush=True)
    for epoch in range(1,6):
        model.train(); total = 0.0
        for batch, labels in train:
            optimizer.zero_grad(); loss = criterion(model(batch.to(DEVICE)),labels.to(DEVICE)); loss.backward(); optimizer.step(); total += loss.item()*len(labels)
        metrics,_,_ = evaluate(model,validation); history.append({"epoca": epoch, "acuracia": metrics["acuracia"], "f1_macro": metrics["f1_macro"], "recall_macro": metrics["recall_macro"], "loss": total/len(train.dataset)}); print(f"Época {epoch}/5 | val_acuracia={metrics['acuracia']:.4f} | loss={total/len(train.dataset):.4f} | val_f1_macro={metrics['f1_macro']:.4f} | val_recall_macro={metrics['recall_macro']:.4f}",flush=True)
        if metrics["f1_macro"] > best: best = metrics["f1_macro"]; torch.save(model.state_dict(),weights)
    model.load_state_dict(torch.load(weights,map_location=DEVICE)); model.to(DEVICE)
    test_root = OUTPUT / "dados_teste"
    for label in CLASSES:
        for image in images(BASE/"Teste"/label): link(image,test_root/label/f"{label}__{image.name}")
    metrics, expected, predicted = evaluate(model,DataLoader(datasets.ImageFolder(test_root,transform=EVAL_TF),batch_size=32,num_workers=0))
    metrics.update({"cenario":"Cenario1","modelo":"MobileNetV2","epocas":5,"teste_sadios":40,"teste_danificados":40,"pesos":"melhor.pt"})
    (OUTPUT/"metricas.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUTPUT/"metricas.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(metrics)); writer.writeheader(); writer.writerow(metrics)
    fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(confusion_matrix(expected,predicted,labels=[0,1]),display_labels=["Saudável","Danificado"]).plot(ax=ax,cmap="Blues",colorbar=False); ax.set_title("Matriz de confusão — MobileNetV2, Cenário 1"); fig.tight_layout(); fig.savefig(OUTPUT/"matriz_confusao.png",dpi=180); plt.close(fig)
    epoch_table = "\n".join("| {epoca} | {acuracia:.4f} | {f1_macro:.4f} | {recall_macro:.4f} | {loss:.4f} |".format(**item) for item in history)
    (OUTPUT/"RELATORIO.md").write_text(f"""# Relatório — MobileNetV2, Cenário 1 (5 épocas)

Treino: 80% do `Cenario1`; validação interna estratificada: 20%. MobileNetV2 pré-treinada, sem DBSCAN, seed 42. O `Teste` foi usado somente após o treino (40 saudáveis e 40 danificados).

| Acurácia | F1 macro | Recall macro |
|---:|---:|---:|
| {metrics['acuracia']:.4f} | {metrics['f1_macro']:.4f} | {metrics['recall_macro']:.4f} |

## Evolução por época — validação interna

| Época | Acurácia | F1 macro | Recall macro | Loss de treino |
|---:|---:|---:|---:|---:|
{epoch_table}

Arquivos: `metricas.json`, `metricas.csv`, `matriz_confusao.png` e `melhor.pt`.
""",encoding="utf-8")
    print("Teste final concluído. Relatório, métricas e matriz de confusão foram criados.",flush=True)

if __name__ == "__main__": main()
