"""Mede a inferência dos três modelos no Raspberry Pi, com batch=1."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pesos"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = torch.device("cpu")
MOBILENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FeatureCNN(nn.Module):
    def __init__(self, features: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, features, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    distances = (left * left).sum(1)[:, None] + (right * right).sum(1)[None, :] - 2 * left @ right.T
    return np.exp(-gamma * np.maximum(distances, 0.0))


def images_in_memory(folder: Path) -> list[Image.Image]:
    paths = sorted(path for path in folder.rglob("*") if path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {folder}")
    loaded = []
    for path in paths:
        with Image.open(path) as image:
            loaded.append(image.convert("RGB").copy())
    return loaded


def load_predictors() -> dict[str, object]:
    yolo = YOLO(str(WEIGHTS / "yolo11_dbscan_setting2_best.pt"))

    mobile = models.mobilenet_v2(weights=None)
    mobile.classifier[1] = nn.Linear(mobile.classifier[1].in_features, 2)
    mobile.load_state_dict(torch.load(WEIGHTS / "mobilenetv2_setting2_best.pt", map_location=DEVICE, weights_only=True))
    mobile.eval()

    lmfcn = FeatureCNN().to(DEVICE)
    lmfcn.load_state_dict(torch.load(WEIGHTS / "lmfcn_dbscan_setting2_cnn.pt", map_location=DEVICE, weights_only=True))
    lmfcn.eval()
    classifier = joblib.load(WEIGHTS / "lmfcn_dbscan_setting2_svm.joblib")
    reference = np.load(WEIGHTS / "lmfcn_dbscan_setting2_reference.npz")

    def yolo_predict(image: Image.Image) -> None:
        yolo(image, verbose=False)

    def mobile_predict(image: Image.Image) -> None:
        with torch.inference_mode():
            mobile(MOBILENET_TRANSFORM(image).unsqueeze(0))

    def lmfcn_predict(image: Image.Image) -> None:
        array = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            features = lmfcn(tensor).numpy()
        classifier.predict(rbf_kernel(features, reference["features"], float(reference["gamma"])))

    return {"yolo11_dbscan": yolo_predict, "mobilenetv2": mobile_predict, "lmfcn_svm_dbscan": lmfcn_predict}


def measure(predict, images: list[Image.Image], warmup: int) -> dict[str, float | int]:
    for image in images[:min(warmup, len(images))]:
        predict(image)
    samples = []
    for image in images:
        started = time.perf_counter_ns()
        predict(image)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "n_imagens": len(samples),
        "aquecimento": min(warmup, len(images)),
        "media_ms_por_imagem": statistics.fmean(samples),
        "mediana_ms_por_imagem": statistics.median(samples),
        "p95_ms_por_imagem": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagens", type=Path, required=True, help="Pasta com as imagens de teste.")
    parser.add_argument("--aquecimento", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--saida", type=Path, default=Path("benchmark_raspberry_pi.json"))
    args = parser.parse_args()
    if args.aquecimento < 0 or args.threads < 1:
        raise ValueError("--aquecimento deve ser >= 0 e --threads deve ser >= 1.")
    torch.set_num_threads(args.threads)
    images = images_in_memory(args.imagens)
    predictors = load_predictors()
    result = {
        "hardware": {"plataforma": platform.platform(), "processador": platform.processor(), "threads_torch": args.threads},
        "protocolo": "pesos carregados uma vez; imagens pré-carregadas na RAM; batch=1; inclui pré-processamento e predição; não inclui leitura de disco nem carregamento dos pesos",
        "resultados": {name: measure(predict, images, args.aquecimento) for name, predict in predictors.items()},
    }
    args.saida.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
