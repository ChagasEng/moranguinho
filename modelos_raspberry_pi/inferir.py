"""Inferência de uma imagem com os modelos selecionados para o Raspberry Pi."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pesos"
CLASSES = ["fresh", "rotten"]
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


def predict_yolo(image: Path) -> str:
    from ultralytics import YOLO
    result = YOLO(str(WEIGHTS / "yolo11_dbscan_setting2_best.pt"))(str(image), verbose=False)[0]
    return str(result.names[int(result.probs.top1)])


def predict_mobilenet(image: Path) -> str:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(WEIGHTS / "mobilenetv2_setting2_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    with Image.open(image) as opened:
        tensor = MOBILENET_TRANSFORM(opened.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        return CLASSES[int(model(tensor).argmax(dim=1).item())]


def predict_lmfcn(image: Path) -> str:
    model = FeatureCNN().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS / "lmfcn_dbscan_setting2_cnn.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    with Image.open(image) as opened:
        array = np.asarray(opened.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        features = model(tensor).numpy()
    classifier = joblib.load(WEIGHTS / "lmfcn_dbscan_setting2_svm.joblib")
    stored = np.load(WEIGHTS / "lmfcn_dbscan_setting2_reference.npz")
    return CLASSES[int(classifier.predict(rbf_kernel(features, stored["features"], float(stored["gamma"])))[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=("yolo", "mobilenet", "lmfcn"), required=True)
    parser.add_argument("--imagem", type=Path, required=True)
    args = parser.parse_args()
    if not args.imagem.is_file():
        raise FileNotFoundError(args.imagem)
    predictors = {"yolo": predict_yolo, "mobilenet": predict_mobilenet, "lmfcn": predict_lmfcn}
    start = time.perf_counter()
    prediction = predictors[args.modelo](args.imagem)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"modelo={args.modelo}; classe={prediction}; tempo_fim_a_fim_ms={elapsed_ms}")


if __name__ == "__main__":
    main()
