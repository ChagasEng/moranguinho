"""Carregamento sob demanda das imagens, sem duplicar o conjunto na RAM."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image(path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class ImageDataset(Dataset):
    def __init__(self, rows: list, size: int):
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        label, path = self.rows[index]
        return load_image(path, self.size), label, index
