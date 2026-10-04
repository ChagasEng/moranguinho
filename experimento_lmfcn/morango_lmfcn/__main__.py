"""CLI para preparar dados, treinar e prever uma imagem."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento modular CNN + SVM para morangos")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="gera train.txt, val.txt e test.txt")
    prep.add_argument("--dataset", type=Path, default=Path("../Dataset"))
    prep.add_argument("--output", type=Path, default=Path("saida/manifests"))
    prep.add_argument("--seed", type=int, default=42)

    training = commands.add_parser("train", help="treina CNN + SVM e avalia no teste")
    training.add_argument("--manifests", type=Path, default=Path("saida/manifests"))
    training.add_argument("--output", type=Path, default=Path("saida/model"))
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--image-size", type=int, default=128)
    training.add_argument("--features", type=int, default=16)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--svm-c", type=float, default=1.0)
    training.add_argument("--margin", type=float, default=1.0)
    training.add_argument("--opposite-weight", type=float, default=0.5)
    training.add_argument("--seed", type=int, default=42)

    prediction = commands.add_parser("predict", help="classifica uma imagem com o modelo salvo")
    prediction.add_argument("image", type=Path)
    prediction.add_argument("--model", type=Path, default=Path("saida/model"))
    args = parser.parse_args()

    if args.command == "prepare":
        from .manifest import prepare
        counts = prepare(args.dataset, args.output, args.seed)
        for name, count in counts.items():
            print(f"{name}: fresco={count[0]} podre={count[1]}")
    elif args.command == "train":
        from .training import TrainConfig, train
        config = TrainConfig(image_size=args.image_size, features=args.features,
                             batch_size=args.batch_size, epochs=args.epochs,
                             learning_rate=args.learning_rate, svm_c=args.svm_c,
                             margin=args.margin, opposite_weight=args.opposite_weight, seed=args.seed)
        train(args.manifests, args.output, config)
    else:
        import joblib
        import numpy as np
        import torch
        from .data import load_image
        from .model import FeatureCNN
        from .svm import predict_svm
        model_dir = args.model
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        config = metrics["config"]
        model = FeatureCNN(config["features"])
        model.load_state_dict(torch.load(model_dir / "cnn.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            features = model(load_image(args.image, config["image_size"]).unsqueeze(0)).numpy()
        classifier = joblib.load(model_dir / "svm.joblib")
        with np.load(model_dir / "reference.npz") as reference:
            label = int(predict_svm(classifier, features, reference["features"], float(reference["gamma"]))[0])
        print("fresco" if label == 0 else "podre")


if __name__ == "__main__":
    main()
