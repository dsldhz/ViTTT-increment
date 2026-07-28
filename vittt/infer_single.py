import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import get_config
from models import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="ViT^3 single-image inference")
    parser.add_argument("--cfg", required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--image", required=True, help="Input JPG/PNG image")
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional ImageNet class-name file with one label per line",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions")
    parser.add_argument(
        "--weights-key",
        choices=("auto", "model", "state_dict_ema"),
        default="auto",
        help="Checkpoint state-dict key; auto prefers EMA weights when present",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 autocast")
    return parser.parse_args()


def make_config(cfg_path):
    # get_config() expects the training entrypoint's complete argument namespace.
    config_args = argparse.Namespace(
        cfg=cfg_path,
        opts=None,
        batch_size=None,
        data_path=None,
        zip=False,
        cache_mode=None,
        resume=None,
        use_checkpoint=False,
        amp=False,
        output=None,
        tag=None,
        eval=False,
        throughput=False,
    )
    return get_config(config_args)


def select_state_dict(checkpoint, weights_key):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping")

    if weights_key == "auto":
        if "state_dict_ema" in checkpoint:
            weights_key = "state_dict_ema"
        elif "model" in checkpoint:
            weights_key = "model"
        else:
            return checkpoint, "checkpoint root"

    if weights_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no '{weights_key}' key. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )
    return checkpoint[weights_key], weights_key


def strip_module_prefix(state_dict):
    prefix = "module."
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_labels(path, num_classes):
    if path is None:
        return [f"class_{index}" for index in range(num_classes)]

    with Path(path).open(encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if len(labels) != num_classes:
        raise ValueError(
            f"Expected {num_classes} labels, but found {len(labels)} in {path}"
        )
    return labels


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    config = make_config(args.cfg)
    model = build_model(config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict, selected_key = select_state_dict(checkpoint, args.weights_key)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict

    model.eval().to(device)

    resize_size = int((256 / 224) * config.DATA.IMG_SIZE)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    with Image.open(args.image) as image:
        image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(image_tensor)
            probabilities = logits.float().softmax(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000

    labels = load_labels(args.classes, probabilities.shape[1])
    topk = min(args.topk, probabilities.shape[1])
    scores, indices = probabilities[0].topk(topk)

    print(f"image: {args.image}")
    print(f"weights: {args.checkpoint} ({selected_key})")
    print(f"device: {device}")
    print(f"latency: {latency_ms:.2f} ms")
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), 1):
        print(f"{rank}: {labels[index]} (class_id={index}, probability={score:.6f})")


if __name__ == "__main__":
    main()
