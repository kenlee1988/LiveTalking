"""Benchmark Wav2Lip inference on CPU, CUDA, XPU, or MPS."""

import argparse
import statistics
import time

import torch

from avatars.wav2lip.models import Wav2Lip
from utils.device import initialize_device, synchronize_device


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, location: storage,
    )
    model = Wav2Lip()
    state_dict = {
        name.replace("module.", ""): value
        for name, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu", "mps"), default="auto")
    parser.add_argument("--checkpoint", default="models/wav2lip.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    selected_device = initialize_device(args.device)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {selected_device}")

    model = load_model(args.checkpoint, selected_device)
    image = torch.ones(args.batch_size, 6, 256, 256, device=selected_device)
    mel = torch.ones(args.batch_size, 1, 80, 16, device=selected_device)

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model(mel, image)
        synchronize_device(selected_device)

        timings = []
        for _ in range(args.runs):
            started_at = time.perf_counter()
            model(mel, image)
            synchronize_device(selected_device)
            timings.append(time.perf_counter() - started_at)

    median_seconds = statistics.median(timings)
    frames_per_second = args.batch_size / median_seconds
    print(f"Median batch latency: {median_seconds * 1000:.1f} ms")
    print(f"Throughput: {frames_per_second:.2f} frames/s")


if __name__ == "__main__":
    main()
