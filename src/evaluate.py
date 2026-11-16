"""Evaluate one or more models on the test split and print a Markdown comparison table.

Default compares the baseline (best_kashan.pt, YOLO11n) with the improved model
(best.pt, YOLO11s + road augmentations), each with and without test-time augmentation.
Paste the printed table into the README "Results" section.
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent  # repo root (this file lives in src/)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare YOLO models on the test split.")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=[str(PROJECT_DIR / "best_kashan.pt"), str(PROJECT_DIR / "best.pt")],
        help="One or more .pt files. Default: baseline best_kashan.pt and improved best.pt",
    )
    parser.add_argument("--data", default=str(PROJECT_DIR / "data_2class.yaml"), help="Dataset YAML.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-tta", action="store_true", help="Skip the test-time-augmentation runs.")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")

    rows = []
    for w in args.weights:
        if not Path(w).is_file():
            raise FileNotFoundError(f"Model not found: {w}")
        model = YOLO(w)
        for tta in [False] if args.no_tta else [False, True]:
            m = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=4,
                          device=args.device, workers=0, augment=tta, plots=True, verbose=False,
                          project=str(PROJECT_DIR / "runs"), name=f"eval_{Path(w).stem}{'_tta' if tta else ''}",
                          exist_ok=True)
            per_class = {model.names[c]: m.box.class_result(i) for i, c in enumerate(m.box.ap_class_index)}
            pot = per_class.get("pothole", (float("nan"),) * 4)
            rows.append(f"| {Path(w).name} | {'on' if tta else 'off'} | {m.box.mp:.3f} | {m.box.mr:.3f} | "
                        f"{m.box.map50:.3f} | {m.box.map:.3f} | {pot[1]:.3f} | {pot[2]:.3f} |")

    print(f"\n### {args.split.capitalize()} split @ {args.imgsz}px\n")
    print("| Model | TTA | Precision | Recall | mAP50 | mAP50-95 | Pothole recall | Pothole mAP50 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    print("\n".join(rows))
    print("\nConfusion matrices and PR curves are saved under runs/eval_*/")


if __name__ == "__main__":
    main()
