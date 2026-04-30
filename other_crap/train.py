"""
Fine-tune YOLOv8 on the TACO garbage dataset.

Prerequisites:
    python download_dataset.py   # run this first

Usage:
    python train.py                          # defaults: nano model, 50 epochs
    python train.py --model s --epochs 80    # small model, more epochs
    python train.py --resume                 # resume interrupted training

After training, best weights are saved to:
    runs/detect/garbage_finetune/weights/best.pt

Pass that path to garbage_detector.py:
    python garbage_detector.py --source 0 --weights runs/detect/garbage_finetune/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


DEFAULT_DATASET = Path("dataset/data.yaml")
CUSTOM_DATASET = Path("custom_data/data.yaml")
DEFAULT_MODEL = "n"   # nano — fastest; use "s" or "m" for better accuracy
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16    # lower to 8 if you hit OOM on GPU / CPU is too slow
RUN_NAME = "garbage_finetune"


def train(model_size: str, epochs: int, batch: int, imgsz: int, resume: bool, device: str, data_yaml: Path):
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found.")
        print("If using custom data, place images in custom_data/images/train and custom_data/images/val")
        print("and ensure custom_data/data.yaml exists.")
        return

    if resume:
        # Look for the last checkpoint from a previous run
        last_ckpt = Path(f"runs/detect/{RUN_NAME}/weights/last.pt")
        if not last_ckpt.exists():
            print(f"No checkpoint found at {last_ckpt}. Starting fresh.")
            resume = False

    if resume:
        print(f"Resuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        base_weights = f"yolo26{model_size}.pt"
        print(f"Starting fine-tune: base={base_weights}, epochs={epochs}, imgsz={imgsz}, batch={batch}")
        model = YOLO(base_weights)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=RUN_NAME,
            device=device,

            # Augmentation — important for garbage which appears in varied conditions
            hsv_h=0.02,       # slight hue jitter (lighting changes)
            hsv_s=0.6,        # saturation jitter
            hsv_v=0.4,        # brightness jitter
            fliplr=0.5,       # horizontal flip
            flipud=0.1,       # vertical flip (garbage on ground, could be upside down)
            mosaic=1.0,       # mosaic augmentation (mixes 4 images, great for small objects)
            scale=0.5,        # random scaling
            translate=0.1,    # random translation
            degrees=10.0,     # slight rotation

            # Training settings
            patience=20,      # stop early if no improvement for 20 epochs
            save_period=10,   # save checkpoint every 10 epochs
            plots=True,       # save training plots (loss curves, PR curve, etc.)
            val=True,
        )

    best_weights = Path(f"runs/detect/{RUN_NAME}/weights/best.pt")
    if best_weights.exists():
        print(f"\nTraining complete!")
        print(f"Best weights saved to: {best_weights}")
        print(f"\nTo run the detector with your fine-tuned model:")
        print(f"  python garbage_detector.py --source 0 --weights {best_weights}")
    else:
        print("\nTraining finished but best.pt not found. Check runs/detect/ directory.")


def validate(weights_path: str, data_yaml: Path = DEFAULT_DATASET):
    """Evaluate fine-tuned model on the validation set and print mAP metrics."""
    model = YOLO(weights_path)
    results = model.val(data=str(data_yaml), imgsz=DEFAULT_IMGSZ)
    print("\n=== Validation Results ===")
    print(f"mAP50:     {results.box.map50:.3f}")
    print(f"mAP50-95:  {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall:    {results.box.mr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on TACO garbage dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (n=nano fastest, x=xlarge most accurate)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="Batch size — lower if you run out of memory")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="Training image size (larger = better accuracy, slower)")
    parser.add_argument("--device", default="",
                        help="Device: '' for auto-detect, '0' for GPU 0, 'cpu' to force CPU")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--data", default=None,
                        help="Path to data.yaml. Defaults to custom_data/data.yaml if it exists, else dataset/data.yaml")
    parser.add_argument("--validate", metavar="WEIGHTS",
                        help="Skip training, just evaluate a .pt file on the val set")
    args = parser.parse_args()

    # Auto-pick dataset: custom first, then fallback to TACO
    if args.data:
        data_yaml = Path(args.data)
    elif CUSTOM_DATASET.exists():
        data_yaml = CUSTOM_DATASET
        RUN_NAME = "messy_wires_finetune"
    else:
        data_yaml = DEFAULT_DATASET

    if args.validate:
        validate(args.validate, data_yaml)
    else:
        train(args.model, args.epochs, args.batch, args.imgsz, args.resume, args.device, data_yaml)


if __name__ == "__main__":
    main()
