"""
Garbage Detection using YOLOv8

Two modes:
  1. Fine-tuned model (recommended) — pass --weights to a .pt file trained on TACO.
     Every detected class IS garbage, so no filtering is needed.

  2. Pretrained COCO fallback — no --weights flag. Uses a hardcoded list of
     COCO class IDs that look like litter (bottles, cups, food, etc.).
     Less accurate — treats proxy objects as garbage.

Usage:
    python garbage_detector.py --source 0                          # webcam, COCO fallback
    python garbage_detector.py --source 0 --weights best.pt       # webcam, fine-tuned
    python garbage_detector.py --source image.jpg --save          # image, save output
    python garbage_detector.py --source video.mp4 --weights best.pt --save
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# COCO fallback: class IDs that are proxy-garbage when no fine-tuned model
# is available.  Only used when --weights is NOT supplied.
# --------------------------------------------------------------------------- #
COCO_GARBAGE_CLASSES = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    26: "handbag",
    28: "suitcase",
    67: "cell phone",
    73: "book",
    76: "scissors",
}

GARBAGE_COLOR = (0, 0, 255)
CLEAN_COLOR   = (0, 200, 0)
LABEL_BG      = (20, 20, 20)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Cleanliness scoring
# --------------------------------------------------------------------------- #

def compute_cleanliness(garbage_count: int, frame_area: int, boxes) -> tuple[str, float]:
    """Return (status_label, score_0_to_1).  Score 1.0 = perfectly clean."""
    if garbage_count == 0:
        return "CLEAN", 1.0

    coverage = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    coverage_ratio = min(coverage / frame_area, 1.0)

    count_penalty = min(garbage_count / 10.0, 1.0)
    score = max(0.0, 1.0 - (0.6 * count_penalty + 0.4 * coverage_ratio))

    if score >= 0.7:
        status = "MOSTLY CLEAN"
    elif score >= 0.4:
        status = "DIRTY"
    else:
        status = "VERY DIRTY"

    return status, score


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def draw_garbage_box(frame, x1, y1, x2, y2, label: str, conf: float):
    cv2.rectangle(frame, (x1, y1), (x2, y2), GARBAGE_COLOR, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), GARBAGE_COLOR, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status_overlay(frame, status: str, score: float, count: int, model_tag: str):
    h, w = frame.shape[:2]

    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), LABEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = (
        CLEAN_COLOR if status == "CLEAN" else
        (0, 200, 255) if status == "MOSTLY CLEAN" else
        GARBAGE_COLOR
    )

    tag = f"[{model_tag}]  "
    status_text = f"{tag}{status}  |  Items: {count}  |  Score: {score:.0%}"
    cv2.putText(frame, status_text, (10, 33), FONT, 0.68, color, 2, cv2.LINE_AA)

    bar_y = bar_h + 5
    bar_w = int(w * score)
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + 6), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + 6), color, -1)


# --------------------------------------------------------------------------- #
# Per-frame detection
# --------------------------------------------------------------------------- #

def process_frame(model, frame, conf_threshold: float, class_names: dict | None):
    """
    class_names:
      - None → fine-tuned mode: all detections are garbage, use model's own names
      - dict  → COCO fallback mode: filter to only class IDs in the dict
    """
    h, w = frame.shape[:2]
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    garbage_boxes = []
    garbage_count = 0

    for box in results.boxes:
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_names is None:
            # Fine-tuned model — every class is garbage
            label = results.names[cls_id]
        else:
            # COCO fallback — skip non-garbage classes
            if cls_id not in class_names:
                continue
            label = class_names[cls_id]

        draw_garbage_box(frame, x1, y1, x2, y2, label, conf)
        garbage_boxes.append((x1, y1, x2, y2))
        garbage_count += 1

    status, score = compute_cleanliness(garbage_count, h * w, garbage_boxes)
    return frame, status, score, garbage_count


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def run(source, conf_threshold: float, save: bool, model_size: str, weights_path: str | None):
    if weights_path:
        print(f"Loading fine-tuned model: {weights_path}")
        model = YOLO(weights_path)
        class_names = None          # fine-tuned: all classes are garbage
        model_tag = "FINETUNED"
    else:
        model_name = f"yolov8{model_size}.pt"
        print(f"Loading COCO model: {model_name}  (tip: run train.py for better results)")
        model = YOLO(model_name)
        class_names = COCO_GARBAGE_CLASSES
        model_tag = "COCO-fallback"

    is_webcam  = str(source).isdigit()
    source_val = int(source) if is_webcam else source

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: cannot open source '{source}'")
        return

    writer = None
    if save and not is_webcam:
        src_path = Path(source)
        out_path = src_path.with_stem(src_path.stem + "_detected")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        print(f"Saving output to: {out_path}")

    is_image = not is_webcam and Path(str(source)).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    print("Press 'q' to quit, 's' to save current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, status, score, count = process_frame(model, frame, conf_threshold, class_names)
        draw_status_overlay(processed, status, score, count, model_tag)

        cv2.imshow("Garbage Detector", processed)

        if writer:
            writer.write(processed)

        if is_image:
            if save:
                src_path = Path(source)
                out_path = src_path.with_stem(src_path.stem + "_detected")
                cv2.imwrite(str(out_path), processed)
                print(f"Saved: {out_path}")
            print(f"Result -> {status} | Items: {count} | Score: {score:.0%}")
            cv2.waitKey(0)
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = f"snapshot_{Path(str(source)).stem}.jpg"
            cv2.imwrite(snap_path, processed)
            print(f"Snapshot saved: {snap_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def resolve_weights(weights_arg: str | None) -> str | None:
    """Return weights path to use: explicit arg → best.pt in same folder → None (COCO fallback)."""
    if weights_arg:
        return weights_arg
    default = Path(__file__).parent / "best.pt"
    if default.exists():
        print(f"Found best.pt in project folder — using fine-tuned model.")
        return str(default)
    return None


def main():
    parser = argparse.ArgumentParser(description="Garbage detection with YOLOv8")
    parser.add_argument("--source", default="0",
                        help="Input: 0=webcam, or path to image/video")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--save", action="store_true",
                        help="Save output image or video")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 base model size (only used when --weights is not set)")
    parser.add_argument("--weights",
                        help="Path to fine-tuned .pt file (defaults to best.pt in this folder if it exists)")
    args = parser.parse_args()
    run(args.source, args.conf, args.save, args.model, resolve_weights(args.weights))


if __name__ == "__main__":
    main()
