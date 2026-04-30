"""
Garbage Detector — detects garbage in images, videos, or webcam.

Usage:
    python garbage_detector.py                  # webcam
    python garbage_detector.py photo.jpg        # image
    python garbage_detector.py video.mp4        # video
    Press 'q' to quit.

================================================================================
READING ORDER (follow this so you don't get confused):
================================================================================
1.  IMPORTS & CONFIG    (lines ~21–35)
    → What libraries we bring in and what settings we set.

2.  detect()            (lines ~140–145)
    → THE HEART: feeds an image into the AI and gets back what it found.
    This is where YOLO (the CNN) runs.

3.  cleanliness()       (lines ~100–125)
    → THE BRAIN: takes the list of found trash items and computes a
    cleanliness score (0% = filthy, 100% = spotless).

4.  annotate()          (lines ~128–185)
    → THE ARTIST: draws red boxes, labels, and the cleanliness banner
    on the image using cv2.

5.  run_image()         (lines ~188–200)
    → Handles a single photo: load → detect → annotate → show.

6.  run_video()         (lines ~203–220)
    → Handles webcam / video: read frames in a loop, detect & annotate
    each frame, show live. Press 'q' to quit.

7.  __main__ block      (lines ~223–235)
    → Entry point: decides if the user gave us an image file or a
    video source, then calls run_image() or run_video().
================================================================================
"""

import sys  # Lets us read command-line arguments (e.g. python file.py image.jpg)
from pathlib import Path  # Easy, modern way to work with file paths

import cv2  # OpenCV: reads images/video, draws boxes/text, shows windows
import numpy as np  # NumPy: used here for the dummy warmup image (see bottom of section 1).
import torch  # PyTorch: Ultralytics is built on top of this. We use it to pick the best device.
from ultralytics import YOLO  # YOLOv8: the object-detection CNN (neural network)

# =============================================================================
# 1. IMPORTS & CONFIG
# =============================================================================
# WEIGHTS: path to the fine-tuned model file (best.pt) that was created by
# train.py. This model was trained specifically to recognise garbage types.
# If it doesn't exist, we fall back to the generic pre-trained model.
WEIGHTS = Path(__file__).parent / "best.pt"

# MODEL: this line loads the actual neural network into memory.
#   - If best.pt exists → we load our custom garbage-trained model.
#   - Else → we load yolov8n.pt, the generic "nano" YOLO model that knows
#     80 everyday objects (person, car, bottle, etc.) but was NOT trained
#     on garbage specifically.
MODEL = YOLO(str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")

# CONF (confidence threshold): YOLO gives every prediction a confidence
# score (0.0 to 1.0). Only predictions ABOVE 0.35 are kept.
#   Example: if YOLO says "this looks like a bottle with 0.91 confidence",
#   we keep it. If it says "0.12 confidence", we ignore it.
CONF = 0.35

# =============================================================================
# 1.5  DEVICE SELECTION + WARMUP   (added for performance)
# =============================================================================
# By default Ultralytics will run on whatever PyTorch picks, which on a Mac is
# the CPU. CPU inference of YOLOv8 is slow (~100–200 ms per frame). We can do
# much better by moving the model onto a GPU if there is one:
#
#     CUDA  →  NVIDIA GPUs (Linux / Windows machines)
#     MPS   →  Apple Silicon GPUs (M1 / M2 / M3 / M4 Macs) via Metal
#     CPU   →  fallback if neither of the above is available
#
# Speedup is typically 3–5× per frame on MPS vs CPU. The code below picks
# whichever device is best on YOUR machine, automatically.
# =============================================================================
if torch.cuda.is_available():
    DEVICE = "cuda"          # NVIDIA GPU available
elif torch.backends.mps.is_available():
    DEVICE = "mps"           # Apple Silicon GPU available
else:
    DEVICE = "cpu"           # plain CPU fallback

# Move every weight tensor in the model onto the chosen device. After this,
# every call to MODEL(...) will run inference on that device automatically.
MODEL.to(DEVICE)
print(f"[garbage_detector] model loaded on device: {DEVICE}")

# -----------------------------------------------------------------------------
# WARMUP — pay the "first inference" cost NOW, not on the user's first frame.
# -----------------------------------------------------------------------------
# The very first call to MODEL(...) is much slower than every call after it,
# because PyTorch has to:
#   • JIT-compile its compute kernels for the exact input shape we use,
#   • ask the GPU backend (MPS / CUDA) to allocate buffers and build its
#     internal shader / kernel graphs,
#   • run a bunch of one-time Ultralytics setup that's deferred until first use.
#
# That cost is unavoidable. The trick is to PAY IT AT SERVER STARTUP instead
# of when the user clicks "Start camera" and waits for the first frame.
#
# So we feed the model one fake, all-black 640×640 image. We don't care about
# the result — we just want the side effects (caches warmed, kernels compiled).
# -----------------------------------------------------------------------------
_dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)  # plain black image, BGR
MODEL(_dummy_frame, conf=CONF, verbose=False)            # one inference; result thrown away
print("[garbage_detector] model warmed up — first real frame will be fast")


# COCO_CLUTTER: when we fall back to the generic model (yolov8n.pt),
# we show classes that look like garbage OR everyday clutter / messy-room items.
#
# Original garbage items:
#   39 = bottle, 41 = cup, 45 = bowl, 46 = banana, 47 = apple,
#   49 = orange, 54 = pizza, 67 = cell phone.
#
# Added daily clutter / floor-items:
#   24 = backpack, 25 = umbrella, 26 = handbag, 27 = tie,
#   28 = suitcase, 32 = sports ball, 37 = surfboard, 38 = tennis racket,
#   40 = wine glass, 43 = knife, 44 = spoon, 48 = sandwich,
#   50 = broccoli, 51 = carrot, 52 = hot dog, 53 = donut,
#   55 = cake, 56 = chair, 57 = couch, 58 = potted plant,
#   59 = bed, 60 = dining table, 62 = tv, 63 = laptop,
#   64 = mouse, 65 = remote, 66 = keyboard, 73 = book,
#   74 = clock, 75 = vase, 76 = scissors, 77 = teddy bear,
#   78 = hair drier, 79 = toothbrush.
COCO_CLUTTER = {
    # original garbage
    39, 41, 45, 46, 47, 49, 54, 67,
    # bags / accessories
    24, 25, 26, 27, 28,
    # sports / hobby
    32, 37, 38,
    # kitchen / food clutter
    40, 43, 44, 48, 50, 51, 52, 53, 55,
    # furniture
    56, 57, 58, 59, 60,
    # electronics / desk clutter
    62, 63, 64, 65, 66, 73, 74,
    # misc household
    75, 76, 77, 78, 79,
}


# =============================================================================
# 3. cleanliness()  —  compute how dirty the scene is
# =============================================================================
# Inputs:
#   n_items  → how many garbage objects were detected
#   boxes    → list of (x1, y1, x2, y2) pixel coordinates for each object
#   h, w     → height and width of the image (in pixels)
#
# Output:
#   Returns a (status_string, score) tuple.
#   score = 1.0 means perfectly clean, 0.0 means extremely dirty.
#
# How the formula works (in plain English):
#   There are TWO things that make a scene look dirty:
#     A) How MANY pieces of trash there are.
#     B) How much AREA of the image they cover.--- there is also something else that matters what if i use this to say " only if the image is a certain percentage of the screen aka i get close enough i can surely say that this place has garbage "
#   We combine them: 60% weight on count, 40% weight on area coverage.
#   Then we subtract that from 1.0 so that MORE trash = LOWER score.
# =============================================================================
def cleanliness(n_items, boxes, h, w):
    """Score from 0.0 (filthy) to 1.0 (clean)."""

    # If zero items detected → scene is perfectly clean, short-circuit.
    if not n_items:
        return "CLEAN", 1.0

    # coverage = total area of all trash boxes ÷ total image area.
    # Each box area = width × height = (x2 - x1) × (y2 - y1).
    # We sum this for every detected box, then divide by the image size.
    # Result: a number between 0.0 (no trash pixels) and 1.0 (entire image is trash).
    coverage = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes) / (h * w)

    # Step 1 — count factor:
    #   n_items / 10  → normalise by 10 (10+ items = max factor of 1.0).
    #   min(..., 1)   → cap it at 1.0 so 50 items don't explode the score.
    #   × 0.6         → 60% weight.
    #
    # Step 2 — area factor:
    #   coverage      → already 0.0 to 1.0.
    #   min(..., 1)   → safety cap.
    #   × 0.4         → 40% weight.
    #
    # Step 3 — combine and invert:
    #   1.0 - (step1 + step2)  → more trash = lower final score.
    #   max(0.0, ...)          → never go below 0 (sanity clamp).
    score = max(
        0.0,
        1.0
        - (
            0.6 * min(n_items / 10, 1)  # count contribution (60%)
            + 0.4 * min(coverage, 1)  # area contribution (40%)
        ),
    )

    # Convert the numeric score into a human-readable label.
    #   >= 0.7  → MOSTLY CLEAN
    #   >= 0.4  → DIRTY
    #   <  0.4  → VERY DIRTY
    label = (
        "MOSTLY CLEAN" if score >= 0.7 else "DIRTY" if score >= 0.4 else "VERY DIRTY"
    )

    return label, score


# =============================================================================
# 4. annotate()  —  draw bounding boxes, labels, and cleanliness banner
# =============================================================================
# This function takes the RAW image (frame) and the YOLO detection results.
# It modifies the image IN-PLACE by drawing:
#   • A red rectangle around every detected garbage object.
#   • White text above each rectangle saying the class name + confidence.
#   • A big banner at the top-left saying e.g. "DIRTY | 35%".
#
# IMPORTANT: cv2 (OpenCV) does NOT do AI. It only draws. YOLO already
# found the objects; cv2 just paints the picture so YOU can see them.
# =============================================================================
def annotate(frame, results):
    """Draw bounding boxes and cleanliness banner on frame."""

    boxes = []  # We will collect every drawn box here so that
    # cleanliness() can compute the area coverage later.
    finetuned = WEIGHTS.exists()  # True if we loaded best.pt (garbage-trained).

    # -------------------------------------------------------------------------
    # Loop through every detection that YOLO returned.
    # results.boxes is a list-like object; each element is one detected object.
    # -------------------------------------------------------------------------
    for box in results.boxes:
        cls = int(box.cls[0])  # class ID   (e.g. 0 = "plastic bag")
        conf = float(box.conf[0])  # confidence (e.g. 0.87 = 87% sure)

        # box.xyxy[0] gives the bounding box in pixel coordinates as
        # [x1, y1, x2, y2].  map(int, ...) converts them to whole numbers
        # because pixels can't be decimals.
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Look up the human-readable name for this class ID.
        label = results.names[cls]  # e.g. "trash", "bottle", "wrapper", ...

        # -----------------------------------------------------------------
        # FILTER: if we are using the GENERIC model (not our fine-tuned one),
        # we ONLY draw boxes for classes that look like garbage.
        # See COCO_CLUTTER above. If cls is not in that set, we skip it.
        # -----------------------------------------------------------------
        if not finetuned and cls not in COCO_CLUTTER:
            continue

        # ================================================================
        # cv2.rectangle  —  draws the red box
        # ================================================================
        # frame        → the image (NumPy array) we are drawing on
        # (x1, y1)     → top-left corner of the box (pixels)
        # (x2, y2)     → bottom-right corner of the box (pixels)
        # (0, 0, 255)  → colour in BGR format = pure RED
        # 2            → line thickness in pixels
        # ================================================================
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ================================================================
        # cv2.putText  —  writes the label above the box
        # ================================================================
        # frame                    → the image canvas
        # f"{label} {conf:.0%}"    → text string, e.g. "Plastic bag 87%"
        # (x1, y1 - 8)             → position: 8 pixels ABOVE the box top-left
        # cv2.FONT_HERSHEY_SIMPLEX → a clean, readable font
        # 0.55                     → font scale (size multiplier)
        # (255, 255, 255)          → colour in BGR = pure WHITE
        # 1                        → text thickness in pixels
        # ================================================================
        cv2.putText(
            frame,
            f"{label} {conf:.0%}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        # Remember this box so cleanliness() can use it later.
        boxes.append((x1, y1, x2, y2))

    # -------------------------------------------------------------------------
    # Now draw the cleanliness banner at the top-left of the image.
    # -------------------------------------------------------------------------
    h, w = frame.shape[:2]  # get image height & width
    status, score = cleanliness(len(boxes), boxes, h, w)

    # Pick banner colour based on score:
    #   >= 0.7  → green  (0, 200, 0)      = MOSTLY CLEAN
    #   >= 0.4  → orange (0, 200, 255)    = DIRTY
    #   <  0.4  → red    (0, 0, 255)      = VERY DIRTY
    color = (
        (0, 200, 0) if score >= 0.7 else (0, 200, 255) if score >= 0.4 else (0, 0, 255)
    )

    # Draw the banner text, e.g. "DIRTY | 35%" or "CLEAN | 100%"
    cv2.putText(
        frame,
        f"{status} | {score:.0%}",
        (10, 35),  # 10px from left, 35px from top
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,  # larger font scale for visibility
        color,
        2,  # thicker line for banner
    )

    return frame


# =============================================================================
# 2. detect()  —  run the CNN on an image
# =============================================================================
# This is the bridge between OpenCV and YOLO.
#   Input:  a single image (frame), which is a NumPy array from OpenCV.
#   Output: the first result object containing all detected boxes, classes,
#           and confidence scores.
#
# What's happening inside (one sentence):
#   MODEL(...)  → YOLO takes the image, runs it through its convolutional
#                 layers, outputs bounding boxes + class predictions.
#   conf=CONF   → filter out low-confidence predictions (anything < 0.35).
#   verbose=False → don't print timing/progress spam to the terminal.
#   [0]         → YOLO returns a list of results (one per image in a batch).
#                 Since we only passed ONE image, we grab the first result.
# =============================================================================
def detect(source):
    """Run YOLO on the source and return the first result."""
    return MODEL(source, conf=CONF, verbose=False)[0]


# =============================================================================
# 5. run_image()  —  process a single photo
# =============================================================================
# Steps:
#   1. cv2.imread(path)   → load the image from disk into a NumPy array.
#   2. detect(frame)      → run YOLO on it, get detection results.
#   3. annotate(...)      → draw boxes + banner on the image.
#   4. cv2.imshow(...)    → open a window showing the annotated image.
#   5. cv2.waitKey(0)     → freeze the window until ANY key is pressed.
#   6. cv2.destroyAllWindows() → close the window.
# =============================================================================
def run_image(path):
    """Detect on a single image, show it, wait for keypress."""
    frame = cv2.imread(path)  # 1. read image
    if frame is None:  # safety check: file might be missing/corrupt
        return print(f"ERROR: can't read {path}")

    # 2. detect → 3. annotate → 4. show
    # The inner call order is RIGHT to LEFT:
    #   detect(frame) runs first, then annotate receives its output.
    cv2.imshow("Garbage Detector", annotate(frame, detect(frame)))

    cv2.waitKey(0)  # 5. wait for any key press
    cv2.destroyAllWindows()  # 6. close the window


# =============================================================================
# 6. run_video()  —  process webcam or video file, frame by frame
# =============================================================================
# This is a LOOP that runs until the video ends or the user presses 'q'.
#
# Steps PER FRAME (~30 times per second for webcam):
#   1. cap.read()         → grab the next frame from webcam / video file.
#   2. detect(frame)      → run YOLO on this frame.
#   3. annotate(...)      → draw boxes + banner.
#   4. cv2.imshow(...)    → display the annotated frame.
#   5. cv2.waitKey(1)     → wait 1 millisecond, also checks for keypress.
#      If the pressed key is 'q', break the loop.
#
# After loop ends:
#   cap.release()         → free the webcam / close the video file.
#   cv2.destroyAllWindows() → close all OpenCV windows.
# =============================================================================
def run_video(source):
    """Detect on video/webcam frame-by-frame. Press 'q' to quit."""
    cap = cv2.VideoCapture(source)  # open webcam (0) or video file path

    while cap.isOpened():  # keep looping while the stream is alive
        ok, frame = cap.read()  # 1. read next frame
        if not ok:  # if no frame returned (end of video / error)
            break

        # 2. detect → 3. annotate → 4. show
        cv2.imshow("Garbage Detector", annotate(frame, detect(frame)))

        # 5. wait 1 ms for a key; bitwise-AND with 0xFF keeps only the low byte.
        # If the key is 'q', we exit the loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # release the camera / file handle
    cv2.destroyAllWindows()  # close the display window(s)


# =============================================================================
# 7. MAIN ENTRY POINT  —  decide what the user wants
# =============================================================================
# When you run `python garbage_detector.py`, Python executes this block.
# When you import the file (e.g. `import garbage_detector`), this block
# does NOT run.
#
# Logic:
#   Read sys.argv[1] (the first argument after the script name).
#   If nothing is given → default to "0" (webcam).
#   If the argument ends with .jpg / .jpeg / .png / .bmp / .webp → it's an image.
#       → call run_image(argument)
#   Otherwise → it's a video file path or a camera index (e.g. 0, 1).
#       → call run_video(...)
#
# int(source) if source.isdigit() else source:
#   If the user typed "0" or "1", convert to integer (OpenCV expects int
#   for camera indices). Otherwise leave it as a string path.
# =============================================================================
if __name__ == "__main__":
    # sys.argv is a list: ['garbage_detector.py', 'optional_argument']
    source = sys.argv[1] if len(sys.argv) > 1 else "0"

    # Check the file extension to decide image vs video
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image(source)  # single photo path
    else:
        run_video(int(source) if source.isdigit() else source)  # webcam or video
