"""
video_processor.py — Custom WebRTC video track that runs the GARBAGE detector
on each frame, by reusing the existing garbage_detector.py module.

WHAT THIS FILE DOES
-------------------
We define a class `YOLOVideoTrack` that behaves like a normal WebRTC video
track from aiortc's point of view, but secretly pipelines every frame
through the garbage-detection logic that already lives in
`garbage_detector.py` (one directory up from this file).

We do NOT redefine YOLO loading, drawing, or cleanliness scoring here.
Those already exist in garbage_detector.py and we import and call them.

KEY CONCEPTS
------------
- In aiortc, a "video track" is any class that inherits from MediaStreamTrack,
  declares `kind = "video"`, and implements an async `recv()` method. The
  library calls `recv()` repeatedly (roughly at the camera's frame rate).
  Each call must return an `av.VideoFrame` object.

- `av.VideoFrame` is the data type used by PyAV (the FFmpeg bindings aiortc
  is built on). It holds raw pixel data plus two timing fields:
        pts        = presentation timestamp (when this frame should be shown)
        time_base  = the unit of pts (e.g. 1/90000 sec for video)
  These two together let the receiving side play frames at the correct
  speed. We MUST forward them from the input frame to the output frame,
  otherwise the browser will play things at the wrong rate or stutter.

- To run YOLO we need pixels as a NumPy array (height x width x 3 in BGR
  order — the format OpenCV uses, which is exactly what garbage_detector.py
  already expects). PyAV gives us that with `frame.to_ndarray(format="bgr24")`.
  After garbage_detector annotates it we go the other way with
  `av.VideoFrame.from_ndarray(arr, format="bgr24")`.
"""

# --- Standard library ------------------------------------------------------
import asyncio  # Lets us run blocking inference in a worker thread so the
                # asyncio event loop stays free to handle network I/O.
import os    # For path manipulation when adding the parent dir to sys.path.
import sys   # We append the parent directory to sys.path so we can import
             # garbage_detector.py, which lives one level up from this file.

# --- Third-party -----------------------------------------------------------
# `av` is PyAV, the Python bindings around FFmpeg's libav* libraries.
# aiortc gives us frames as av.VideoFrame and expects them back the same way.
import av

# MediaStreamTrack is the abstract base class for any media track in aiortc.
# We subclass it to make our custom processor.
from aiortc import MediaStreamTrack

# --- Make the parent directory importable ----------------------------------
# garbage_detector.py lives at /Users/.../garbage-yolo/garbage_detector.py,
# while THIS file is in /Users/.../garbage-yolo/project/. Python won't find
# garbage_detector by default because only the script's own directory is on
# sys.path. We compute the absolute path of the parent directory and prepend
# it to sys.path so `import garbage_detector` works.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))           # .../garbage-yolo/project
PARENT_DIR  = os.path.dirname(PROJECT_DIR)                          # .../garbage-yolo
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# --- Import the existing detector ------------------------------------------
# garbage_detector.py already does, at module import time:
#   • loads best.pt (your fine-tuned garbage-trained YOLOv8 weights), or
#     falls back to the generic yolov8n.pt if best.pt isn't there,
#   • exposes detect(frame)   -> YOLO results object, with conf threshold
#     applied,
#   • exposes annotate(frame, results) -> draws red boxes, white labels,
#     and the colored "CLEAN | 100%", "DIRTY | 35%", etc. cleanliness banner
#     in-place on the frame,
#   • exposes cleanliness(...) used internally by annotate().
#
# Importing the module triggers MODEL = YOLO(...) inside garbage_detector.py,
# so the heavy weight-loading happens exactly ONCE (when this server boots),
# not per frame.
from garbage_detector import detect, annotate


# ---------------------------------------------------------------------------
# Per-frame work, packaged as a plain (synchronous) function.
# ---------------------------------------------------------------------------
# We pull the slow part out of the async `recv()` method so we can hand it to
# a worker thread. The thread does the heavy lifting (YOLO inference + drawing)
# while the asyncio event loop is left free to receive new frames over the
# WebRTC connection and ship annotated frames back. Without this split, the
# event loop would freeze for ~50–150 ms per frame and the stream would
# stutter / fall behind.
# ---------------------------------------------------------------------------
def _process_frame(image):
    """Run YOLO + draw boxes on one image. Pure-sync — safe to run in a thread."""
    results = detect(image)            # 1) YOLO inference (the slow part).
    return annotate(image, results)    # 2) OpenCV drawing (fast, but bundled
                                       #    here so we make ONE thread hop per
                                       #    frame instead of two).


class YOLOVideoTrack(MediaStreamTrack):
    """
    A video track that wraps another video track. Each frame coming out of
    the wrapped (source) track is run through the GARBAGE detector and the
    annotated frame is what consumers of THIS track see.

    aiortc identifies tracks by their `kind` attribute. For a video track
    this MUST be the string "video" (other valid values are "audio" and
    "application").
    """

    # Tell aiortc this is a video track.
    kind = "video"

    def __init__(self, source_track):
        """
        source_track: the original MediaStreamTrack (the browser's webcam,
                      handed to us by aiortc when the browser's offer arrives).
        """
        # MediaStreamTrack.__init__ wires up some internal state (track id,
        # event emitter, etc.). We must call it before doing anything else.
        super().__init__()

        # Save a reference to the upstream track. We'll pull frames from it
        # inside our recv() method below.
        self.source_track = source_track

    async def recv(self):
        """
        Called by aiortc whenever the next outgoing frame is needed.
        Must return an av.VideoFrame.

        Pipeline for each call:
          1. Await one frame from the upstream track (the browser's webcam).
          2. Convert it to a NumPy BGR image.
          3. Hand it to garbage_detector.detect()   -> YOLO results.
          4. Hand both to garbage_detector.annotate() -> draws red boxes,
             labels, and the cleanliness banner directly on the image.
          5. Build a new VideoFrame from the annotated array.
          6. Copy the timing fields (pts/time_base) so playback speed is
             preserved on the receiving side.
          7. Return the new frame.
        """
        # 1) Grab the next frame from the source. `recv()` is async because
        #    it may have to wait for the next frame to actually arrive over
        #    the network from the browser.
        frame = await self.source_track.recv()

        # 2) Convert the av.VideoFrame to a NumPy ndarray in BGR pixel order.
        #    Shape is (height, width, 3), dtype uint8 — the standard OpenCV
        #    image format. "bgr24" means 8 bits per channel, B-G-R order,
        #    which is exactly what garbage_detector.py works with internally.
        image = frame.to_ndarray(format="bgr24")

        # 3+4) Run YOLO + drawing in a WORKER THREAD instead of inline.
        #
        #      Why: detect() is blocking and takes 50–150 ms per frame. If we
        #      called it directly here, those milliseconds would freeze the
        #      whole asyncio event loop — meaning aiortc could not send or
        #      receive ANY other network packets during that time. The result
        #      was the slow / stuttery stream you were seeing.
        #
        #      `loop.run_in_executor(None, fn, *args)` does this:
        #        • `None`              → use Python's default thread pool.
        #        • `fn`                → the sync function to run there.
        #        • `*args`             → positional args passed to fn.
        #        • returns an awaitable → we `await` it; while we wait, the
        #                                 event loop is free to do other work
        #                                 (receive the NEXT camera frame, ship
        #                                 the previous annotated frame, etc.).
        loop = asyncio.get_running_loop()
        annotated = await loop.run_in_executor(None, _process_frame, image)

        # 5) Wrap the annotated NumPy array back into an av.VideoFrame.
        new_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")

        # 6) Carry over the timing metadata from the input frame. Without
        #    these, the receiving browser doesn't know when to display each
        #    frame and you get frozen / fast-forward / stutter playback.
        #
        #    pts        = "presentation timestamp": when this frame should
        #                 be shown, expressed as an integer count of
        #                 time_base units.
        #    time_base  = the duration of one pts unit (a Fraction like 1/90000).
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        # 7) Hand the new frame back to aiortc — it will encode it (with VP8
        #    or H.264) and ship it down the WebRTC connection to the browser.
        return new_frame
