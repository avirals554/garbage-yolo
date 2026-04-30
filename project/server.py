"""
server.py — Python backend for our WebRTC + YOLO demo.

WHAT THIS FILE DOES (high level)
--------------------------------
1. Runs a tiny web server (aiohttp) that hands the browser our index.html page.
2. Acts as a WebRTC "peer" — meaning it can send and receive live audio/video
   directly to/from the browser over a peer-to-peer connection.
3. When the browser sends us its webcam video, we run YOLOv8 object detection
   on every frame, draw boxes on the detections, and stream the annotated
   video back to the browser in real time.

KEY CONCEPTS (read this once, the rest of the file will make sense)
-------------------------------------------------------------------
- WebRTC: a browser standard for real-time peer-to-peer media (video, audio,
  generic data). Once the two peers have negotiated, frames flow directly
  between them with very low latency — no per-frame HTTP request/response.

- Peer: one of the two endpoints in a WebRTC call. Here the two peers are:
      Peer A = the browser (Chrome/Safari/Firefox/etc.)
      Peer B = this Python process (using the `aiortc` library)
  Neither peer is "the server" in the WebRTC sense — we just happen to also
  be running an aiohttp web server on the side for delivering index.html
  and exchanging the SDP messages.

- RTCPeerConnection: the object that represents the connection on each side.
  Both peers create one. They configure tracks (video/audio) on it, then
  exchange a couple of small text messages (SDP, see below) to agree on
  codecs, IPs, ports, encryption keys, etc.

- SDP (Session Description Protocol): a plain-text format describing
  "here's what I can send/receive, and here are my candidate network
  addresses." Each peer creates an SDP "offer" or "answer" and sends it to
  the other. Despite being for real-time media, the SDP itself is exchanged
  through ANY channel you like — this is called "signaling." We use a
  normal HTTP POST to /offer for signaling.

- ICE (Interactive Connectivity Establishment): the procedure each peer
  uses to figure out how it can be reached over the network — local IPs,
  public IPs (via STUN), relays (via TURN), etc. Each possible address is
  called an "ICE candidate." For a localhost demo we don't even need
  STUN/TURN — the OS provides a 127.0.0.1 candidate that works fine.

- MediaStreamTrack: one stream of media (one camera, one microphone, one
  processed video, ...). A peer connection can carry multiple tracks.

OVERALL FLOW for this demo
--------------------------
    [browser] getUserMedia()  ──► local webcam track
    [browser] new RTCPeerConnection, addTrack(local webcam)
    [browser] createOffer() → SDP text → POST /offer to this server
    [server]  new RTCPeerConnection, on track received → wrap with YOLO,
              add the YOLO-annotated track back to the connection
    [server]  createAnswer() → SDP text → return it as the HTTP response
    [browser] setRemoteDescription(answer) → ICE finishes → media flows
    [browser] ontrack fires with the YOLO-annotated track → <video> shows it
"""

# --- Standard library imports ----------------------------------------------
import asyncio   # Python's built-in async runtime; aiohttp/aiortc are async libs.
import json      # Unused here directly, but useful when debugging SDP payloads.
import os        # For reading the directory this file lives in (to find index.html).

# --- Third-party imports ---------------------------------------------------
# aiohttp is an async HTTP server (and client). We use it to:
#   - serve index.html when the browser visits "/"
#   - accept the SDP offer at "/offer" and reply with an SDP answer
from aiohttp import web

# aiortc is the Python implementation of WebRTC.
#   RTCPeerConnection      = our side of the peer connection.
#   RTCSessionDescription  = a wrapper around the SDP text + its type
#                            ("offer" or "answer").
from aiortc import RTCPeerConnection, RTCSessionDescription

# Our custom track class lives in video_processor.py. It takes an incoming video
# track from the browser, runs YOLO on every frame, and emits annotated frames.
from video_processor import YOLOVideoTrack


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

# We keep a set of every active RTCPeerConnection so that when the server
# shuts down we can close them cleanly. Without this, half-open connections
# can leak and the process won't exit properly.
peer_connections = set()

# Absolute path to the directory THIS server.py file is in.
# We use it to locate index.html, no matter where the server was launched from.
HERE = os.path.dirname(os.path.abspath(__file__))

# All static assets (HTML pages, CSS, JS) live under project/static/.
STATIC = os.path.join(HERE, "static")


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

def html_handler(filename):
    """Factory: return an aiohttp handler that serves a single HTML file from STATIC."""
    async def handler(request):
        with open(os.path.join(STATIC, filename), "r") as f:
            return web.Response(content_type="text/html", text=f.read())
    return handler


async def offer(request):
    """
    Handler for POST /offer — the WebRTC signaling endpoint.

    The browser will POST a JSON body shaped like:
        {"sdp": "<the offer SDP text>", "type": "offer"}

    We:
      1. Build an RTCPeerConnection on our side.
      2. Hook into "track" events so when the browser sends us its webcam
         track, we wrap it with YOLO and add the wrapped track back to the
         connection (so the browser will receive our annotated version).
      3. Apply the browser's offer as the "remote description."
      4. Generate our own "answer" SDP and send it back as JSON.
    """
    # Parse the JSON body of the request to get the offer SDP and its type.
    params = await request.json()

    # Wrap the raw SDP text and its type into the object aiortc expects.
    offer_description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create our peer connection. With no arguments it uses default ICE settings,
    # which is enough for a same-machine localhost demo.
    pc = RTCPeerConnection()
    peer_connections.add(pc)  # remember it so we can close it on shutdown

    # ----- Event: connection state changes (useful for debugging) ------------
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        # States flow:
        #   "new" -> "connecting" -> "connected" -> "closed"/"failed"/"disconnected"
        # If anything goes wrong (peer hangs up, network dies, etc.) we drop it.
        print(f"[server] connection state -> {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            peer_connections.discard(pc)

    # ----- Event: a new media track has arrived from the browser ------------
    @pc.on("track")
    def on_track(track):
        # `track` is a MediaStreamTrack representing one of the browser's tracks.
        # Its `kind` attribute is "video" or "audio". We only care about video.
        print(f"[server] received track: kind={track.kind}")

        if track.kind == "video":
            # Wrap the raw incoming video track in our YOLO-processing track.
            # That class will pull frames out of `track`, run YOLO, and produce
            # new frames with bounding boxes drawn on them.
            processed_track = YOLOVideoTrack(track)

            # Add the processed track to the peer connection. We do this BEFORE
            # we call createAnswer() below, so the answer SDP will advertise
            # this outgoing track to the browser. As a result, the browser's
            # `ontrack` event will fire with the YOLO-annotated stream.
            pc.addTrack(processed_track)

        # When the incoming track ends (browser closes the camera, tab closes,
        # etc.) we just log it. aiortc cleans up the rest.
        @track.on("ended")
        async def on_ended():
            print(f"[server] track ended: kind={track.kind}")

    # ----- Apply the browser's offer ----------------------------------------
    # setRemoteDescription tells our peer connection: "this is what the OTHER
    # side wants to do" (which codecs they support, what tracks they're sending).
    # IMPORTANT: this is what triggers the "track" event above for each track
    # in the offer, BEFORE this await returns. So by the time we move on, our
    # YOLO track has already been added to `pc`.
    await pc.setRemoteDescription(offer_description)

    # ----- Build our answer --------------------------------------------------
    # createAnswer() looks at the remote offer and our local tracks/transceivers
    # and produces a matching SDP answer (compatible codecs, our network info, etc.).
    answer_description = await pc.createAnswer()

    # setLocalDescription locks in our side of the negotiation. aiortc also
    # waits here for ICE gathering to finish, so by the time this returns,
    # `pc.localDescription.sdp` already contains all our ICE candidates baked in.
    # (This style is called "non-trickle ICE" — simpler than streaming candidates
    # over a separate channel.)
    await pc.setLocalDescription(answer_description)

    # Send the SDP answer back to the browser as JSON.
    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

async def on_shutdown(app):
    """
    Called by aiohttp when the server is stopping (e.g. you pressed Ctrl+C).
    We close every active peer connection so frames stop flowing and
    sockets are released cleanly.
    """
    # Build the close coroutines first, then await them all in parallel.
    close_tasks = [pc.close() for pc in peer_connections]
    await asyncio.gather(*close_tasks)
    peer_connections.clear()


def build_app():
    """
    Construct the aiohttp Application — basically a router + lifecycle hooks.
    """
    app = web.Application()

    # Route table: which URL goes to which handler.
    app.router.add_get("/", html_handler("index.html"))     # landing
    app.router.add_get("/demo", html_handler("demo.html"))  # WebRTC demo
    app.router.add_get("/about", html_handler("about.html"))  # how it works
    app.router.add_post("/offer", offer)                    # WebRTC signaling
    app.router.add_static("/static/", path=STATIC)          # CSS/JS assets

    # Register the shutdown hook so peer connections get closed on Ctrl+C.
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    # web.run_app starts the asyncio event loop and runs the server until Ctrl+C.
    # We bind to 127.0.0.1 (localhost only, NOT exposed on the LAN) on port 8080.
    # Open http://127.0.0.1:8080/ in a browser to use it.
    print("[server] starting on http://127.0.0.1:8080")
    web.run_app(build_app(), host="127.0.0.1", port=8080)
