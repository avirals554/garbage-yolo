const logEl = document.getElementById("log");
const statusText = document.getElementById("statusText");
const statusBadge = document.getElementById("statusBadge");
const startBtn = document.getElementById("startBtn");

function log(msg) {
  console.log(msg);
  logEl.textContent += "\n" + msg;
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text, live = false) {
  statusText.textContent = text;
  statusBadge.classList.toggle("live", live);
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Connecting...";
  setStatus("starting");
  logEl.textContent = "starting...";

  const localStream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });
  log("got webcam");

  const pc = new RTCPeerConnection({});

  pc.addEventListener("track", (event) => {
    log("received remote track from server: kind=" + event.track.kind);
    document.getElementById("remoteVideo").srcObject = event.streams[0];
  });

  pc.addEventListener("connectionstatechange", () => {
    log("pc state -> " + pc.connectionState);
    if (pc.connectionState === "connected") {
      setStatus("live", true);
      startBtn.textContent = "Live";
    } else if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
      setStatus("disconnected");
      startBtn.disabled = false;
      startBtn.textContent = "Start camera";
    }
  });

  for (const track of localStream.getTracks()) {
    pc.addTrack(track, localStream);
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  if (pc.iceGatheringState !== "complete") {
    await new Promise((resolve) => {
      const check = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", check);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", check);
    });
  }
  log("ICE gathering done, sending offer to /offer");

  const response = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  const answer = await response.json();

  await pc.setRemoteDescription(answer);
  log("remote description set — media should start any moment");
});
