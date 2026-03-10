"""
avatar_azure.py – Travel Assistant with Azure AI Talking Avatar
Replaces HeyGen with Azure Cognitive Services Speech Talking Avatar.

Required env vars:
  AZURE_SPEECH_KEY     – from azureaimit CognitiveServices resource (eastus)
  AZURE_SPEECH_REGION  – default: eastus
  AVATAR_CHARACTER     – default: lisa
  AVATAR_STYLE         – default: casual-sitting
  SPEECH_VOICE         – default: en-US-JennyNeural
"""
import os
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv
from openai import AzureOpenAI
from agent import PlacesCapture, CalendarCapture, get_agent, GOOGLE_MAPS_KEY

load_dotenv()

_openai = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
)
_DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

def _spoken_version(full_reply: str) -> str:
    """Turn the full agent reply into one short natural spoken sentence."""
    try:
        r = _openai.chat.completions.create(
            model=_DEPLOY,
            messages=[
                {"role": "system", "content": (
                    "You are a friendly travel assistant speaking out loud. "
                    "Convert the reply below into ONE short natural spoken sentence (max 25 words). "
                    "Sound warm and conversational — like a human guide, not a text reader. "
                    "Do not list places, addresses, ratings or numbers. "
                    "Just give a brief friendly spoken intro to the answer."
                )},
                {"role": "user", "content": full_reply},
            ],
            max_tokens=60,
            temperature=0.6,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
AVATAR_FILE   = "images.webp"

SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")
AVATAR_CHAR   = os.environ.get("AVATAR_CHARACTER", "lisa")
AVATAR_STYLE  = os.environ.get("AVATAR_STYLE", "casual-sitting")
SPEECH_VOICE  = os.environ.get("SPEECH_VOICE", "en-US-JennyNeural")


def _speech_headers():
    return {"Ocp-Apim-Subscription-Key": SPEECH_KEY}


def _reverse_geocode(lat, lng):
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"latlng": f"{lat},{lng}", "key": GOOGLE_MAPS_KEY, "result_type": "locality"},
            timeout=5,
        )
        results = resp.json().get("results", [])
        if results:
            return results[0].get("formatted_address", f"{lat:.4f},{lng:.4f}")
    except Exception:
        pass
    return f"{lat:.4f},{lng:.4f}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(HTML, mimetype='text/html')


@app.route("/avatar-img")
def avatar_img():
    return send_from_directory(BASE_DIR, AVATAR_FILE)


@app.route("/speech-token")
def speech_token():
    """Return a short-lived Azure Speech auth token + avatar config."""
    if not SPEECH_KEY:
        return jsonify({"error": "AZURE_SPEECH_KEY not set"}), 503
    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    try:
        r = requests.post(url, headers=_speech_headers(), timeout=10)
        if r.status_code != 200:
            return jsonify({"error": f"Token error {r.status_code}: {r.text[:200]}"}), 500
        return jsonify({
            "token": r.text,
            "region": SPEECH_REGION,
            "character": AVATAR_CHAR,
            "style": AVATAR_STYLE,
            "voice": SPEECH_VOICE,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ice-token")
def ice_token():
    """Return Azure TURN/ICE relay credentials for WebRTC."""
    if not SPEECH_KEY:
        return jsonify({"error": "AZURE_SPEECH_KEY not set"}), 503
    url = f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1"
    try:
        r = requests.get(url, headers=_speech_headers(), timeout=10)
        if r.status_code != 200:
            return jsonify({"error": f"ICE error {r.status_code}: {r.text[:200]}"}), 500
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()
    session_id = data.get("session_id", "azure_avatar")
    if not user_input:
        return jsonify({"error": "empty message"}), 400
    today = datetime.now().strftime("%Y-%m-%d")
    user_input = f"[Today's date: {today}] {user_input}"
    lat, lng = data.get("lat"), data.get("lng")
    if lat and lng:
        city = _reverse_geocode(lat, lng)
        user_input = f"[User's current location: {city} ({lat:.4f},{lng:.4f})] {user_input}"
    try:
        capture  = PlacesCapture()
        cal      = CalendarCapture()
        executor = get_agent(session_id)
        result   = executor.invoke({"input": user_input}, config={"callbacks": [capture, cal]})
        full     = result["output"]
        speak    = _spoken_version(full)
        return jsonify({"reply": full, "speak": speak})
    except Exception as e:
        return jsonify({"reply": f"Sorry, something went wrong: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0e0b23">
<title>Travel Assistant</title>
<script src="https://aka.ms/csspeech/jsbrowserpackageraw"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0e0b23;
    --surface: #13102e;
    --card:    #1e1a42;
    --orange:  #F47920;
    --purple:  #6c63ff;
    --green:   #22c55e;
    --amber:   #ffa726;
    --red:     #e53935;
    --text:    #e2e0f0;
    --muted:   #8b82b5;
    --border:  #2a2550;
  }

  html, body {
    height: 100%; background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }

  /* ── Background blobs ── */
  .bg-circle {
    position: fixed; border-radius: 50%;
    filter: blur(80px); opacity: 0.35;
    pointer-events: none; z-index: 0;
  }
  .circle-red    { width: 320px; height: 320px; background: #ef4444; top: -80px;  left: -80px; }
  .circle-orange { width: 260px; height: 260px; background: #F47920; bottom: 0;   right: -60px; }
  .circle-blue   { width: 200px; height: 200px; background: #6c63ff; bottom: 30%; left: -40px; }

  /* ── App shell ── */
  #app {
    position: relative; z-index: 1;
    display: flex; flex-direction: column;
    width: min(420px, 100vw);
    height: 92vh; max-height: 820px;
    border-radius: 18px; overflow: hidden;
    border: 2px solid var(--orange);
    box-shadow: 0 8px 48px rgba(0,0,0,0.6), 0 0 28px rgba(244,121,32,0.2);
    background: var(--surface);
  }

  /* ── Header ── */
  .header {
    background: var(--surface);
    padding: 14px 16px 12px;
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .header-icon {
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, var(--orange), var(--purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
  }
  .header-info { flex: 1; }
  .header-title { font-size: 15px; font-weight: 700; color: #fff; }
  .header-sub {
    display: flex; align-items: center; gap: 5px;
    font-size: 12px; color: var(--muted); margin-top: 2px;
  }
  .dot-green { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; }

  /* ── Avatar section ── */
  #avatar-section {
    flex: 1; position: relative; overflow: hidden;
    background: #07060f;
  }

  #avatar-frame {
    width: 100%; height: 100%;
    position: relative; transition: box-shadow 0.35s;
  }
  #avatar-frame.speaking { animation: speakPulse 1.4s ease-in-out infinite; }
  #avatar-frame.listening {
    box-shadow: inset 0 0 0 3px var(--green), inset 0 0 50px rgba(34,197,94,0.18);
  }
  #avatar-frame.thinking {
    box-shadow: inset 0 0 0 3px var(--amber), inset 0 0 40px rgba(255,167,38,0.18);
  }
  @keyframes speakPulse {
    0%,100% { box-shadow: inset 0 0 0 3px var(--purple), inset 0 0 40px rgba(108,99,255,0.25); }
    50%      { box-shadow: inset 0 0 0 3px var(--purple), inset 0 0 70px rgba(108,99,255,0.5); }
  }

  #avatar-img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: center top; display: block;
  }

  /* Azure live video — shown once WebRTC stream is active */
  #azure-video {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover; object-position: center top;
    display: none;
  }

  /* Listening ripples */
  #ripples {
    position: absolute; inset: 0;
    pointer-events: none; display: none;
  }
  #ripples.visible { display: block; }
  .ripple {
    position: absolute; inset: 0;
    border: 2px solid var(--green);
    opacity: 0;
    animation: rippleAnim 2s ease-out infinite;
  }
  .ripple:nth-child(2) { animation-delay: 0.65s; }
  .ripple:nth-child(3) { animation-delay: 1.3s; }
  @keyframes rippleAnim {
    0%   { transform: scale(1);    opacity: 0.5; }
    100% { transform: scale(1.05); opacity: 0; }
  }

  /* Status badge — floats on avatar */
  #status-label {
    position: absolute;
    bottom: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(14,11,35,0.82); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 5px 18px;
    font-size: 12px; font-weight: 500; color: var(--muted);
    white-space: nowrap; z-index: 10;
    letter-spacing: 0.03em; transition: color 0.3s, border-color 0.3s;
  }
  #status-label.speaking  { color: #a78bfa; border-color: rgba(167,139,250,0.45); }
  #status-label.listening { color: var(--green);  border-color: rgba(34,197,94,0.45); }
  #status-label.thinking  { color: var(--amber);  border-color: rgba(255,167,38,0.45); }

  /* ── Subtitle ── */
  #subtitle {
    padding: 10px 18px 8px;
    font-size: 13px; line-height: 1.55;
    color: var(--text); text-align: center;
    min-height: 54px; max-height: 72px;
    overflow-y: auto;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  /* ── Waveform ── */
  #waveform {
    height: 32px; display: block; flex-shrink: 0;
    background: var(--surface);
  }

  /* ── Controls ── */
  #controls {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px;
    padding-bottom: calc(10px + env(safe-area-inset-bottom, 0px));
    background: var(--surface);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }

  #mic-btn {
    width: 50px; height: 50px; border-radius: 50%; flex-shrink: 0;
    background: var(--orange); color: #fff; border: none;
    font-size: 20px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 18px rgba(244,121,32,0.4);
    transition: transform 0.1s, background 0.2s, box-shadow 0.2s;
  }
  #mic-btn.listening {
    background: var(--red);
    box-shadow: 0 4px 22px rgba(229,57,53,0.5);
    animation: micPulse 1s ease-in-out infinite;
  }
  #mic-btn:active { transform: scale(0.92); }
  @keyframes micPulse {
    0%,100% { box-shadow: 0 4px 22px rgba(229,57,53,0.5); }
    50%      { box-shadow: 0 4px 34px rgba(229,57,53,0.8); }
  }

  #text-input {
    flex: 1; background: var(--card);
    color: var(--text); border: 1px solid var(--border);
    border-radius: 22px; padding: 10px 16px;
    font-size: 14px; outline: none; font-family: inherit;
    -webkit-appearance: none;
  }
  #text-input::placeholder { color: var(--muted); }
  #text-input:focus { border-color: rgba(244,121,32,0.5); }

  #send-btn {
    width: 44px; height: 44px; border-radius: 50%; flex-shrink: 0;
    background: #5b21b6; color: #fff; border: none;
    font-size: 17px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s;
  }
  #send-btn:hover { background: #6d28d9; }

  #stop-btn {
    width: 40px; height: 40px; border-radius: 50%; flex-shrink: 0;
    background: rgba(229,57,53,0.15); color: var(--red); border: none;
    font-size: 17px; cursor: pointer;
    display: none; align-items: center; justify-content: center;
    transition: background 0.15s;
  }
  #stop-btn.visible { display: flex; }
  #stop-btn:active  { background: rgba(229,57,53,0.25); }
</style>
</head>
<body>
<div class="bg-circle circle-red"></div>
<div class="bg-circle circle-orange"></div>
<div class="bg-circle circle-blue"></div>

<div id="app">

  <div class="header">
    <div class="header-icon">&#9992;&#65039;</div>
    <div class="header-info">
      <div class="header-title">Travel Assistant</div>
      <div class="header-sub">
        <span class="dot-green"></span>
        Restaurants &middot; Hotels &middot; Sightseeing
      </div>
    </div>
  </div>

  <div id="avatar-section">
    <div id="avatar-frame">
      <img id="avatar-img" src="/avatar-img" alt="Travel Assistant">
      <video id="azure-video" playsinline autoplay muted></video>
      <audio id="azure-audio" style="display:none"></audio>
    </div>
    <div id="ripples">
      <div class="ripple"></div>
      <div class="ripple"></div>
      <div class="ripple"></div>
    </div>
    <div id="status-label">Tap &#127908; to ask anything</div>
  </div>

  <div id="subtitle">Hi! I&apos;m your travel assistant. Ask me about restaurants, hotels, day tours &mdash; anything travel.</div>

  <canvas id="waveform"></canvas>

  <div id="controls">
    <button id="mic-btn" onclick="toggleMic()">&#127908;</button>
    <input id="text-input" type="text" placeholder="Or type a question...">
    <button id="send-btn" onclick="sendText()">&#10148;</button>
    <button id="stop-btn" onclick="stopSpeaking()">&#9209;</button>
  </div>

</div>
<script>
const SESSION_ID  = "av_" + Math.random().toString(36).slice(2);
const avatarFrame = document.getElementById("avatar-frame");
const ripples     = document.getElementById("ripples");
const statusLabel = document.getElementById("status-label");
const subtitleEl  = document.getElementById("subtitle");
const micBtn      = document.getElementById("mic-btn");
const stopBtn     = document.getElementById("stop-btn");
const textInput   = document.getElementById("text-input");
const azureVideo  = document.getElementById("azure-video");
const azureAudio  = document.getElementById("azure-audio");
const avatarImg   = document.getElementById("avatar-img");

// ── State ─────────────────────────────────────────────────────────────────────
let synthesizer     = null;
let peerConn        = null;
let azureReady      = false;
let azureInitPromise = null;
let isSpeaking      = false;

// ── Waveform ──────────────────────────────────────────────────────────────────
const wCanvas = document.getElementById("waveform");
const wCtx    = wCanvas.getContext("2d");
let waveAF    = null;
let waveColor = [124, 111, 247];

function resizeWave() {
  const r = wCanvas.getBoundingClientRect();
  wCanvas.width = r.width; wCanvas.height = r.height;
}
window.addEventListener("resize", resizeWave);
setTimeout(resizeWave, 80);

function drawWave(active) {
  wCtx.clearRect(0, 0, wCanvas.width, wCanvas.height);
  const cy = wCanvas.height / 2;
  if (!active) {
    wCtx.strokeStyle = "rgba(255,255,255,0.08)";
    wCtx.lineWidth = 1;
    wCtx.beginPath(); wCtx.moveTo(0, cy); wCtx.lineTo(wCanvas.width, cy); wCtx.stroke();
    return;
  }
  const bars = 42, bw = wCanvas.width / bars;
  const [r, g, b] = waveColor;
  for (let i = 0; i < bars; i++) {
    const t = Date.now() / 180 + i * 0.52;
    const h = (0.25 + 0.75 * Math.abs(Math.sin(t))) * cy * 0.82;
    const a = 0.45 + 0.55 * Math.abs(Math.sin(t * 0.8));
    wCtx.fillStyle = `rgba(${r},${g},${b},${a})`;
    const x = i * bw + bw * 0.15, w = bw * 0.7;
    wCtx.beginPath();
    wCtx.moveTo(x + 2, cy - h); wCtx.lineTo(x + w - 2, cy - h);
    wCtx.lineTo(x + w, cy - h + 2); wCtx.lineTo(x + w, cy + h - 2);
    wCtx.lineTo(x + w - 2, cy + h); wCtx.lineTo(x + 2, cy + h);
    wCtx.lineTo(x, cy + h - 2); wCtx.lineTo(x, cy - h + 2);
    wCtx.closePath(); wCtx.fill();
  }
}

function startWave(color) {
  waveColor = color;
  if (waveAF) cancelAnimationFrame(waveAF);
  const loop = () => { drawWave(true); waveAF = requestAnimationFrame(loop); };
  loop();
}
function stopWave() {
  if (waveAF) { cancelAnimationFrame(waveAF); waveAF = null; }
  drawWave(false);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setState(state) {
  avatarFrame.className = state || "";
  ripples.className     = state === "listening" ? "visible" : "";
  statusLabel.className = state || "";
  const labels = { speaking: "Speaking\u2026", listening: "Listening\u2026", thinking: "Thinking\u2026" };
  statusLabel.textContent = labels[state] || "Tap \U0001F3A4 to ask anything";
}

function cleanForSpeech(text) {
  return text
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[#*_`]/g, "")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/CHART:[^\\n]*/g, "")
    .slice(0, 700);
}

// ── Browser TTS fallback ──────────────────────────────────────────────────────
function speakFallback(text) {
  if (!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt  = new SpeechSynthesisUtterance(text);
  utt.lang   = "en-US"; utt.rate = 0.91; utt.pitch = 1.06;
  const voices = speechSynthesis.getVoices();
  const pick = voices.find(v => v.lang.startsWith("en") &&
    /daniel|alex|fred|male|man|guy|david|james|tom/i.test(v.name)) ||
    voices.find(v => v.lang.startsWith("en"));
  if (pick) utt.voice = pick;
  utt.onstart = () => { isSpeaking = true; setState("speaking"); stopBtn.classList.add("visible"); startWave([124, 111, 247]); };
  utt.onend = utt.onerror = () => { isSpeaking = false; setState(""); stopBtn.classList.remove("visible"); stopWave(); };
  speechSynthesis.speak(utt);
}

function stopFallback() {
  if (window.speechSynthesis) speechSynthesis.cancel();
}

// ── Azure Talking Avatar ───────────────────────────────────────────────────────
async function azureInit() {
  if (typeof SpeechSDK === "undefined") {
    console.warn("Azure Speech SDK not loaded — falling back to browser TTS");
    return;
  }
  try {
    statusLabel.textContent = "Connecting avatar\u2026";

    // 1. Fetch speech token + avatar config
    const cfg = await fetch("/speech-token").then(r => r.json());
    if (cfg.error) throw new Error(cfg.error);

    // 2. Fetch ICE relay token for WebRTC
    const ice = await fetch("/ice-token").then(r => r.json());
    if (ice.error) throw new Error(ice.error);

    // 3. Setup WebRTC peer connection with Azure ICE servers
    peerConn = new RTCPeerConnection({
      iceServers: [{
        urls: ice.Urls,
        username: ice.Username,
        credential: ice.Password,
      }]
    });

    // Receive video + audio tracks from Azure avatar service
    peerConn.addTransceiver("video", { direction: "recvonly" });
    peerConn.addTransceiver("audio", { direction: "recvonly" });

    peerConn.ontrack = (evt) => {
      if (evt.track.kind === "video") {
        azureVideo.srcObject = evt.streams[0];
        azureVideo.onplaying = () => {
          azureVideo.style.display = "block";
          avatarImg.style.display  = "none";
          azureReady = true;
          setState("");
        };
        azureVideo.play().catch(() => {});
      }
      if (evt.track.kind === "audio") {
        azureAudio.srcObject = evt.streams[0];
        azureAudio.muted = false;
        azureAudio.play().catch(() => {});
        // Also route through Web Audio API so desktop Chrome plays it reliably
        try {
          const ctx = getAudioCtx();
          if (ctx) {
            const src = ctx.createMediaStreamSource(evt.streams[0]);
            src.connect(ctx.destination);
            if (ctx.state === "suspended") ctx.resume();
          }
        } catch(e) {}
      }
    };

    peerConn.oniceconnectionstatechange = () => {
      if (peerConn.iceConnectionState === "failed" ||
          peerConn.iceConnectionState === "disconnected") {
        azureReady = false;
        azureVideo.style.display = "none";
        avatarImg.style.display  = "block";
        setState("");
        stopBtn.classList.remove("visible");
        stopWave();
        statusLabel.textContent = "Tap \U0001F3A4 to ask anything";
      }
    };

    // 4. Create Speech SDK config
    const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(cfg.token, cfg.region);
    speechConfig.speechSynthesisVoiceName = cfg.voice;

    // 5. Create Avatar config — dark background to match app theme
    const avatarConfig = new SpeechSDK.AvatarConfig(cfg.character, cfg.style);
    avatarConfig.backgroundColor = "#0A0A0FFF";

    // 6. Create synthesizer and attach events
    synthesizer = new SpeechSDK.AvatarSynthesizer(speechConfig, avatarConfig);

    synthesizer.avatarEventReceived = (s, e) => {
      const ev = (e.description || "").toLowerCase();
      if (ev.includes("switchedtoidle") || ev.includes("talkingended") || ev.includes("idle")) {
        isSpeaking = false;
        setState("");
        stopBtn.classList.remove("visible");
        stopWave();
      }
      // Video stays visible at all times once connected — Azure shows idle pose when not speaking
    };

    // 7. Start avatar — this sends the WebRTC offer to Azure
    const result = await synthesizer.startAvatarAsync(peerConn);
    if (result.reason === SpeechSDK.ResultReason.Canceled) {
      const details = SpeechSDK.CancellationDetails.fromResult(result);
      throw new Error("Avatar start cancelled: " + details.errorDetails);
    }

  } catch(e) {
    console.warn("Azure Avatar init failed:", e.message || e);
    azureReady = false;
    synthesizer = null;
    avatarImg.style.display  = "block";
    azureVideo.style.display = "none";
    statusLabel.textContent  = "Tap \U0001F3A4 to ask anything";
  }
}

// Ensures init runs once, even if called multiple times concurrently
async function ensureAzureReady() {
  if (azureReady) return true;
  if (!azureInitPromise) azureInitPromise = azureInit();
  await azureInitPromise;
  return azureReady;
}

async function azureSpeak(text) {
  if (!synthesizer || !azureReady) return false;
  try {
    isSpeaking = true;
    setState("speaking");
    stopBtn.classList.add("visible");
    startWave([124, 111, 247]);
    await synthesizer.speakTextAsync(text);
    return true;
  } catch(e) {
    console.warn("azureSpeak failed:", e);
    isSpeaking = false;
    setState("");
    stopBtn.classList.remove("visible");
    stopWave();
    return false;
  }
}

async function stopSpeaking() {
  stopFallback();
  if (synthesizer && azureReady) {
    try { await synthesizer.stopSpeakingAsync(); } catch(e) {}
  }
  isSpeaking = false;
  setState("");
  stopBtn.classList.remove("visible");
  stopWave();
}

// ── STT ───────────────────────────────────────────────────────────────────────
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null, isRecording = false;

if (SR) {
  recognition = new SR();
  recognition.continuous     = false;
  recognition.interimResults = true;
  recognition.lang           = "en-US";

  recognition.onresult = e => {
    const t = Array.from(e.results).map(r => r[0].transcript).join("");
    subtitleEl.textContent = t;
    if (e.results[e.results.length - 1].isFinal) {
      isRecording = false;   // mark done BEFORE stop so onend doesn't restart
      recognition.stop();
      sendToBot(t);
    }
  };
  recognition.onend = () => {
    if (isRecording) {
      // Ended without a final result (silence/no-speech) — keep mic open
      try { recognition.start(); return; } catch(e) {}
    }
    micBtn.classList.remove("listening");
    micBtn.textContent = "\U0001F3A4";
    if (!isSpeaking) { setState(""); stopWave(); }
  };
  recognition.onerror = (err) => {
    if (err.error === "no-speech") return;  // onend handles the restart
    isRecording = false;
    micBtn.classList.remove("listening");
    micBtn.textContent = "\U0001F3A4";
    setState(""); stopWave();
  };
}

async function toggleMic() {
  unlockAudio();
  if (!recognition) {
    subtitleEl.textContent = "Speech not supported \u2014 please type.";
    return;
  }
  if (isRecording) { recognition.stop(); return; }
  if (isSpeaking) await stopSpeaking();
  // Lazy-init Azure avatar on first interaction
  ensureAzureReady();
  isRecording = true;
  micBtn.classList.add("listening");
  micBtn.textContent = "\u23F9";
  subtitleEl.textContent = "";
  setState("listening");
  startWave([76, 175, 80]);
  recognition.start();
}

// ── Send message ───────────────────────────────────────────────────────────────
async function sendToBot(text) {
  if (!text.trim()) return;
  subtitleEl.textContent = "\u2026";
  setState("thinking");
  startWave([255, 167, 38]);
  try {
    const res  = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: SESSION_ID }),
    });
    const data  = await res.json();
    const reply = data.reply || "I didn\u2019t get that, please try again.";
    subtitleEl.textContent = reply.length > 220 ? reply.slice(0, 220) + "\u2026" : reply;
    stopWave();

    const clean  = data.speak || cleanForSpeech(reply);
    const spoken = await azureSpeak(clean);
    if (!spoken) speakFallback(clean);          // fallback to browser TTS
  } catch(e) {
    setState(""); stopWave();
    subtitleEl.textContent = "Connection error. Please try again.";
  }
}

// ── Audio context (desktop unlock) ────────────────────────────────────────────
let audioCtx = null;
let audioUnlocked = false;

function getAudioCtx() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) audioCtx = new AC();
  }
  return audioCtx;
}

function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  // Resume AudioContext — required on desktop Chrome/Edge before audio plays
  const ctx = getAudioCtx();
  if (ctx && ctx.state === "suspended") ctx.resume();
  azureAudio.muted = false;
  azureAudio.play().catch(() => {});
  // Request geolocation on first interaction (avoids popup on page load)
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      p => { window._userLoc = {lat: p.coords.latitude, lng: p.coords.longitude}; },
      () => {},
      { timeout: 8000 }
    );
  }
}

function sendText() {
  const t = textInput.value.trim();
  if (!t) return;
  unlockAudio();
  subtitleEl.textContent = t;
  textInput.value = "";
  sendToBot(t);
}

textInput.addEventListener("keydown", e => { if (e.key === "Enter") sendText(); });

// Pre-load browser voices
if (window.speechSynthesis) {
  speechSynthesis.getVoices();
  speechSynthesis.addEventListener("voiceschanged", () => speechSynthesis.getVoices());
}

// Start Azure avatar on page load so Lisa is visible immediately
azureInit();


</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
