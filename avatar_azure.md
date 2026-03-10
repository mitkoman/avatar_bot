# Azure Talking Avatar Integration

`avatar_azure.py` — Travel Assistant powered by Azure Cognitive Services Talking Avatar (WebRTC).

## Architecture Overview

```
Browser
  │
  ├─── GET /speech-token  ─→ Flask ─→ Azure STS (issues short-lived auth token)
  ├─── GET /ice-token     ─→ Flask ─→ Azure TTS relay (TURN credentials)
  ├─── POST /chat         ─→ Flask ─→ LangChain agent ─→ reply text
  │
  └─── WebRTC (video + audio)
         │
         └─── Azure Avatar Service (streams Lisa speaking + idle poses)
```

The page loads the Azure Speech SDK from Microsoft's CDN, initialises a WebRTC peer connection using Azure TURN servers, and streams the avatar video directly from Azure into a `<video>` element. The Flask backend only proxies the auth tokens and chat — it does not process any media.

---

## Backend Routes

### `GET /speech-token`

Issues a short-lived (10-minute) Azure Speech auth token and returns avatar configuration.

**Azure endpoint:**
```
POST https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken
Header: Ocp-Apim-Subscription-Key: {AZURE_SPEECH_KEY}
```

**Response:**
```json
{
  "token": "eyJ...",
  "region": "eastus",
  "character": "lisa",
  "style": "casual-sitting",
  "voice": "en-US-JennyNeural"
}
```

The token is passed to `SpeechSDK.SpeechConfig.fromAuthorizationToken(token, region)` in the browser. It expires after 10 minutes — each page load fetches a fresh one.

---

### `GET /ice-token`

Fetches TURN/ICE relay credentials from Azure so the browser can establish a WebRTC connection through NAT/firewalls.

**Azure endpoint:**
```
GET https://{region}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1
Header: Ocp-Apim-Subscription-Key: {AZURE_SPEECH_KEY}
```

**Response (passed directly to the browser):**
```json
{
  "Urls": ["turn:relay1.azure.com:3478", ...],
  "Username": "...",
  "Password": "..."
}
```

Used to configure `RTCPeerConnection`:
```js
peerConn = new RTCPeerConnection({
  iceServers: [{
    urls: ice.Urls,
    username: ice.Username,
    credential: ice.Password,
  }]
});
```

---

### `POST /chat`

Sends the user message to the LangChain travel agent and returns a reply.

**Request:**
```json
{
  "message": "Best restaurants near me?",
  "session_id": "av_abc123",
  "lat": 48.8566,
  "lng": 2.3522
}
```

The backend automatically prepends context before calling the agent:
- Today's date: `[Today's date: 2024-01-15]`
- User location (if lat/lng provided): `[User's current location: Paris, France (48.8566,2.3522)]`

Location is resolved via Google Maps Geocoding API (`/geocode/json` with `result_type=locality`).

**Response:**
```json
{ "reply": "I recommend Café de Flore on Boulevard Saint-Germain..." }
```

---

### `GET /avatar-img`

Serves `images.webp` — the static placeholder image shown while WebRTC is connecting or if the connection fails.

### `GET /health`

Returns `{"status": "ok"}` — used by Azure App Service health checks.

---

## Frontend: WebRTC Initialisation

The full sequence runs inside `azureInit()`:

```js
async function azureInit() {
  // 1. Fetch speech token + avatar config
  const cfg = await fetch("/speech-token").then(r => r.json());

  // 2. Fetch ICE relay credentials
  const ice = await fetch("/ice-token").then(r => r.json());

  // 3. Create WebRTC peer connection with Azure TURN servers
  peerConn = new RTCPeerConnection({
    iceServers: [{ urls: ice.Urls, username: ice.Username, credential: ice.Password }]
  });

  // 4. Add recv-only transceivers (browser will not send any media)
  peerConn.addTransceiver("video", { direction: "recvonly" });
  peerConn.addTransceiver("audio", { direction: "recvonly" });

  // 5. Wire incoming tracks to video/audio elements
  peerConn.ontrack = (evt) => { ... };

  // 6. Create Speech SDK config + Avatar config
  const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(cfg.token, cfg.region);
  speechConfig.speechSynthesisVoiceName = cfg.voice;

  const avatarConfig = new SpeechSDK.AvatarConfig(cfg.character, cfg.style);
  avatarConfig.backgroundColor = "#0A0A0FFF";  // matches app dark theme

  // 7. Create synthesizer
  synthesizer = new SpeechSDK.AvatarSynthesizer(speechConfig, avatarConfig);
  synthesizer.avatarEventReceived = (s, e) => { ... };

  // 8. Send WebRTC offer to Azure — Azure answers and streams video
  await synthesizer.startAvatarAsync(peerConn);
}
```

`azureInit()` is called immediately on page load so Lisa appears as soon as possible.

---

## Video Display Logic

The avatar video (`#azure-video`) is hidden by CSS initially (`display: none`). The static image (`#avatar-img`) is shown as a loading placeholder.

**One-time transition on first frame:**
```js
azureVideo.onplaying = () => {
  azureVideo.style.display = "block";
  avatarImg.style.display  = "none";
  azureReady = true;
};
```

Once the live video is visible it is **never hidden** — Azure continuously streams the avatar (idle pose when silent, speaking animation when talking). The video only falls back to the static image on ICE failure:

```js
peerConn.oniceconnectionstatechange = () => {
  if (peerConn.iceConnectionState === "failed" ||
      peerConn.iceConnectionState === "disconnected") {
    azureReady = false;
    azureVideo.style.display = "none";
    avatarImg.style.display  = "block";
  }
};
```

---

## Speaking Flow

### Triggering speech

```js
async function azureSpeak(text) {
  if (!synthesizer || !azureReady) return false;
  isSpeaking = true;
  setState("speaking");          // purple pulse on avatar frame
  stopBtn.classList.add("visible");
  startWave([124, 111, 247]);
  await synthesizer.speakTextAsync(text);  // resolves immediately — does NOT wait for speech to finish
  return true;
}
```

> **Important:** `speakTextAsync` resolves as soon as the request is queued, not when speech finishes. Do not use its resolution to reset UI state.

### Detecting when speech ends

Use `avatarEventReceived` — the only reliable signal that speaking has finished:

```js
synthesizer.avatarEventReceived = (s, e) => {
  const ev = (e.description || "").toLowerCase();
  if (ev.includes("switchedtoidle") || ev.includes("talkingended") || ev.includes("idle")) {
    isSpeaking = false;
    setState("");
    stopBtn.classList.remove("visible");
    stopWave();
    // Video is NOT hidden here — Azure streams idle pose continuously
  }
};
```

### Stopping speech mid-sentence

```js
async function stopSpeaking() {
  if (synthesizer && azureReady) {
    await synthesizer.stopSpeakingAsync();
  }
  isSpeaking = false;
  setState("");
  stopBtn.classList.remove("visible");
  stopWave();
}
```

---

## Audio Routing

Azure streams audio as a separate WebRTC track. To ensure reliable playback on desktop Chrome (which requires user gesture to start audio):

```js
peerConn.ontrack = (evt) => {
  if (evt.track.kind === "audio") {
    azureAudio.srcObject = evt.streams[0];
    azureAudio.muted = false;
    azureAudio.play().catch(() => {});
    // Also route through Web Audio API for Chrome reliability
    const ctx = getAudioCtx();
    const src = ctx.createMediaStreamSource(evt.streams[0]);
    src.connect(ctx.destination);
    if (ctx.state === "suspended") ctx.resume();
  }
};
```

`unlockAudio()` is called on the first user interaction (mic tap or text send) to resume the AudioContext.

---

## Speech-to-Text (STT)

Uses the browser's built-in `SpeechRecognition` API (no Azure STT):

```js
recognition = new SpeechRecognition();
recognition.continuous     = false;
recognition.interimResults = true;
recognition.lang           = "en-US";

recognition.onresult = e => {
  const transcript = Array.from(e.results).map(r => r[0].transcript).join("");
  if (e.results[e.results.length - 1].isFinal) {
    recognition.stop();
    sendToBot(transcript);    // → POST /chat → azureSpeak(reply)
  }
};
// Silence/no-speech: restart automatically to keep mic open
recognition.onend = () => {
  if (isRecording) recognition.start();
};
```

Text input is also supported — the `<input>` sends via `sendToBot()` on Enter or send button click.

---

## Geolocation

Requested on first user interaction (to avoid browser popup on load):

```js
navigator.geolocation.getCurrentPosition(
  p => { window._userLoc = { lat: p.coords.latitude, lng: p.coords.longitude }; }
);
```

Coordinates are sent with every `/chat` request so the agent can give location-aware recommendations.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AZURE_SPEECH_KEY` | *(required)* | Azure Cognitive Services Speech resource key |
| `AZURE_SPEECH_REGION` | `eastus` | Azure region (must match the Speech resource) |
| `AVATAR_CHARACTER` | `lisa` | Avatar character name |
| `AVATAR_STYLE` | `casual-sitting` | Avatar pose/style |
| `SPEECH_VOICE` | `en-US-JennyNeural` | Azure Neural TTS voice |
| `GOOGLE_MAPS_KEY` | *(required)* | For Places API and reverse geocoding |
| `OPENAI_API_KEY` | *(required)* | For LangChain agent |

---

## Deployment

Containerised with Docker and deployed to Azure App Service:

**Dockerfile.avatar:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "avatar_azure:app"]
```

Single worker is intentional — the LangChain agent stores session memory in-process.

**deploy_avatar.sh** builds for `linux/amd64`, pushes to `travelbotacr.azurecr.io/travel-avatar:latest`, and restarts the `avatartravel` App Service (resource group: `ai`).

Live URL: `https://avatartravel.azurewebsites.net/`
