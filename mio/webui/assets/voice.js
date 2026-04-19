// voice.js — Voice Mode (browser-only MVP).
//
// Uses the Web Speech API for STT (SpeechRecognition) and TTS
// (SpeechSynthesis). No Python backend needed — works today on
// Safari 17+, Chrome, Edge. Firefox has SpeechSynthesis only, no
// SpeechRecognition; we detect and degrade.
//
// UI: a floating 🎤 action button (bottom-right, above the sovereignty
// bar). Clicking opens a full-screen orb overlay. The orb animates
// per state:
//   idle       — slow breath pulse
//   listening  — concentric ripple, live transcript grows below
//   thinking   — spinning dots while the model answers
//   speaking   — bright shimmer while TTS is active
//
// Interruption: tap the orb while speaking to cancel TTS; hold
// Space to push-to-talk instead of always-on.
//
// A later iteration will swap Web Speech for local Whisper + Kokoro
// via a /ui/ws/voice WebSocket so nothing leaves the machine.

(function () {
  window.Mio = window.Mio || {};
  if (window.Mio.voice) return;

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hasSR = !!SR;
  const hasTTS = typeof window.speechSynthesis !== "undefined";
  const PTT_KEY = "Space";

  let state = "idle"; // idle | listening | thinking | speaking
  let overlay = null, orb = null, transcriptEl = null, answerEl = null;
  let rec = null;
  let currentText = "";

  function mountFab() {
    if (document.querySelector(".mio-voice-fab")) return;
    const btn = document.createElement("button");
    btn.className = "mio-voice-fab";
    btn.title = hasSR ? "Voice Mode (hold Space to push-to-talk)" : "Voice Mode — needs Chrome/Edge/Safari 17+";
    btn.innerHTML = "🎤";
    btn.addEventListener("click", openOverlay);
    document.body.appendChild(btn);
  }

  function openOverlay() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.className = "mio-voice-overlay";
    overlay.innerHTML = `
      <button class="mio-voice-close" aria-label="Close">×</button>
      <div class="mio-voice-orb" data-state="idle">
        <div class="mio-voice-orb-core"></div>
        <div class="mio-voice-ripple"></div>
        <div class="mio-voice-ripple"></div>
        <div class="mio-voice-ripple"></div>
      </div>
      <div class="mio-voice-transcript" id="voice-transcript">${hasSR ? "Tap the orb or hold Space to talk." : "Voice recognition not supported in this browser. You can still receive TTS replies by sending a message from the chat."}</div>
      <div class="mio-voice-answer"    id="voice-answer"></div>
      <div class="mio-voice-hint">${hasSR ? "Tap orb = continuous · Hold Space = push-to-talk · Esc = close" : ""}</div>
    `;
    document.body.appendChild(overlay);
    orb          = overlay.querySelector(".mio-voice-orb");
    transcriptEl = overlay.querySelector("#voice-transcript");
    answerEl     = overlay.querySelector("#voice-answer");
    overlay.querySelector(".mio-voice-close").addEventListener("click", closeOverlay);
    orb.addEventListener("click", () => {
      if (!hasSR) return;
      if (state === "listening") stopListening();
      else if (state === "speaking") cancelSpeech();
      else startListening(false); // continuous
    });
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup",   onKeyUp);
  }

  function closeOverlay() {
    stopListening();
    cancelSpeech();
    overlay?.remove();
    overlay = orb = transcriptEl = answerEl = null;
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("keyup",   onKeyUp);
    setState("idle");
  }

  function onKey(e) {
    if (!overlay) return;
    if (e.key === "Escape") { e.preventDefault(); closeOverlay(); return; }
    if (!hasSR) return;
    if (e.code === PTT_KEY && !e.repeat) {
      // Ignore if focus is in a text input (shouldn't be during overlay)
      const t = e.target;
      if (t?.tagName === "TEXTAREA" || t?.tagName === "INPUT") return;
      e.preventDefault();
      if (state !== "listening") startListening(true);
    }
  }
  function onKeyUp(e) {
    if (!overlay) return;
    if (e.code === PTT_KEY && state === "listening" && rec && rec._ptt) {
      e.preventDefault();
      stopListening();
    }
  }

  function setState(s) {
    state = s;
    if (orb) orb.dataset.state = s;
  }

  function startListening(ptt) {
    if (!hasSR) return;
    try { rec?.abort(); } catch {}
    rec = new SR();
    rec._ptt = !!ptt;
    rec.continuous = !ptt;
    rec.interimResults = true;
    rec.lang = "en-US";
    currentText = "";
    setState("listening");
    if (transcriptEl) transcriptEl.textContent = "Listening…";
    if (answerEl)     answerEl.textContent = "";

    rec.onresult = (ev) => {
      let interim = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) currentText += r[0].transcript + " ";
        else           interim += r[0].transcript;
      }
      if (transcriptEl) {
        transcriptEl.textContent = (currentText || "") + (interim ? " · " + interim : "");
      }
    };
    rec.onerror = () => { setState("idle"); };
    rec.onend = () => {
      if (state === "listening") {
        // Either continuous rec stopped (mic glitch) or PTT released
        commit();
      }
    };
    try { rec.start(); } catch { setState("idle"); }
  }

  function stopListening() {
    if (!rec) return;
    try { rec.stop(); } catch {}
  }

  async function commit() {
    const text = currentText.trim();
    currentText = "";
    if (!text) { setState("idle"); return; }
    if (transcriptEl) transcriptEl.textContent = "You: " + text;
    setState("thinking");
    if (answerEl) answerEl.textContent = "Thinking…";

    try {
      const answer = await askModel(text);
      if (answerEl) answerEl.textContent = "Mio: " + answer;
      speak(answer);
    } catch (e) {
      if (answerEl) answerEl.textContent = "Error: " + e.message;
      setState("idle");
    }
  }

  async function askModel(prompt) {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mio-auto",
        messages: [
          { role: "system", content: "You are Mio in voice mode. Keep replies short and conversational — one or two sentences unless the question genuinely needs more." },
          { role: "user", content: prompt },
        ],
        temperature: 0.7, max_tokens: 280, stream: false,
      }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    return data.choices?.[0]?.message?.content || "(no reply)";
  }

  function speak(text) {
    if (!hasTTS) { setState("idle"); return; }
    cancelSpeech();
    setState("speaking");
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0; u.pitch = 1.0;
    u.onend = u.onerror = () => setState("idle");
    speechSynthesis.speak(u);
  }

  function cancelSpeech() {
    if (!hasTTS) return;
    try { speechSynthesis.cancel(); } catch {}
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountFab, { once: true });
  } else {
    mountFab();
  }

  window.Mio.voice = { open: openOverlay, close: closeOverlay };
})();
