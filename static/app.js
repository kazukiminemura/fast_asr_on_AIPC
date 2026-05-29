const state = {
  inputMode: "mic",
  liveRunning: false,
  warmed: false,
  stream: null,
  audioContext: null,
  source: null,
  processor: null,
  analyser: null,
  meterTimer: null,
  chunkTimer: null,
  chunks: [],
  submitChain: Promise.resolve(),
  chunkIndex: 0,
};

const els = {
  micPane: document.querySelector("#micPane"),
  startLiveButton: document.querySelector("#startLiveButton"),
  stopLiveButton: document.querySelector("#stopLiveButton"),
  clearButton: document.querySelector("#clearButton"),
  warmupNotice: document.querySelector("#warmupNotice"),
  liveLog: document.querySelector("#liveLog"),
  meterBar: document.querySelector("#meterBar"),
  runStatus: document.querySelector("#runStatus"),
  transcript: document.querySelector("#transcript"),
  metrics: document.querySelector("#metrics"),
  deviceStatus: document.querySelector("#deviceStatus"),
};

els.startLiveButton.addEventListener("click", startLive);
els.stopLiveButton.addEventListener("click", stopLive);
els.clearButton.addEventListener("click", clearOutput);
document.querySelector("#model").addEventListener("input", resetWarmup);
document.querySelector("#device").addEventListener("change", resetWarmup);

loadDevices();

async function loadDevices() {
  try {
    const response = await fetch("/api/devices");
    const data = await response.json();
    els.deviceStatus.textContent = data.devices?.length
      ? `OpenVINO: ${data.devices.join(", ")}`
      : "OpenVINO devices unavailable";
  } catch {
    els.deviceStatus.textContent = "device check failed";
  }
}

async function startLive() {
  if (state.liveRunning) {
    return;
  }

  clearOutput();
  setBusy(true, state.warmed ? "starting" : "warming up");
  els.startLiveButton.disabled = true;

  try {
    await warmupModel();
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioContext = new AudioContext();
    state.source = state.audioContext.createMediaStreamSource(state.stream);
    state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 256;
    state.chunks = [];
    state.chunkIndex = 0;
    state.submitChain = Promise.resolve();

    state.source.connect(state.analyser);
    state.analyser.connect(state.processor);
    state.processor.connect(state.audioContext.destination);
    state.processor.onaudioprocess = (event) => {
      if (state.liveRunning) {
        state.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      }
    };

    state.liveRunning = true;
    els.stopLiveButton.disabled = false;
    els.liveLog.textContent = "listening";
    setBusy(false, "listening");
    startMeter();
    state.chunkTimer = setInterval(flushLiveChunk, getChunkMs());
  } catch (error) {
    showError(error.message || String(error));
    await cleanupLive();
  } finally {
    els.startLiveButton.disabled = state.liveRunning;
  }
}

async function warmupModel() {
  if (state.warmed) {
    return;
  }

  els.warmupNotice.classList.add("active");
  els.liveLog.textContent = "warming up model; first load can take a while";
  const response = await fetch("/api/warmup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: document.querySelector("#model").value.trim(),
      device: document.querySelector("#device").value,
    }),
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || "Warmup failed.");
  }

  state.warmed = true;
  els.warmupNotice.textContent = data.cache_hit
    ? `Warm cache ready on ${data.selected_device}. OpenVINO cache: ${data.cache_dir}`
    : `Warmup complete on ${data.selected_device} in ${formatNumber(data.model_load_seconds)}s. OpenVINO cache saved to ${data.cache_dir}`;
  els.liveLog.textContent = "warmup complete";
}

function resetWarmup() {
  state.warmed = false;
  els.warmupNotice.classList.remove("active");
  els.warmupNotice.textContent =
    "The first start warms up the Whisper model and OpenVINO device cache. It can take a while. Later starts reuse the in-memory pipeline, and OpenVINO cache files remain under cache\\openvino for faster startup after restart.";
}

async function stopLive() {
  if (!state.liveRunning) {
    return;
  }

  state.liveRunning = false;
  setBusy(true, "stopping");
  clearInterval(state.chunkTimer);
  clearInterval(state.meterTimer);
  els.meterBar.style.width = "0%";
  els.stopLiveButton.disabled = true;
  await flushLiveChunk();
  await state.submitChain;
  await cleanupLive();
  els.startLiveButton.disabled = false;
  els.liveLog.textContent = "stopped";
  setBusy(false, "idle");
}

async function cleanupLive() {
  state.processor?.disconnect();
  state.analyser?.disconnect();
  state.source?.disconnect();
  state.stream?.getTracks().forEach((track) => track.stop());
  if (state.audioContext) {
    await state.audioContext.close();
  }
  state.stream = null;
  state.audioContext = null;
  state.source = null;
  state.processor = null;
  state.analyser = null;
}

function startMeter() {
  const levelData = new Uint8Array(state.analyser.frequencyBinCount);
  state.meterTimer = setInterval(() => {
    state.analyser.getByteFrequencyData(levelData);
    const average = levelData.reduce((sum, value) => sum + value, 0) / levelData.length;
    els.meterBar.style.width = `${Math.min(100, average * 1.4)}%`;
  }, 100);
}

async function flushLiveChunk() {
  if (!state.chunks.length || !state.audioContext) {
    return;
  }

  const chunks = state.chunks;
  state.chunks = [];
  const chunkNumber = state.chunkIndex + 1;
  state.chunkIndex = chunkNumber;
  const merged = mergeFloat32(chunks);
  const resampled = resampleLinear(merged, state.audioContext.sampleRate, 16000);
  const wav = encodeWav(resampled, 16000);
  const blob = new Blob([wav], { type: "audio/wav" });

  state.submitChain = state.submitChain.catch(() => {}).then(async () => {
    els.liveLog.textContent = `transcribing chunk ${chunkNumber}`;
    const data = await submitAudio(blob, `live-${chunkNumber}.wav`, { keepBusy: true });
    appendLiveResult(data, chunkNumber);
    els.liveLog.textContent = state.liveRunning ? "listening" : "stopped";
  });
  await state.submitChain.catch((error) => showError(error.message || String(error)));
}

async function submitAudio(blob, filename, options = {}) {
  const formData = buildFormData();
  formData.append("audio", blob, filename);

  if (!options.keepBusy) {
    setBusy(true, "transcribing");
  }

  const response = await fetch("/api/transcribe", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || "Transcription failed.");
  }
  return data;
}

function buildFormData() {
  const formData = new FormData();
  formData.append("model", document.querySelector("#model").value.trim());
  formData.append("device", document.querySelector("#device").value);
  formData.append("language", document.querySelector("#language").value);
  formData.append("task", document.querySelector("#task").value);
  formData.append("max_new_tokens", document.querySelector("#maxTokens").value);
  formData.append("duration", document.querySelector("#duration").value);
  formData.append("timestamps", document.querySelector("#timestamps").checked ? "true" : "false");
  return formData;
}

function appendLiveResult(data, chunkNumber) {
  els.transcript.classList.remove("error");
  const text = data.text?.trim();
  if (text) {
    els.transcript.textContent += `${els.transcript.textContent ? "\n" : ""}[${chunkNumber}] ${text}`;
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }
  renderMetrics(data, `chunk ${chunkNumber}`);
}

function renderMetrics(data, label) {
  const benchmark = data.benchmark || {};
  els.metrics.innerHTML = [
    metric("source", label),
    metric("device", data.selected_device),
    metric("audio sec", formatNumber(benchmark.audio_duration_seconds)),
    metric("rtf", benchmark.rtf == null ? "n/a" : formatNumber(benchmark.rtf)),
    metric("load sec", formatNumber(benchmark.model_load_seconds)),
    metric("infer sec", formatNumber(benchmark.inference_seconds)),
    metric("total sec", formatNumber(benchmark.total_processing_seconds)),
  ].join("");
}

function metric(label, value) {
  return `<div class="metric"><b>${label}</b><span>${value ?? ""}</span></div>`;
}

function formatNumber(value) {
  return Number(value).toFixed(4);
}

function showError(message) {
  setBusy(false, "error");
  els.transcript.classList.add("error");
  els.transcript.textContent = message;
  els.liveLog.textContent = "error";
}

function clearOutput() {
  els.transcript.classList.remove("error");
  els.transcript.textContent = "";
  els.metrics.innerHTML = "";
}

function setBusy(isBusy, status) {
  els.runStatus.textContent = status;
  els.startLiveButton.disabled = isBusy || state.liveRunning;
  els.stopLiveButton.disabled = !state.liveRunning;
}

function getChunkMs() {
  const seconds = Number(document.querySelector("#duration").value || 4);
  return Math.max(1000, seconds * 1000);
}

function mergeFloat32(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function resampleLinear(input, fromRate, toRate) {
  if (fromRate === toRate) {
    return input;
  }
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const index = i * ratio;
    const before = Math.floor(index);
    const after = Math.min(before + 1, input.length - 1);
    const weight = index - before;
    output[i] = input[before] * (1 - weight) + input[after] * weight;
  }
  return output;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += 2;
  }
  return view;
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}
