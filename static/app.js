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
  chunks: [],
  chunkSampleCount: 0,
  hasSpeech: false,
  silenceMs: 0,
  flushInProgress: false,
  submitChain: Promise.resolve(),
  chunkIndex: 0,
};

const chunking = {
  speechRmsThreshold: 0.01,
  submitRmsThreshold: 0.003,
  minChunkMs: 350,
  trailingSilenceMs: 280,
  maxChunkMs: 2400,
  idleDropMs: 900,
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
  modelInput: document.querySelector("#model"),
};

els.startLiveButton.addEventListener("click", startLive);
els.stopLiveButton.addEventListener("click", stopLive);
els.clearButton.addEventListener("click", clearOutput);
document.querySelector("#device").addEventListener("change", resetWarmup);

loadDevices();

function getSelectedModel() {
  return els.modelInput.value.trim();
}

async function loadDevices() {
  try {
    const response = await fetch(`/api/devices?model=${encodeURIComponent(getSelectedModel())}`);
    const data = await response.json();
    updateDeviceSelect(data.choices || []);
    const autoChoice = (data.choices || []).find((choice) => choice.value === "auto");
    els.deviceStatus.textContent = data.devices?.length
      ? `Device: ${autoChoice?.target || data.devices[0]}`
      : "ASR devices unavailable";
  } catch {
    els.deviceStatus.textContent = "device check failed";
  }
}

function updateDeviceSelect(choices) {
  const deviceSelect = document.querySelector("#device");
  const currentValue = deviceSelect.value || "auto";
  deviceSelect.innerHTML = "";

  for (const choice of choices) {
    const option = document.createElement("option");
    option.value = choice.value;
    option.textContent = choice.available ? choice.label : `${choice.label} unavailable`;
    option.title = choice.reason || choice.target || choice.label;
    option.disabled = !choice.available;
    deviceSelect.appendChild(option);
  }

  const hasCurrentValue = [...deviceSelect.options].some(
    (option) => option.value === currentValue && !option.disabled,
  );
  deviceSelect.value = hasCurrentValue ? currentValue : "auto";
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
    state.processor = state.audioContext.createScriptProcessor(1024, 1, 1);
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 256;
    resetChunkState();
    state.submitChain = Promise.resolve();

    state.source.connect(state.analyser);
    state.analyser.connect(state.processor);
    state.processor.connect(state.audioContext.destination);
    state.processor.onaudioprocess = (event) => {
      if (state.liveRunning) {
        collectAudioFrame(event.inputBuffer.getChannelData(0));
      }
    };

    state.liveRunning = true;
    els.stopLiveButton.disabled = false;
    els.liveLog.textContent = "listening";
    setBusy(false, "listening");
    startMeter();
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
      model: getSelectedModel(),
      device: document.querySelector("#device").value,
    }),
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || "Warmup failed.");
  }

  state.warmed = true;
  els.warmupNotice.textContent = data.cache_hit
    ? `Warm cache ready on ${formatWarmupDevice(data)}. Cache: ${data.cache_dir}`
    : `Warmup complete on ${formatWarmupDevice(data)} in ${formatNumber(data.model_load_seconds)}s. Cache saved to ${data.cache_dir}`;
  els.liveLog.textContent = "warmup complete";
}

function formatWarmupDevice(data) {
  return data.fallback_device ? `${data.selected_device} + ${data.fallback_device} fallback` : data.selected_device;
}

function resetWarmup() {
  state.warmed = false;
  els.warmupNotice.classList.remove("active");
  els.warmupNotice.textContent = warmupNoticeText();
}

function warmupNoticeText() {
  const model = getSelectedModel();
  return `The first start downloads, converts, and warms up the Japanese model ${model}. Intel NPU/GPU uses OpenVINO IR saved under cache\\openvino.`;
}

async function stopLive() {
  if (!state.liveRunning) {
    return;
  }

  state.liveRunning = false;
  setBusy(true, "stopping");
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
  if (!state.chunks.length || !state.audioContext || state.flushInProgress) {
    return;
  }

  state.flushInProgress = true;
  const chunks = state.chunks;
  resetChunkState();
  state.flushInProgress = false;
  const chunkNumber = state.chunkIndex + 1;
  state.chunkIndex = chunkNumber;
  const merged = mergeFloat32(chunks);
  const resampled = resampleLinear(merged, state.audioContext.sampleRate, 16000);
  if (audioRms(resampled) < 0.003) {
    els.liveLog.textContent = state.liveRunning ? "listening" : "stopped";
    return;
  }
  const wav = encodeWav(resampled, 16000);
  const blob = new Blob([wav], { type: "audio/wav" });

  state.submitChain = state.submitChain.catch(() => {}).then(async () => {
    els.liveLog.textContent = "transcribing";
    const data = await submitAudio(blob, `live-${chunkNumber}.wav`, { keepBusy: true });
    appendLiveResult(data);
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
  formData.append("model", getSelectedModel());
  formData.append("device", document.querySelector("#device").value);
  return formData;
}

function appendLiveResult(data) {
  els.transcript.classList.remove("error");
  const text = extractAsrText(data.text);
  if (text) {
    els.transcript.textContent += `${els.transcript.textContent ? "\n" : ""}${text}`;
    els.transcript.scrollTop = els.transcript.scrollHeight;
    renderMetrics(data, "latest");
  }
}

function extractAsrText(rawText) {
  let text = String(rawText || "").trim();
  if (!text) {
    return "";
  }

  const asrTag = "<asr_text>";
  const tagIndex = text.indexOf(asrTag);
  if (tagIndex >= 0) {
    text = text.slice(tagIndex + asrTag.length);
  }

  return text
    .replace(/<\/?asr_text>/gi, "")
    .replace(/^language\s+\S+\s*/i, "")
    .replace(/<\|[^|]+?\|>/g, "")
    .trim();
}

function renderMetrics(data, label) {
  const benchmark = data.benchmark || {};
  els.metrics.innerHTML = [
    metric("status", label),
    metric("device", data.fallback_device ? `${data.selected_device} + ${data.fallback_device}` : data.selected_device),
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

function collectAudioFrame(input) {
  const frame = new Float32Array(input);
  state.chunks.push(frame);
  state.chunkSampleCount += frame.length;

  const sampleRate = state.audioContext.sampleRate;
  const frameMs = (frame.length / sampleRate) * 1000;
  const chunkMs = (state.chunkSampleCount / sampleRate) * 1000;
  const frameRms = audioRms(frame);

  if (frameRms >= chunking.speechRmsThreshold) {
    state.hasSpeech = true;
    state.silenceMs = 0;
  } else if (state.hasSpeech) {
    state.silenceMs += frameMs;
  }

  const endedBySilence =
    state.hasSpeech &&
    chunkMs >= chunking.minChunkMs &&
    state.silenceMs >= chunking.trailingSilenceMs;
  const endedByMaxLength = state.hasSpeech && chunkMs >= chunking.maxChunkMs;
  if (endedBySilence || endedByMaxLength) {
    void flushLiveChunk();
    return;
  }

  if (!state.hasSpeech && chunkMs >= chunking.idleDropMs) {
    resetChunkState();
  }
}

function resetChunkState() {
  state.chunks = [];
  state.chunkSampleCount = 0;
  state.hasSpeech = false;
  state.silenceMs = 0;
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

function audioRms(samples) {
  if (!samples.length) {
    return 0;
  }
  let squareSum = 0;
  for (const sample of samples) {
    squareSum += sample * sample;
  }
  return Math.sqrt(squareSum / samples.length);
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
