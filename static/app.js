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
  modelChoices: document.querySelector("#modelChoices"),
  modelInput: document.querySelector("#model"),
  customModelInput: document.querySelector("#customModel"),
  customModelField: document.querySelector("#customModelField"),
};

els.startLiveButton.addEventListener("click", startLive);
els.stopLiveButton.addEventListener("click", stopLive);
els.clearButton.addEventListener("click", clearOutput);
els.modelChoices.addEventListener("click", handleModelButtonClick);
els.customModelInput.addEventListener("input", handleCustomModelInput);
document.querySelector("#device").addEventListener("change", resetWarmup);

updateCustomModelVisibility();
loadModels();
loadDevices();

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    updateModelSelect(data.models || []);
  } catch {
    updateModelSelect([]);
  }
}

function updateModelSelect(models) {
  if (!models.length) {
    return;
  }

  const currentValue = getSelectedModel();
  els.modelChoices.innerHTML = "";

  for (const model of models) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    button.dataset.model = model.value;
    button.textContent = model.label;
    button.title = model.description || model.value;
    els.modelChoices.appendChild(button);
  }

  const customButton = document.createElement("button");
  customButton.type = "button";
  customButton.className = "model-option";
  customButton.dataset.model = "custom";
  customButton.textContent = "Custom";
  customButton.title = "Custom model id or directory";
  els.modelChoices.appendChild(customButton);

  const knownModel = models.find((model) => model.value === currentValue);
  selectModelButton(knownModel ? knownModel.value : "custom");
  if (!knownModel) {
    els.customModelInput.value = currentValue;
  }
  setSelectedModel(currentValue);
}

function handleModelButtonClick(event) {
  const button = event.target.closest(".model-option");
  if (!button) {
    return;
  }

  selectModelButton(button.dataset.model);
  if (button.dataset.model !== "custom") {
    setSelectedModel(button.dataset.model);
  } else {
    setSelectedModel(els.customModelInput.value.trim());
  }
  loadDevices();
  resetWarmup();
}

function handleCustomModelInput() {
  if (getActiveModelChoice() === "custom") {
    setSelectedModel(els.customModelInput.value.trim());
    loadDevices();
    resetWarmup();
  }
}

function selectModelButton(modelValue) {
  for (const button of els.modelChoices.querySelectorAll(".model-option")) {
    const isActive = button.dataset.model === modelValue;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
  updateCustomModelVisibility();
}

function getActiveModelChoice() {
  return els.modelChoices.querySelector(".model-option.active")?.dataset.model || els.modelInput.value;
}

function setSelectedModel(modelValue) {
  els.modelInput.value = modelValue;
}

function updateCustomModelVisibility() {
  els.customModelField.classList.toggle("hidden", getActiveModelChoice() !== "custom");
}

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
      : "Moonshine devices unavailable";
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
    state.processor = state.audioContext.createScriptProcessor(2048, 1, 1);
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
  if (model.includes("Qwen3-ASR")) {
    return `The first start downloads, converts, and warms up ${model}. Intel NPU/GPU uses OpenVINO IR saved under cache\\openvino.`;
  }
  return `The first start downloads and warms up ${model}. Intel NPU/GPU uses OpenVINO when supported and saves converted model files under cache\\openvino for faster startup after restart.`;
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
  if (audioRms(resampled) < 0.003) {
    els.liveLog.textContent = state.liveRunning ? "listening" : "stopped";
    return;
  }
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
  formData.append("model", getSelectedModel());
  formData.append("device", document.querySelector("#device").value);
  formData.append("max_new_tokens", document.querySelector("#maxTokens").value);
  formData.append("duration", document.querySelector("#duration").value);
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

function getChunkMs() {
  const seconds = Number(document.querySelector("#duration").value || 1);
  return Math.max(500, seconds * 1000);
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
