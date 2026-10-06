// Realtime ASR Pipeline Client Script (v3.5)
let ws = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;

// ==============================================================================
// 0. DOM ELEMENTS & SYSTEM REFERENCES
// ==============================================================================
const wsStatusBadge = document.getElementById("wsStatusBadge");
const wsStatusText = document.getElementById("wsStatusText");
const interimTextBox = document.getElementById("interimTextBox");
const transcriptFeed = document.getElementById("transcriptFeed");
const meterBar = document.getElementById("meterBar");
const recordTimer = document.getElementById("recordTimer");
const btnStartRecord = document.getElementById("btnStartRecord");
const btnStopRecord = document.getElementById("btnStopRecord");
const btnCancelRecord = document.getElementById("btnCancelRecord");
const btnClearTranscript = document.getElementById("btnClearTranscript");
const btnApplyToggles = document.getElementById("btnApplyToggles");

// Toggles
const toggleDenoise = document.getElementById("toggleDenoise");
const toggleAsr = document.getElementById("toggleAsr");
const toggleDiarization = document.getElementById("toggleDiarization");
const toggleVerification = document.getElementById("toggleVerification");
const toggleHotwords = document.getElementById("toggleHotwords");
const togglePunctuation = document.getElementById("togglePunctuation");

// Sidebar Toggle Progress & Alert
const toggleProgressWrapper = document.getElementById("toggleProgressWrapper");
const toggleProgressBar = document.getElementById("toggleProgressBar");
const toggleAlert = document.getElementById("toggleAlert");

// Tab 1: Microphone Finalization Progress & Alert
const micProgressCard = document.getElementById("micProgressCard");
const micProgressBar = document.getElementById("micProgressBar");
const micProgressTitle = document.getElementById("micProgressTitle");
const micProgressSubStatus = document.getElementById("micProgressSubStatus");
const micProgressSpinner = document.getElementById("micProgressSpinner");
const micProgressPercentBadge = document.getElementById("micProgressPercentBadge");
const micAlertBanner = document.getElementById("micAlertBanner");

// Tab 1: Microphone Audio Preview & Recorded Audio Buffer
const micAudioPreviewWrapper = document.getElementById("micAudioPreviewWrapper");
const micAudioPreview = document.getElementById("micAudioPreview");
const btnDownloadMicAudio = document.getElementById("btnDownloadMicAudio");
let recordedPcmChunks = [];
let recordedBlobUrl = null;

// Tab 2: File Upload & Offline Stepper
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileSelectedInfo = document.getElementById("fileSelectedInfo");
const selectedFileName = document.getElementById("selectedFileName");
const btnProcessFile = document.getElementById("btnProcessFile");
const btnCancelProcess = document.getElementById("btnCancelProcess");
const mediaPreviewWrapper = document.getElementById("mediaPreviewWrapper");
const videoPreview = document.getElementById("videoPreview");
const audioPreview = document.getElementById("audioPreview");
const mediaTypeIcon = document.getElementById("mediaTypeIcon");
const selectedFileSize = document.getElementById("selectedFileSize");
const offlineProgressCard = document.getElementById("offlineProgressCard");
const btnCancelProcessHeader = document.getElementById("btnCancelProcessHeader");
const offlineProgressBar = document.getElementById("offlineProgressBar");
const progressPercentBadge = document.getElementById("progressPercentBadge");
const progressMainTitle = document.getElementById("progressMainTitle");
const progressSubStatus = document.getElementById("progressSubStatus");
const progressSpinner = document.getElementById("progressSpinner");
const statElapsed = document.getElementById("statElapsed");
const statDuration = document.getElementById("statDuration");
const statSpeed = document.getElementById("statSpeed");
const offlineAlertBanner = document.getElementById("offlineAlertBanner");

// Modals: Speaker Enrollment, Hotwords & Model Settings
const modalSpeakers = document.getElementById("modalSpeakers");
const modalHotwords = document.getElementById("modalHotwords");
const modalModelSettings = document.getElementById("modalModelSettings");
const modelSettingsProgressWrapper = document.getElementById("modelSettingsProgressWrapper");
const modelSettingsProgressBar = document.getElementById("modelSettingsProgressBar");
const modelSettingsProgressStatus = document.getElementById("modelSettingsProgressStatus");
const modelSettingsAlert = document.getElementById("modelSettingsAlert");
const modelSettingsGroupsContainer = document.getElementById("modelSettingsGroupsContainer");
const formModelSettings = document.getElementById("formModelSettings");
const btnResetModelSettings = document.getElementById("btnResetModelSettings");

const enrollProgressWrapper = document.getElementById("enrollProgressWrapper");
const enrollProgressBar = document.getElementById("enrollProgressBar");
const enrollProgressStatus = document.getElementById("enrollProgressStatus");
const speakerModalAlert = document.getElementById("speakerModalAlert");
const hotwordProgressWrapper = document.getElementById("hotwordProgressWrapper");
const hotwordProgressBar = document.getElementById("hotwordProgressBar");
const hotwordProgressStatus = document.getElementById("hotwordProgressStatus");
const hotwordModalAlert = document.getElementById("hotwordModalAlert");

let currentModelConfig = {};
let modelSettingsMeta = null;

let selectedAudioFile = null;
let lastProcessedResult = null;
let offlineProgressTimer = null;
let offlineStartTime = null;
let currentOfflineAbortController = null;
let currentOfflineTaskId = null;
let isOfflineCancelled = false;

// ==============================================================================
// 1. UNIFIED NOTIFICATION (TOAST) & INLINE ALERT SYSTEM
// ==============================================================================
function showNotification(message, type = "info", duration = 4000, customTitle = null) {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const iconMap = {
    success: "fa-solid fa-circle-check",
    error: "fa-solid fa-circle-xmark",
    info: "fa-solid fa-circle-info",
    warning: "fa-solid fa-triangle-exclamation"
  };

  const defaultTitles = {
    success: "Thành công",
    error: "Lỗi xử lý",
    info: "Thông báo",
    warning: "Cảnh báo"
  };

  const titleText = customTitle || defaultTitles[type] || "Thông báo";
  const iconClass = iconMap[type] || iconMap.info;

  toast.innerHTML = `
    <i class="${iconClass} toast-icon"></i>
    <div class="toast-body">
      <div class="toast-title">${titleText}</div>
      <div class="toast-message">${message}</div>
    </div>
    <button class="toast-close" title="Đóng">&times;</button>
  `;

  const closeBtn = toast.querySelector(".toast-close");
  const dismiss = () => {
    toast.classList.add("toast-hiding");
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  };
  closeBtn.addEventListener("click", dismiss);

  if (duration > 0) {
    setTimeout(dismiss, duration);
  }

  container.appendChild(toast);
}

function showInlineAlert(containerId, message, type = "info") {
  const el = document.getElementById(containerId);
  if (!el) return;
  const iconMap = {
    success: "fa-solid fa-circle-check",
    error: "fa-solid fa-circle-exmark",
    info: "fa-solid fa-circle-info",
    warning: "fa-solid fa-triangle-exclamation"
  };
  el.className = `alert-banner alert-${type}`;
  el.innerHTML = `<i class="${iconMap[type] || iconMap.info}"></i> <span>${message}</span>`;
  el.style.display = "flex";
}

function hideInlineAlert(containerId) {
  const el = document.getElementById(containerId);
  if (el) {
    el.style.display = "none";
    el.innerHTML = "";
  }
}

// ==============================================================================
// 2. WEBSOCKET CONNECTION (REALTIME STREAMING)
// ==============================================================================
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio`;

  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    wsStatusBadge.className = "badge badge-connected";
    wsStatusText.textContent = "WebSocket: Đã kết nối";
    console.log("[WS] Đã kết nối thành công tới máy chủ.");
    sendToggleConfig();
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleServerMessage(data);
    } catch (err) {
      console.error("[WS] Lỗi parse JSON từ server:", err);
    }
  };

  ws.onclose = () => {
    wsStatusBadge.className = "badge badge-disconnected";
    wsStatusText.textContent = "WebSocket: Ngắt kết nối (Đang thử lại...)";
    setTimeout(initWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error("[WS] Lỗi WebSocket:", err);
  };
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatNaturalCase(str) {
  if (!str) return "";
  const trimmed = String(str).trim();
  if (!trimmed) return "";

  const hasUpper = /[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]/.test(trimmed);
  const hasLower = /[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]/.test(trimmed);

  if ((hasUpper && !hasLower) || (hasLower && !hasUpper)) {
    const lower = trimmed.toLowerCase();
    return lower.replace(/(^\s*|[.?!]\s+)([a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])/g, 
      (match, p1, p2) => p1 + p2.toUpperCase()
    );
  }
  return trimmed;
}

function sendToggleConfig() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const inlineParams = typeof getInlineModelParameters === "function" ? getInlineModelParameters() : {};
  const toggleRtDiar = document.getElementById("toggleRealtimeDiarization");
  const paramMaxWords = document.getElementById("param_max_recent_words_per_speaker");
  const paramInactTimeout = document.getElementById("param_speaker_inactivity_timeout_s");

  const config = {
    action: "configure",
    options: {
      denoise: toggleDenoise.checked,
      asr: toggleAsr.checked,
      diarization: toggleDiarization.checked,
      verification: toggleVerification.checked,
      hotwords: toggleHotwords.checked,
      punctuation: togglePunctuation.checked,
      realtime_diarization: toggleRtDiar ? toggleRtDiar.checked : true,
      max_recent_words_per_speaker: paramMaxWords ? parseInt(paramMaxWords.value, 10) : 25,
      speaker_inactivity_timeout_s: paramInactTimeout ? parseFloat(paramInactTimeout.value) : 7.0,
      ...inlineParams
    }
  };
  ws.send(JSON.stringify(config));
}

function renderLiveMultiSpeakers(speakers, container, countBadge) {
  if (!container || !speakers) return;

  const activeCount = speakers.filter(s => s.is_speaking).length;
  if (countBadge) {
    countBadge.textContent = `${activeCount} đang nói`;
    countBadge.className = activeCount > 0 ? "badge badge-success" : "badge badge-accent";
  }

  const existingCards = {};
  container.querySelectorAll(".live-speaker-card").forEach(card => {
    const spkId = card.getAttribute("data-speaker-id");
    if (spkId) existingCards[spkId] = card;
  });

  speakers.forEach(spk => {
    let card = existingCards[spk.speaker_id];
    const isExpired = spk.is_expired;
    const isSpeaking = spk.is_speaking;

    const initials = spk.name ? spk.name.trim().substring(0, 2).toUpperCase() : spk.speaker_id.substring(0, 2);
    const isVerified = spk.score >= 0.55 && !spk.name.startsWith("Chưa đăng ký") && !spk.name.startsWith("speaker_");
    const simBadgeClass = isVerified ? "similarity-badge verified" : "similarity-badge unverified";
    const simBadgeText = isVerified 
      ? `<i class="fa-solid fa-check"></i> ${Math.round(spk.score * 100)}% khớp` 
      : (spk.score > 0 ? `${Math.round(spk.score * 100)}%` : "Chưa đăng ký");

    const speakingTag = isSpeaking 
      ? `<span class="speaking-now-tag"><i class="fa-solid fa-waveform"></i> Đang nói</span>` 
      : `<span style="font-size: 10px; color: var(--text-muted);">${spk.last_active_ago}s trước</span>`;

    const formattedRecentText = formatNaturalCase(spk.recent_text);
    const textHtml = formattedRecentText 
      ? escapeHtml(formattedRecentText)
      : `<span class="speaker-bubble-empty">Chưa có phát biểu gần đây...</span>`;

    if (!card) {
      card = document.createElement("div");
      card.className = "live-speaker-card";
      card.setAttribute("data-speaker-id", spk.speaker_id);
      container.appendChild(card);
    }

    if (isSpeaking) {
      card.classList.add("is-speaking");
    } else {
      card.classList.remove("is-speaking");
    }

    if (isExpired) {
      card.classList.add("is-expired");
      card.style.display = "none";
    } else {
      card.classList.remove("is-expired");
      card.style.display = "flex";
    }

    card.innerHTML = `
      <div class="speaker-card-header">
        <div class="speaker-profile">
          <div class="speaker-avatar">${initials}</div>
          <div class="speaker-title-wrap">
            <span class="speaker-display-name">
              ${escapeHtml(spk.name)}
              ${isVerified ? '<i class="fa-solid fa-circle-check text-success" style="font-size: 12px; margin-left: 4px;" title="Đã xác thực Voiceprint"></i>' : ''}
            </span>
            <span class="speaker-meta-info">${escapeHtml(spk.speaker_id)} &bull; ${spk.total_words} từ</span>
          </div>
        </div>
        <div class="speaker-card-badges">
          <span class="${simBadgeClass}">${simBadgeText}</span>
          ${speakingTag}
        </div>
      </div>
      <div class="speaker-bubble">
        ${textHtml}
      </div>
      <div class="speaker-footer">
        <span><i class="fa-regular fa-clock"></i> Cập nhật: ${spk.last_active_ago}s trước</span>
        <span>${spk.is_speaking ? '<strong class="text-primary">Đang phát biểu</strong>' : 'Tạm dừng'}</span>
      </div>
    `;
  });
}

function handleCommittedTurnEvent(turnEvent) {
  if (!turnEvent || !transcriptFeed) return;

  const emptyState = transcriptFeed.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }

  const turnId = turnEvent.turn_id;
  const spkKey = turnEvent.raw_speaker || turnEvent.speaker || "speaker_0";
  const spkClass = getSpeakerClass(spkKey);
  const cardId = `turn_card_${turnId}`;

  if (turnEvent.action === "turn_new") {
    const card = document.createElement("div");
    card.id = cardId;
    card.className = `transcript-card ${spkClass} live-committed-turn`;
    card.innerHTML = `
      <div class="card-top">
        <span class="speaker-tag ${spkClass}"><i class="fa-solid fa-user"></i> ${escapeHtml(turnEvent.speaker)}</span>
        <span class="time-tag"><i class="fa-solid fa-circle-check" style="font-size:10px; color:var(--success);"></i> <span class="time-range">${turnEvent.start.toFixed(2)}s - ${turnEvent.end.toFixed(2)}s</span></span>
      </div>
      <p class="card-text">${escapeHtml(turnEvent.text)}</p>
    `;
    transcriptFeed.appendChild(card);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  } else if (turnEvent.action === "turn_update") {
    let card = document.getElementById(cardId);
    if (!card) {
      card = document.createElement("div");
      card.id = cardId;
      card.className = `transcript-card ${spkClass} live-committed-turn`;
      card.innerHTML = `
        <div class="card-top">
          <span class="speaker-tag ${spkClass}"><i class="fa-solid fa-user"></i> ${escapeHtml(turnEvent.speaker)}</span>
          <span class="time-tag"><i class="fa-solid fa-circle-check" style="font-size:10px; color:var(--success);"></i> <span class="time-range">${turnEvent.start.toFixed(2)}s - ${turnEvent.end.toFixed(2)}s</span></span>
        </div>
        <p class="card-text">${escapeHtml(turnEvent.text)}</p>
      `;
      transcriptFeed.appendChild(card);
    } else {
      const cardText = card.querySelector(".card-text");
      if (cardText) {
        cardText.textContent = turnEvent.text;
      }
      const timeRange = card.querySelector(".time-range");
      if (timeRange) {
        timeRange.textContent = `${turnEvent.start.toFixed(2)}s - ${turnEvent.end.toFixed(2)}s`;
      }
    }
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }
}

function handleServerMessage(data) {
  if (data.type === "partial") {
    // 1. Cập nhật 2-Pass ZipFormer-150M committed turns theo thời gian thực
    if (data.turn_event) {
      handleCommittedTurnEvent(data.turn_event);
    }

    // 2. Hiển thị chữ tức thì (không in hoa gắt gỏng)
    if (typeof data.text !== "undefined") {
      latestActiveSpkBadge = (data.realtime_diarization && data.active_speaker)
        ? `<span class="badge badge-accent" style="margin-right: 8px; vertical-align: middle;"><i class="fa-solid fa-user"></i> ${escapeHtml(data.active_speaker)}</span>`
        : "";
      
      latestInterimRawText = data.text || "";
      renderInterimText();
    }

    // Hiển thị và cập nhật bảng Multi-Speaker Live
    const liveMultiSpeakerPanel = document.getElementById("liveMultiSpeakerPanel");
    const liveSpeakersGrid = document.getElementById("liveSpeakersGrid");
    const liveActiveCountBadge = document.getElementById("liveActiveSpeakerCountBadge");

    if (data.realtime_diarization && data.speakers && data.speakers.length > 0) {
      if (liveMultiSpeakerPanel) liveMultiSpeakerPanel.style.display = "block";
      renderLiveMultiSpeakers(data.speakers, liveSpeakersGrid, liveActiveCountBadge);
    } else {
      if (liveMultiSpeakerPanel) liveMultiSpeakerPanel.style.display = "none";
    }
  } else if (data.type === "final") {
    // Mở lại nút Bắt đầu thu âm sau khi server đã hoàn tất chốt phiên
    btnStartRecord.disabled = false;

    // Nhận kết quả chốt câu (Final transcript)
    if (micProgressCard && micProgressCard.style.display !== "none") {
      micProgressBar.className = "progress-bar-fill";
      micProgressBar.style.width = "100%";
      micProgressBar.style.background = "var(--success)";
      micProgressTitle.textContent = "Xử lý âm thanh hoàn tất!";
      micProgressSubStatus.textContent = "Đã phân tách người nói và chuẩn hóa dấu câu thành công.";
      micProgressSpinner.className = "spinner-pulse done";
      micProgressSpinner.innerHTML = '<i class="fa-solid fa-check"></i>';
      micProgressPercentBadge.textContent = "100%";
      showInlineAlert("micAlertBanner", "Nhận diện & phân tích giọng nói hoàn tất thành công!", "success");
      setTimeout(() => {
        if (micProgressCard) micProgressCard.style.display = "none";
      }, 5000);
    }

    const durationInfo = data.total_duration_s ? ` (Thời lượng: ${data.total_duration_s.toFixed(1)}s)` : "";
    showNotification(`Đã nhận diện và chốt hội thoại thành công!${durationInfo}`, "success", 4000);

    // Cập nhật đường dẫn tải và nghe lại âm thanh microphone đã lưu
    if (data.audio_url) {
      if (btnDownloadMicAudio) {
        btnDownloadMicAudio.href = data.audio_url;
        btnDownloadMicAudio.download = "recording_online.wav";
      }
      if (recordedBlobUrl) {
        URL.revokeObjectURL(recordedBlobUrl);
        recordedBlobUrl = null;
      }
      if (micAudioPreview) {
        micAudioPreview.src = data.audio_url;
        micAudioPreview.load();
      }
      if (micAudioPreviewWrapper) {
        micAudioPreviewWrapper.style.display = "block";
      }
    }

    data.source = "mic";
    renderFinalResult(data);
    resetInterimState();
    interimTextBox.innerHTML = `<em>(Đã hoàn tất chốt hội thoại. Sẵn sàng cho lượt thu âm mới!)</em>`;
  } else if (data.type === "error") {
    // Mở lại nút Bắt đầu thu âm nếu gặp lỗi
    btnStartRecord.disabled = false;

    if (micProgressCard && micProgressCard.style.display !== "none") {
      micProgressBar.className = "progress-bar-fill";
      micProgressBar.style.background = "var(--danger)";
      micProgressTitle.textContent = "Gặp lỗi khi xử lý âm thanh!";
      micProgressSubStatus.textContent = data.detail || "Lỗi không xác định từ pipeline";
      showInlineAlert("micAlertBanner", "Lỗi: " + (data.detail || "Không xác định"), "error");
    }
    showNotification("Lỗi xử lý âm thanh microphone: " + (data.detail || "Không xác định"), "error", 6000);
  }
}

// ==============================================================================
// 3. MICROPHONE STREAMING (LIVE RECORDING)
// ==============================================================================
// Helper: Chuyển đổi danh sách mảng Float32 thành file WAV Blob 16-bit 16kHz
function exportWavBlob(chunks, sampleRate = 16000) {
  let totalLength = 0;
  for (let i = 0; i < chunks.length; i++) {
    totalLength += chunks[i].length;
  }
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (let i = 0; i < chunks.length; i++) {
    merged.set(chunks[i], offset);
    offset += chunks[i].length;
  }

  const buffer = new ArrayBuffer(44 + merged.length * 2);
  const view = new DataView(buffer);

  function writeStr(pos, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(pos + i, str.charCodeAt(i));
    }
  }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + merged.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM Format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate (16000 * 1 * 2)
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // 16 bits per sample
  writeStr(36, "data");
  view.setUint32(40, merged.length * 2, true);

  let pcmOffset = 44;
  for (let i = 0; i < merged.length; i++) {
    let s = Math.max(-1, Math.min(1, merged[i]));
    view.setInt16(pcmOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    pcmOffset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

async function startRecording() {
  if (micProgressCard) micProgressCard.style.display = "none";
  hideInlineAlert("micAlertBanner");

  // Làm sạch màn hình và reset dữ liệu phiên cũ khi bắt đầu một phiên thu âm mới
  transcriptFeed.innerHTML = "";
  speakerColorMap.clear();
  lastProcessedResult = null;

  // Dọn dẹp bản ghi cũ khi bắt đầu một phiên xử lý mới
  recordedPcmChunks = [];
  resetInterimState();
  if (recordedBlobUrl) {
    URL.revokeObjectURL(recordedBlobUrl);
    recordedBlobUrl = null;
  }
  if (micAudioPreview) {
    micAudioPreview.pause();
    micAudioPreview.src = "";
  }
  if (micAudioPreviewWrapper) {
    micAudioPreviewWrapper.style.display = "none";
  }
  // Gửi yêu cầu xóa file ghi âm cũ trên server
  fetch("/api/recording/latest", { method: "DELETE" }).catch(() => {});

  // Gửi tín hiệu khởi tạo phiên mới tới WebSocket
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "start" }));
  }

  // Dọn dẹp danh sách thẻ người nói trực tiếp cũ
  const liveGrid = document.getElementById("liveSpeakersGrid");
  if (liveGrid) liveGrid.innerHTML = "";
  const liveActiveBadge = document.getElementById("liveActiveSpeakerCountBadge");
  if (liveActiveBadge) {
    liveActiveBadge.textContent = "0 đang nói";
    liveActiveBadge.className = "badge badge-accent";
  }
  const livePanel = document.getElementById("liveMultiSpeakerPanel");
  const tgRealtime = document.getElementById("toggleRealtimeDiarization");
  if (livePanel) {
    livePanel.style.display = (tgRealtime && tgRealtime.checked) ? "block" : "none";
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: false,
        noiseSuppression: false, // Để DeepFilterNet3 xử lý khử nhiễu
        autoGainControl: false,
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000
    });

    const source = audioContext.createMediaStreamSource(mediaStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;

      const inputData = e.inputBuffer.getChannelData(0);

      // Lưu trữ mảng PCM thô vào buffer để phát lại âm thanh
      recordedPcmChunks.push(new Float32Array(inputData));

      // Tính toán mức âm lượng (Audio Level Meter)
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      const levelPercent = Math.min(100, Math.round(rms * 400));
      meterBar.style.width = `${levelPercent}%`;

      // Chuyển Float32 [-1.0, 1.0] -> 16-bit PCM Int16 Little-Endian
      const pcmBuffer = new ArrayBuffer(inputData.length * 2);
      const view = new DataView(pcmBuffer);
      for (let i = 0; i < inputData.length; i++) {
        let s = Math.max(-1, Math.min(1, inputData[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }

      // Gửi binary PCM qua WebSocket
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcmBuffer);
      }
    };

    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    isRecording = true;
    sendToggleConfig();
    recordStartTime = Date.now();
    recordTimer.textContent = "00:00";
    timerInterval = setInterval(updateTimer, 1000);

    btnStartRecord.disabled = true;
    btnStopRecord.disabled = false;
    if (btnCancelRecord) btnCancelRecord.style.display = "inline-flex";
    interimTextBox.innerHTML = `<em>(Đang lắng nghe âm thanh từ microphone...)</em>`;
    showNotification("Đã bắt đầu thu âm từ Microphone. Hãy bắt đầu nói!", "info", 3000);

  } catch (err) {
    console.error("[Mic] Không thể truy cập Microphone:", err);
    showNotification("Không thể mở Microphone: " + err.message + ". Vui lòng cấp quyền truy cập!", "error", 6000);
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  clearInterval(timerInterval);
  resetInterimState();
  meterBar.style.width = "0%";

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  btnStartRecord.disabled = true;
  btnStopRecord.disabled = true;
  if (btnCancelRecord) btnCancelRecord.style.display = "none";

  // Xuất file WAV Blob tức thì từ dữ liệu mic đã thu để người dùng có thể nghe lại ngay trong lúc đang xử lý
  if (recordedPcmChunks.length > 0) {
    try {
      if (recordedBlobUrl) {
        URL.revokeObjectURL(recordedBlobUrl);
        recordedBlobUrl = null;
      }
      const wavBlob = exportWavBlob(recordedPcmChunks, 16000);
      recordedBlobUrl = URL.createObjectURL(wavBlob);
      if (micAudioPreview) {
        micAudioPreview.src = recordedBlobUrl;
      }
      if (btnDownloadMicAudio) {
        btnDownloadMicAudio.href = recordedBlobUrl;
        btnDownloadMicAudio.download = "recording_online.wav";
      }
      if (micAudioPreviewWrapper) {
        micAudioPreviewWrapper.style.display = "block";
      }
    } catch (wavErr) {
      console.warn("[Mic] Lỗi tạo WAV Blob:", wavErr);
    }
  }

  // Hiển thị thanh tiến trình chốt âm thanh trong Tab 1
  if (micProgressCard) {
    micProgressCard.style.display = "block";
    micProgressBar.className = "progress-bar-fill progress-indeterminate";
    micProgressBar.style.width = "100%";
    micProgressBar.style.background = "";
    micProgressTitle.textContent = "Đang chốt âm thanh & phân tích người nói...";
    micProgressSubStatus.textContent = "Khởi chạy Diarization, TitaNet Verification & ViBERT-capu...";
    micProgressSpinner.className = "spinner-pulse";
    micProgressSpinner.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
    micProgressPercentBadge.textContent = "Đang phân tích...";
    hideInlineAlert("micAlertBanner");
  }

  showNotification("Đã dừng thu âm. Đang xử lý phân tích người nói & định danh...", "info", 3500);

  // Gửi lệnh kết thúc tới WebSocket để chốt kết quả
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "finish" }));
  } else {
    btnStartRecord.disabled = false;
    showNotification("WebSocket đã bị ngắt kết nối!", "error", 5000);
  }
}

function cancelRecording() {
  if (!isRecording) return;
  isRecording = false;

  clearInterval(timerInterval);
  resetInterimState();
  meterBar.style.width = "0%";

  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  // Xóa sạch buffer và URL blob đã thu
  recordedPcmChunks = [];
  if (recordedBlobUrl) {
    URL.revokeObjectURL(recordedBlobUrl);
    recordedBlobUrl = null;
  }
  if (micAudioPreview) {
    micAudioPreview.pause();
    micAudioPreview.src = "";
  }
  if (micAudioPreviewWrapper) {
    micAudioPreviewWrapper.style.display = "none";
  }
  fetch("/api/recording/latest", { method: "DELETE" }).catch(() => {});

  btnStartRecord.disabled = false;
  btnStopRecord.disabled = true;
  if (btnCancelRecord) btnCancelRecord.style.display = "none";

  if (micProgressCard) {
    micProgressCard.style.display = "none";
  }

  // Gửi lệnh reset tới server để giải phóng bộ đệm âm thanh đang stream
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "reset" }));
  }

  recordTimer.textContent = "00:00";
  interimTextBox.innerHTML = `<em>(Đã hủy phiên thu âm. Bấm "Bắt Đầu Thu Âm" để nói lại.)</em>`;
  showNotification("Đã hủy phiên thu âm từ Microphone!", "warning", 3500);
}

function updateTimer() {
  const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
  const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
  const secs = String(elapsedSec % 60).padStart(2, "0");
  recordTimer.textContent = `${mins}:${secs}`;
}

// Quản lý màu sắc đồng nhất cho từng người nói
const speakerColorMap = new Map();
const colorPalette = ["spk-0", "spk-1", "spk-2", "spk-3", "spk-4", "spk-5"];

let latestInterimRawText = "";
let latestActiveSpkBadge = "";
let interimCutoffIndex = 0;       // Vị trí từ đã bị cắt/ẩn đi (do đủ maxWords hoặc do idle timeout)
let lastInterimRawContent = "";    // Nội dung thô của lần trước để phát hiện từ mới
let lastWordAddedTimestamp = 0;   // Thời điểm nhận được từ mới gần nhất
let interimIdleTimer = null;       // Timer đếm ngược m giây không thêm từ mới
let isInterimIdleHidden = false;   // Cờ đánh dấu đang ẩn do quá m giây không thêm từ mới

function getInterimIdleTimeoutMs() {
  const idleEl = document.getElementById("param_interim_idle_timeout_s") || document.getElementById("param_interim_timeout_s");
  const val = parseFloat(idleEl?.value || 3);
  return Math.max(0.5, isNaN(val) ? 3 : val) * 1000;
}

function resetInterimState() {
  latestInterimRawText = "";
  latestActiveSpkBadge = "";
  interimCutoffIndex = 0;
  lastInterimRawContent = "";
  lastWordAddedTimestamp = 0;
  isInterimIdleHidden = false;
  if (interimIdleTimer) {
    clearTimeout(interimIdleTimer);
    interimIdleTimer = null;
  }
}

function onInterimIdleTimeout() {
  interimIdleTimer = null;
  if (!isRecording) return;

  const text = (latestInterimRawText || "").trim();
  if (text) {
    const allWords = text.split(/\s+/);
    // Khi quá m giây không thêm từ mới, kết thúc lượt này bằng cách cắt toàn bộ từ hiện tại
    interimCutoffIndex = allWords.length;
  }
  isInterimIdleHidden = true;

  if (interimTextBox) {
    interimTextBox.innerHTML = `${latestActiveSpkBadge}<em>(Đang lắng nghe âm thanh từ microphone...)</em>`;
  }
}

function renderInterimText() {
  const text = (latestInterimRawText || "").trim();
  if (!text) {
    interimTextBox.innerHTML = `${latestActiveSpkBadge}<em>(Bắt đầu nói để thấy chữ xuất hiện theo thời gian thực...)</em>`;
    return;
  }

  const allWords = text.split(/\s+/);
  const total = allWords.length;

  if (total === 0) {
    interimTextBox.innerHTML = `${latestActiveSpkBadge}<em>(Đang đợi âm thanh...)</em>`;
    return;
  }

  // Đảm bảo interimCutoffIndex không vượt quá total (phòng ngừa reset từ server)
  if (interimCutoffIndex > total) {
    interimCutoffIndex = 0;
  }

  // Kiểm tra xem có từ mới hoặc nội dung có thay đổi so với lần trước không
  const hasNewContent = (text !== lastInterimRawContent);
  if (hasNewContent) {
    lastInterimRawContent = text;
    lastWordAddedTimestamp = Date.now();
    isInterimIdleHidden = false; // Có từ mới, bỏ trạng thái ẩn

    // Đặt lại idle timer: nếu quá m giây không có từ mới thì tự động ẩn đi
    if (interimIdleTimer) {
      clearTimeout(interimIdleTimer);
    }
    const idleTimeoutMs = getInterimIdleTimeoutMs();
    interimIdleTimer = setTimeout(onInterimIdleTimeout, idleTimeoutMs);
  }

  // Nếu đang ở trạng thái ẩn do quá m giây không có từ mới và chưa có từ mới xuất hiện
  if (isInterimIdleHidden) {
    interimTextBox.innerHTML = `${latestActiveSpkBadge}<em>(Đang lắng nghe âm thanh từ microphone...)</em>`;
    return;
  }

  const maxWordsEl = document.getElementById("param_interim_max_words");
  const maxWords = Math.max(1, parseInt(maxWordsEl?.value || 20, 10));

  // Số từ trong lượt hiện tại tính từ điểm cắt gần nhất
  const wordsInCurrentTurn = total - interimCutoffIndex;
  if (wordsInCurrentTurn <= 0) {
    interimTextBox.innerHTML = `${latestActiveSpkBadge}<em>(Đang đợi âm thanh...)</em>`;
    return;
  }

  // Cứ đủ maxWords chữ thì ẩn toàn bộ các chữ trước đó và hiển thị các chữ tiếp theo:
  const chunkOffset = Math.floor((wordsInCurrentTurn - 1) / maxWords) * maxWords;
  const displayStartIndex = interimCutoffIndex + chunkOffset;
  const currentChunk = allWords.slice(displayStartIndex);
  const chunkText = currentChunk.join(" ");
  const naturalText = formatNaturalCase(chunkText);

  interimTextBox.innerHTML = `${latestActiveSpkBadge}<strong>${escapeHtml(naturalText)}</strong>`;
}

function getSpeakerClass(speakerKey) {
  const key = (speakerKey || "speaker_0").trim();
  if (!speakerColorMap.has(key)) {
    const idx = speakerColorMap.size % colorPalette.length;
    speakerColorMap.set(key, colorPalette[idx]);
  }
  return speakerColorMap.get(key);
}

// ==============================================================================
// 4. HIỂN THỊ KẾT QUẢ FINAL TRANSCRIPT & VIDEO SYNC
// ==============================================================================
function getPlayerForResult(resultData) {
  // 1. Nếu là kết quả từ Microphone
  if (resultData && resultData.source === "mic") {
    return micAudioPreview;
  }

  // 2. Nếu là kết quả từ File Offline
  if (resultData && resultData.source === "offline") {
    const isVideo = videoPreview && videoPreview.style.display !== "none" && videoPreview.src && !videoPreview.error;
    if (isVideo) {
      return videoPreview;
    }
    return audioPreview;
  }

  // 3. Fallback theo tab giao diện hiện tại
  const isMicTab = document.getElementById("tab-mic") && document.getElementById("tab-mic").classList.contains("active");
  if (isMicTab) {
    return micAudioPreview;
  }
  const isVideo = videoPreview && videoPreview.style.display !== "none" && videoPreview.src && !videoPreview.error;
  return isVideo ? videoPreview : audioPreview;
}

function renderFinalResult(data) {
  lastProcessedResult = data;
  // Làm sạch màn hình hiển thị trước khi vẽ các thẻ kết quả chốt phiên
  transcriptFeed.innerHTML = "";

  // Hiển thị các nút xuất phụ đề
  const exportActions = document.getElementById("exportActions");
  if (exportActions) exportActions.style.display = "flex";

  // Quản lý media player theo nguồn dữ liệu
  if (data.source === "offline") {
    // Tạm dừng player mic để tránh phát chồng âm thanh
    if (micAudioPreview) micAudioPreview.pause();

    // Nếu server cung cấp audio_url chuẩn 16kHz đã trích xuất & khử nhiễu
    if (data.audio_url) {
      const isVideo = selectedAudioFile && (selectedAudioFile.type.startsWith("video/") || /\.(mp4|mkv|avi|mov|webm|flv|wmv)$/i.test(selectedAudioFile.name));
      const isUnsupportedVideo = selectedAudioFile && /\.(mkv|avi|flv|wmv)$/i.test(selectedAudioFile.name);

      // Nếu là audio hoặc video không được trình duyệt hỗ trợ trực tiếp (MKV, AVI, v.v.) hoặc video bị lỗi decode
      if (!isVideo || isUnsupportedVideo || (videoPreview && videoPreview.error)) {
        if (audioPreview) {
          audioPreview.src = data.audio_url;
          audioPreview.style.display = "block";
        }
        if (videoPreview) {
          videoPreview.style.display = "none";
          videoPreview.pause();
        }
        if (mediaPreviewWrapper) {
          mediaPreviewWrapper.style.display = "block";
        }
      }
    }
  } else if (data.source === "mic") {
    // Tạm dừng audio/video offline
    if (audioPreview) audioPreview.pause();
    if (videoPreview) videoPreview.pause();
  }

  const segments = data.segments || [];
  if (segments.length === 0) {
    if (data.final_text && data.final_text.trim()) {
      const spkClass = getSpeakerClass("speaker_0");
      const card = document.createElement("div");
      card.className = `transcript-card ${spkClass} clickable`;
      card.innerHTML = `
        <div class="card-top">
          <span class="speaker-tag ${spkClass}"><i class="fa-solid fa-user"></i> Người nói</span>
          <span class="time-tag"><i class="fa-solid fa-play"></i> 0.00s - ${data.total_duration_s ? data.total_duration_s.toFixed(2) : "0.00"}s</span>
        </div>
        <div class="card-text">${escapeHtml(formatNaturalCase(data.final_text))}</div>
      `;
      card.addEventListener("click", () => {
        if (data.source === "offline") {
          if (micAudioPreview) micAudioPreview.pause();
        } else {
          if (audioPreview) audioPreview.pause();
          if (videoPreview) videoPreview.pause();
        }

        const player = getPlayerForResult(data);
        if (player && (player.src || player.currentSrc)) {
          try {
            player.currentTime = 0;
            const p = player.play();
            if (p !== undefined) p.catch(e => console.warn("[Player] Lỗi play:", e));
          } catch (err) {
            console.warn("[Player] Lỗi seek:", err);
          }
          document.querySelectorAll(".transcript-card").forEach(c => c.style.outline = "none");
          card.style.outline = "2px solid var(--primary)";
        }
      });
      transcriptFeed.appendChild(card);
    } else {
      transcriptFeed.innerHTML = `
        <div class="empty-state">
          <i class="fa-solid fa-microphone-slash"></i>
          <p>Không phát hiện thấy tiếng nói trong lượt xử lý này.</p>
        </div>
      `;
    }
  } else {
    // Gom các phân đoạn liên tiếp của cùng 1 speaker thành 1 đoạn duy nhất
    const mergedSegments = [];
    segments.forEach((seg) => {
      const text = (seg.text || "").trim();
      if (!text) return; // Bỏ qua phân đoạn rỗng

      const spkKey = seg.raw_speaker || seg.speaker || "speaker_0";
      const spkName = seg.speaker || "Người nói";

      const prev = mergedSegments.length > 0 ? mergedSegments[mergedSegments.length - 1] : null;
      const canMerge = prev && (prev.raw_speaker === spkKey || prev.speaker === spkName);

      if (canMerge) {
        prev.end = Math.max(prev.end, seg.end);
        prev.text = `${prev.text} ${text}`.replace(/\s+/g, " ").trim();
      } else {
        mergedSegments.push({
          start: seg.start,
          end: seg.end,
          speaker: spkName,
          raw_speaker: spkKey,
          text: text
        });
      }
    });

    // Cập nhật lại segments đã gộp vào data để đồng bộ với việc xuất phụ đề SRT / VTT / TXT
    data.segments = mergedSegments;

    mergedSegments.forEach((seg) => {
      // Đảm bảo cùng 1 speaker luôn có cùng 1 màu sắc cố định
      const spkKey = seg.raw_speaker;
      const spkClass = getSpeakerClass(spkKey);

      const card = document.createElement("div");
      card.className = `transcript-card ${spkClass} clickable`;
      card.title = "Bấm để nghe âm thanh / xem video tại thời điểm này";
      card.innerHTML = `
        <div class="card-top">
          <span class="speaker-tag ${spkClass}"><i class="fa-solid fa-user"></i> ${seg.speaker}</span>
          <span class="time-tag"><i class="fa-solid fa-play" style="font-size:10px;"></i> ${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s</span>
        </div>
        <div class="card-text">${escapeHtml(formatNaturalCase(seg.text))}</div>
      `;

      // Đồng bộ Interactive Seek: Click vào thẻ để nhảy video / audio tới thời điểm nói
      card.addEventListener("click", () => {
        if (data.source === "offline") {
          if (micAudioPreview) micAudioPreview.pause();
        } else {
          if (audioPreview) audioPreview.pause();
          if (videoPreview) videoPreview.pause();
        }

        const player = getPlayerForResult(data);
        if (player && (player.src || player.currentSrc)) {
          try {
            player.currentTime = Math.max(0, seg.start);
            const playPromise = player.play();
            if (playPromise !== undefined) {
              playPromise.catch(e => console.warn("[Player] Không thể phát âm thanh:", e));
            }
          } catch (seekErr) {
            console.warn("[Player] Lỗi seek:", seekErr);
          }
          // Hiệu ứng nháy nhẹ thẻ đang chọn
          document.querySelectorAll(".transcript-card").forEach(c => c.style.outline = "none");
          card.style.outline = "2px solid var(--primary)";
        }
      });

      transcriptFeed.appendChild(card);
    });
  }

  transcriptFeed.scrollTop = 0;
}

// ==============================================================================
// 5. XỬ LÝ VIDEO & FILE ÂM THANH OFFLINE (PROGRESS STEPPER)
// ==============================================================================
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "var(--primary)";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "var(--card-border)";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "var(--card-border)";
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileSelected(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileSelected(e.target.files[0]);
  }
});

function handleFileSelected(file) {
  selectedAudioFile = file;
  const isVideo = file.type.startsWith("video/") || /\.(mp4|mkv|avi|mov|webm|flv)$/i.test(file.name);

  // Cập nhật thông tin file
  selectedFileName.textContent = file.name;
  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  selectedFileSize.textContent = `${sizeMb} MB • ${isVideo ? "Video" : "Audio"}`;
  fileSelectedInfo.style.display = "flex";

  if (isVideo) {
    mediaTypeIcon.className = "fa-solid fa-file-video text-primary";
    btnProcessFile.innerHTML = `<i class="fa-solid fa-play"></i> Bắt Đầu Xử Lý Video`;
  } else {
    mediaTypeIcon.className = "fa-solid fa-file-audio text-primary";
    btnProcessFile.innerHTML = `<i class="fa-solid fa-play"></i> Bắt Đầu Xử Lý Âm Thanh`;
  }

  // Khởi tạo Player Preview
  const fileUrl = URL.createObjectURL(file);
  mediaPreviewWrapper.style.display = "block";

  if (isVideo) {
    videoPreview.src = fileUrl;
    videoPreview.style.display = "block";
    audioPreview.style.display = "none";
    audioPreview.src = "";
  } else {
    audioPreview.src = fileUrl;
    audioPreview.style.display = "block";
    videoPreview.style.display = "none";
    videoPreview.src = "";
  }

  // Ẩn tiến trình cũ nếu có
  offlineProgressCard.style.display = "none";
  hideInlineAlert("offlineAlertBanner");
  showNotification(`Đã tải file "${file.name}" (${sizeMb} MB). Bấm "Bắt Đầu Xử Lý" để tiến hành!`, "info", 3500);
}

// Hàm cập nhật trạng thái Stepper
function updateStepperStep(stepKey, state, detailText) {
  const el = document.getElementById(`step-${stepKey}`);
  if (!el) return;

  el.classList.remove("pending", "active", "done", "skipped", "error", "cancelled");
  el.classList.add(state);

  const detailEl = document.getElementById(`stepDetail-${stepKey}`);
  if (detailEl && detailText) {
    detailEl.textContent = detailText;
  }

  const bullet = el.querySelector(".step-bullet");
  if (bullet) {
    if (state === "done") {
      bullet.innerHTML = '<i class="fa-solid fa-check"></i>';
    } else if (state === "active") {
      bullet.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
    } else if (state === "skipped") {
      bullet.innerHTML = '<i class="fa-solid fa-forward"></i>';
    } else if (state === "error") {
      bullet.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    } else if (state === "cancelled") {
      bullet.innerHTML = '<i class="fa-solid fa-ban"></i>';
    }
  }
}

function resetStepper() {
  const steps = ["extract", "denoise", "asr", "diarization", "verification", "punctuation"];
  steps.forEach(k => {
    updateStepperStep(k, "pending", "Chờ xử lý...");
  });
  const bullets = {
    "extract": '<i class="fa-solid fa-film"></i>',
    "denoise": '<i class="fa-solid fa-wand-magic-sparkles"></i>',
    "asr": '<i class="fa-solid fa-microphone-lines"></i>',
    "diarization": '<i class="fa-solid fa-users"></i>',
    "verification": '<i class="fa-solid fa-id-card"></i>',
    "punctuation": '<i class="fa-solid fa-font"></i>'
  };
  steps.forEach(k => {
    const el = document.getElementById(`step-${k}`);
    if (el) {
      const b = el.querySelector(".step-bullet");
      if (b) b.innerHTML = bullets[k];
    }
  });
}

function formatSec(seconds) {
  if (!seconds || isNaN(seconds)) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function handleOfflineCancelled(msg = "Quá trình xử lý đã được hủy thành công.") {
  clearInterval(offlineProgressTimer);
  offlineProgressBar.style.background = "var(--warning)";
  progressSubStatus.innerHTML = `<span style="color: var(--warning);"><i class="fa-solid fa-ban"></i> ${msg}</span>`;
  progressSpinner.className = "spinner-pulse";
  progressSpinner.style.borderColor = "var(--warning)";
  progressSpinner.style.color = "var(--warning)";
  progressSpinner.innerHTML = '<i class="fa-solid fa-ban"></i>';

  const stepOrder = ["extract", "denoise", "asr", "diarization", "verification", "punctuation"];
  stepOrder.forEach(k => {
    const el = document.getElementById(`step-${k}`);
    if (el && el.classList.contains("active")) {
      updateStepperStep(k, "cancelled", "Đã hủy");
    }
  });

  showInlineAlert("offlineAlertBanner", `Đã hủy xử lý: Các tiến trình AI nền đã dừng và file tạm đã được dọn dẹp an toàn.`, "warning");
  showNotification("Đã hủy xử lý file!", "warning", 4000);
  interimTextBox.innerHTML = `<em>(Đã hủy xử lý video)</em>`;
}

async function cancelOfflineProcessing() {
  if (!currentOfflineTaskId && !currentOfflineAbortController) return;

  const taskIdToCancel = currentOfflineTaskId;
  isOfflineCancelled = true;

  // 1. Gửi tín hiệu hủy tức thì tới backend
  if (taskIdToCancel) {
    try {
      fetch(`/api/cancel-process?task_id=${encodeURIComponent(taskIdToCancel)}`, {
        method: "POST"
      }).catch(e => console.warn("[Cancel] Backend cancel request failed:", e));
    } catch (e) {}
  }

  // 2. Ngắt kết nối SSE stream ngay lập tức
  if (currentOfflineAbortController) {
    try {
      currentOfflineAbortController.abort();
    } catch (e) {}
  }

  // 3. Cập nhật UI sang trạng thái Đã hủy
  handleOfflineCancelled("Đã hủy quá trình xử lý theo yêu cầu của người dùng.");

  // 4. Ẩn nút hủy và kích hoạt lại nút Bắt đầu
  if (btnCancelProcess) btnCancelProcess.style.display = "none";
  if (btnCancelProcessHeader) btnCancelProcessHeader.style.display = "none";
  btnProcessFile.disabled = false;
  const isVideo = selectedAudioFile && (selectedAudioFile.type.startsWith("video/") || /\.(mp4|mkv|avi|mov|webm)$/i.test(selectedAudioFile.name));
  btnProcessFile.innerHTML = `<i class="fa-solid fa-play"></i> Bắt Đầu Xử Lý ${isVideo ? "Video" : "Âm Thanh"}`;
}

btnProcessFile.addEventListener("click", async () => {
  if (!selectedAudioFile) return;

  isOfflineCancelled = false;
  currentOfflineTaskId = "task_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);
  currentOfflineAbortController = new AbortController();

  // Xóa màn hình và màu sắc cũ
  transcriptFeed.innerHTML = "";
  speakerColorMap.clear();
  interimTextBox.innerHTML = `<em>(Đang phân tích video offline: <strong>${selectedAudioFile.name}</strong>...)</em>`;

  // 1. Hiển thị UI Tiến trình Offline
  offlineProgressCard.style.display = "block";
  hideInlineAlert("offlineAlertBanner");
  resetStepper();
  offlineProgressBar.style.width = "5%";
  offlineProgressBar.style.background = "";
  progressPercentBadge.textContent = "5%";
  progressMainTitle.textContent = `Đang xử lý: ${selectedAudioFile.name}`;
  progressSubStatus.textContent = "Đang tải file lên máy chủ và khởi tạo pipeline...";
  progressSpinner.className = "spinner-pulse";
  progressSpinner.style.borderColor = "";
  progressSpinner.style.color = "";
  progressSpinner.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';

  statDuration.textContent = "--:--";
  statSpeed.textContent = "--";

  offlineStartTime = Date.now();
  if (offlineProgressTimer) clearInterval(offlineProgressTimer);
  offlineProgressTimer = setInterval(() => {
    const elapsed = (Date.now() - offlineStartTime) / 1000;
    statElapsed.textContent = formatSec(elapsed);
  }, 1000);

  btnProcessFile.disabled = true;
  btnProcessFile.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích...`;
  if (btnCancelProcess) btnCancelProcess.style.display = "inline-flex";
  if (btnCancelProcessHeader) btnCancelProcessHeader.style.display = "inline-flex";
  showNotification(`Bắt đầu xử lý file "${selectedAudioFile.name}"...`, "info", 3000);

  const formData = new FormData();
  formData.append("file", selectedAudioFile);
  formData.append("task_id", currentOfflineTaskId);
  formData.append("denoise", toggleDenoise.checked);
  formData.append("asr", toggleAsr.checked);
  formData.append("diarization", toggleDiarization.checked);
  formData.append("verification", toggleVerification.checked);
  formData.append("hotwords", toggleHotwords.checked);
  formData.append("punctuation", togglePunctuation.checked);

  // Truyền toàn bộ các thông số model đang được tinh chỉnh trên giao diện
  const activeParams = typeof getInlineModelParameters === "function" ? getInlineModelParameters() : currentModelConfig;
  for (const [k, v] of Object.entries(activeParams)) {
    if (v !== undefined && v !== null) {
      formData.append(k, v);
    }
  }

  const stepOrder = ["extract", "denoise", "asr", "diarization", "verification", "punctuation"];

  try {
    const response = await fetch("/api/process-file-stream", {
      method: "POST",
      body: formData,
      signal: currentOfflineAbortController.signal,
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // Giữ lại phần chưa hoàn chỉnh

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const data = JSON.parse(jsonStr);

          // Nhận task_id chính thức từ server nếu cần
          if (data.type === "init") {
            if (data.task_id) currentOfflineTaskId = data.task_id;
          }

          // Sự kiện hủy xử lý từ phía server
          else if (data.type === "cancelled") {
            isOfflineCancelled = true;
            handleOfflineCancelled(data.message || "Quá trình xử lý đã được hủy thành công.");
            return;
          }

          // Sự kiện báo tiến trình từng bước
          else if (data.type === "progress") {
            const p = Math.max(5, Math.min(100, data.percent || 0));
            offlineProgressBar.style.width = `${p}%`;
            progressPercentBadge.textContent = `${p}%`;
            if (data.message) {
              progressSubStatus.textContent = data.message;
            }

            if (data.duration_s) {
              statDuration.textContent = formatSec(data.duration_s);
            }

            const currentKey = data.step_name;
            const currentIndex = stepOrder.indexOf(currentKey);

            if (currentIndex !== -1) {
              for (let i = 0; i < currentIndex; i++) {
                updateStepperStep(stepOrder[i], "done", "Hoàn tất");
              }

              if (data.skipped) {
                updateStepperStep(currentKey, "skipped", data.message || "Bỏ qua");
              } else if (p >= 98 || (currentKey === "punctuation" && p >= 95)) {
                updateStepperStep(currentKey, "done", data.message || "Hoàn tất");
              } else {
                updateStepperStep(currentKey, "active", data.message || "Đang xử lý...");
              }
            }
          }

          // Sự kiện hoàn tất toàn bộ pipeline
          else if (data.type === "complete") {
            clearInterval(offlineProgressTimer);
            offlineProgressBar.style.width = "100%";
            progressPercentBadge.textContent = "100%";
            progressSubStatus.textContent = "✅ Đã hoàn tất xử lý toàn bộ media!";
            progressSpinner.className = "spinner-pulse done";
            progressSpinner.innerHTML = '<i class="fa-solid fa-check"></i>';

            stepOrder.forEach(k => {
              const el = document.getElementById(`step-${k}`);
              if (el && !el.classList.contains("skipped")) {
                updateStepperStep(k, "done", "Hoàn tất");
              }
            });

            const elapsed = (Date.now() - offlineStartTime) / 1000;
            statElapsed.textContent = formatSec(elapsed);
            const totalDur = (data.result && data.result.total_duration_s) || 0;
            let rtfStr = "--";
            if (totalDur > 0 && elapsed > 0) {
              statDuration.textContent = formatSec(totalDur);
              const rtf = (totalDur / elapsed).toFixed(1);
              rtfStr = `${rtf}x Realtime`;
              statSpeed.textContent = rtfStr;
            }

            interimTextBox.innerHTML = `<em>(Đã hoàn thành phân tích: <strong>${selectedAudioFile.name}</strong> trong ${elapsed.toFixed(1)}s!)</em>`;

            // Thông báo thành công
            showInlineAlert("offlineAlertBanner", `Xử lý hoàn tất thành công trong ${elapsed.toFixed(1)}s (Tốc độ: ${rtfStr})! Biên bản hội thoại đã sẵn sàng.`, "success");
            showNotification(`Xử lý file "${selectedAudioFile.name}" thành công!`, "success", 5000);

            // Render kết quả hoàn chỉnh
            if (data.result) data.result.source = "offline";
            renderFinalResult(data.result);
          }

          // Sự kiện báo lỗi từ SSE stream
          else if (data.type === "error") {
            throw new Error(data.message || "Lỗi không xác định từ máy chủ");
          }

        } catch (jsonErr) {
          console.warn("[SSE Stream] Parse error:", jsonErr, jsonStr);
        }
      }
    }

  } catch (err) {
    if (err.name === "AbortError" || isOfflineCancelled) {
      handleOfflineCancelled("Đã hủy quá trình xử lý theo yêu cầu của bạn.");
    } else {
      clearInterval(offlineProgressTimer);
      offlineProgressBar.style.background = "var(--danger)";
      progressSubStatus.innerHTML = `<span style="color: var(--danger);">Lỗi xử lý: ${err.message}</span>`;
      progressSpinner.className = "spinner-pulse";
      progressSpinner.style.borderColor = "var(--danger)";
      progressSpinner.style.color = "var(--danger)";
      progressSpinner.innerHTML = '<i class="fa-solid fa-xmark"></i>';

      // Đánh dấu step đang chạy là error
      stepOrder.forEach(k => {
        const el = document.getElementById(`step-${k}`);
        if (el && el.classList.contains("active")) {
          updateStepperStep(k, "error", "Gặp lỗi");
        }
      });

      showInlineAlert("offlineAlertBanner", `Lỗi trong quá trình xử lý: ${err.message}`, "error");
      showNotification(`Lỗi xử lý video: ${err.message}`, "error", 7000);
      interimTextBox.innerHTML = `<em>(Gặp lỗi khi xử lý video: ${err.message})</em>`;
    }

  } finally {
    if (btnCancelProcess) btnCancelProcess.style.display = "none";
    if (btnCancelProcessHeader) btnCancelProcessHeader.style.display = "none";
    currentOfflineAbortController = null;
    btnProcessFile.disabled = false;
    const isVideo = selectedAudioFile && (selectedAudioFile.type.startsWith("video/") || /\.(mp4|mkv|avi|mov|webm)$/i.test(selectedAudioFile.name));
    btnProcessFile.innerHTML = `<i class="fa-solid fa-play"></i> Bắt Đầu Xử Lý ${isVideo ? "Video" : "Âm Thanh"}`;
  }
});

// ==============================================================================
// 6. XUẤT PHỤ ĐỀ (SRT / VTT / TXT)
// ==============================================================================
async function exportSubtitles(format) {
  if (!lastProcessedResult) {
    showNotification("Chưa có kết quả để xuất phụ đề. Hãy thực hiện nhận diện trước!", "warning", 4000);
    return;
  }

  const filename = (selectedAudioFile ? selectedAudioFile.name : "transcript");
  const btn = document.getElementById(`btnExport${format.charAt(0).toUpperCase() + format.slice(1)}`);
  const originalHtml = btn ? btn.innerHTML : "";
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Xuất...`;
  }

  try {
    const res = await fetch("/api/export-subtitles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        format: format,
        segments: lastProcessedResult.segments || [],
        final_text: lastProcessedResult.final_text || "",
        filename: filename
      })
    });

    if (!res.ok) throw new Error(await res.text());

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const baseName = filename.substring(0, filename.lastIndexOf('.')) || filename;
    a.download = `${baseName}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification(`Đã xuất phụ đề định dạng .${format.toUpperCase()} thành công!`, "success", 3500);

  } catch (e) {
    showNotification(`Lỗi khi xuất phụ đề .${format.toUpperCase()}: ` + e.message, "error", 6000);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = originalHtml;
    }
  }
}

document.getElementById("btnExportSrt")?.addEventListener("click", () => exportSubtitles("srt"));
document.getElementById("btnExportVtt")?.addEventListener("click", () => exportSubtitles("vtt"));
document.getElementById("btnExportTxt")?.addEventListener("click", () => exportSubtitles("txt"));

// ==============================================================================
// 7. TABS, SIDEBAR TOGGLES & BASIC ACTIONS
// ==============================================================================
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");

    // Dừng âm thanh của tab trước đó để tránh phát lẫn
    if (btn.dataset.tab === "tab-mic") {
      if (audioPreview) audioPreview.pause();
      if (videoPreview) videoPreview.pause();
    } else if (btn.dataset.tab === "tab-file") {
      if (micAudioPreview) micAudioPreview.pause();
    }
  });
});

btnStartRecord.addEventListener("click", startRecording);
btnStopRecord.addEventListener("click", stopRecording);
if (btnCancelRecord) btnCancelRecord.addEventListener("click", cancelRecording);
if (btnCancelProcess) btnCancelProcess.addEventListener("click", cancelOfflineProcessing);
if (btnCancelProcessHeader) btnCancelProcessHeader.addEventListener("click", cancelOfflineProcessing);

btnClearTranscript.addEventListener("click", () => {
  transcriptFeed.innerHTML = `
    <div class="empty-state">
      <i class="fa-solid fa-comment-dots"></i>
      <p>Màn hình đã được xóa. Hãy bắt đầu lượt mới!</p>
    </div>
  `;
  speakerColorMap.clear();
  showNotification("Đã xóa sạch màn hình hội thoại.", "info", 2000);
});

btnApplyToggles.addEventListener("click", async () => {
  btnApplyToggles.disabled = true;
  btnApplyToggles.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang áp dụng...`;
  if (toggleProgressWrapper) toggleProgressWrapper.style.display = "block";
  hideInlineAlert("toggleAlert");

  const inlineParams = getInlineModelParameters();

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inlineParams)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không thể cập nhật cấu hình.");

    currentModelConfig = data.config || inlineParams;
    updateInlineModelParametersUI(currentModelConfig);
    if (modelSettingsMeta) updateModelSettingsValues(currentModelConfig);

    sendToggleConfig();

    showInlineAlert("toggleAlert", "Đã cập nhật các tùy chọn & thông số mô hình AI thành công!", "success");
    showNotification("Đã áp dụng thông số mô hình và tùy chọn Bật/Tắt thành công!", "success", 3000);
  } catch (err) {
    showInlineAlert("toggleAlert", "Lỗi cập nhật cấu hình: " + err.message, "error");
    showNotification("Lỗi cập nhật cấu hình: " + err.message, "error", 4000);
  } finally {
    btnApplyToggles.disabled = false;
    btnApplyToggles.innerHTML = `<i class="fa-solid fa-sync"></i> Cập Nhật Cấu Hình Tức Thì`;
    if (toggleProgressWrapper) toggleProgressWrapper.style.display = "none";
    setTimeout(() => hideInlineAlert("toggleAlert"), 4500);
  }
});

const btnResetInlineParams = document.getElementById("btnResetInlineParams");
if (btnResetInlineParams) {
  btnResetInlineParams.addEventListener("click", async () => {
    if (!confirm("Bạn có chắc chắn muốn khôi phục toàn bộ thông số mô hình về giá trị mặc định?")) return;

    btnResetInlineParams.disabled = true;
    hideInlineAlert("toggleAlert");
    if (toggleProgressWrapper) toggleProgressWrapper.style.display = "block";

    try {
      const res = await fetch("/api/config/reset", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Không thể khôi phục cấu hình.");

      currentModelConfig = data.config || {};
      updateInlineModelParametersUI(currentModelConfig);
      if (modelSettingsMeta) updateModelSettingsValues(currentModelConfig);
      sendToggleConfig();

      showInlineAlert("toggleAlert", "Đã khôi phục toàn bộ thông số mô hình về mặc định!", "success");
      showNotification("Đã khôi phục toàn bộ thông số mô hình về mặc định!", "info", 3000);
    } catch (err) {
      showInlineAlert("toggleAlert", "Lỗi khôi phục: " + err.message, "error");
      showNotification("Lỗi khôi phục thông số: " + err.message, "error", 4000);
    } finally {
      btnResetInlineParams.disabled = false;
      if (toggleProgressWrapper) toggleProgressWrapper.style.display = "none";
      setTimeout(() => hideInlineAlert("toggleAlert"), 4500);
    }
  });
}

// ==============================================================================
// 8. MODALS: SPEAKERS & HOTWORDS
// ==============================================================================
document.getElementById("btnOpenSpeakers").addEventListener("click", () => {
  modalSpeakers.classList.add("active");
  hideInlineAlert("speakerModalAlert");
  loadSpeakersList();
});

document.getElementById("btnOpenHotwords").addEventListener("click", () => {
  modalHotwords.classList.add("active");
  hideInlineAlert("hotwordModalAlert");
  loadHotwordsList();
});

document.querySelectorAll(".close-modal, .close-modal-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    modalSpeakers.classList.remove("active");
    modalHotwords.classList.remove("active");
    modalModelSettings.classList.remove("active");
  });
});

window.addEventListener("click", (e) => {
  if (e.target === modalSpeakers) modalSpeakers.classList.remove("active");
  if (e.target === modalHotwords) modalHotwords.classList.remove("active");
  if (e.target === modalModelSettings) modalModelSettings.classList.remove("active");
});

async function loadSpeakersList() {
  const container = document.getElementById("speakersListTable");
  try {
    const res = await fetch("/api/speakers");
    const list = await res.json();
    if (!list || list.length === 0) {
      container.innerHTML = "<em>Chưa có hồ sơ giọng nói nào được lưu trong kho.</em>";
      return;
    }
    container.innerHTML = list.map(item => `
      <div class="list-item">
        <div><strong>${item.name}</strong> <span style="color:var(--text-muted); font-size:12px;">(${item.sample_count} mẫu) - ${item.description || "Không có mô tả"}</span></div>
        <button class="btn btn-danger btn-sm" onclick="deleteSpeaker('${item.name}')"><i class="fa-solid fa-trash"></i></button>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<span style="color:red">Lỗi tải: ${err.message}</span>`;
  }
}

window.deleteSpeaker = async function(name) {
  if (!confirm(`Bạn có chắc muốn xóa hồ sơ của '${name}'?`)) return;
  try {
    const res = await fetch(`/api/speakers/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    showNotification(`Đã xóa hồ sơ người nói '${name}' thành công!`, "success", 3000);
    loadSpeakersList();
  } catch (err) {
    showNotification(`Lỗi xóa hồ sơ: ${err.message}`, "error", 5000);
  }
};

document.getElementById("btnEnrollSpeaker").addEventListener("click", async () => {
  const btn = document.getElementById("btnEnrollSpeaker");
  const name = document.getElementById("speakerNameInput").value.trim();
  const desc = document.getElementById("speakerDescInput").value.trim();
  const fileInput = document.getElementById("speakerVoiceFileInput");

  if (!name) {
    showNotification("Vui lòng nhập họ tên người nói!", "warning", 3000);
    return;
  }
  if (!fileInput.files || fileInput.files.length === 0) {
    showNotification("Vui lòng chọn file mẫu Video hoặc Âm thanh giọng nói!", "warning", 3000);
    return;
  }

  const fd = new FormData();
  fd.append("name", name);
  fd.append("description", desc);
  fd.append("file", fileInput.files[0]);

  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang đăng ký...`;
  if (enrollProgressWrapper) enrollProgressWrapper.style.display = "block";
  hideInlineAlert("speakerModalAlert");

  try {
    const res = await fetch("/api/speakers/enroll", { method: "POST", body: fd });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || "Lỗi không xác định");

    showInlineAlert("speakerModalAlert", json.message || "Đăng ký thành công!", "success");
    showNotification(json.message || `Đã đăng ký thành công hồ sơ của "${name}"!`, "success", 4000);

    document.getElementById("speakerNameInput").value = "";
    document.getElementById("speakerDescInput").value = "";
    fileInput.value = "";
    loadSpeakersList();
  } catch (err) {
    showInlineAlert("speakerModalAlert", "Lỗi đăng ký: " + err.message, "error");
    showNotification("Lỗi đăng ký giọng nói: " + err.message, "error", 6000);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-user-plus"></i> Đăng Ký Voiceprint`;
    if (enrollProgressWrapper) enrollProgressWrapper.style.display = "none";
  }
});

async function loadHotwordsList() {
  const container = document.getElementById("hotwordsListTable");
  try {
    const res = await fetch("/api/hotwords");
    const data = await res.json();
    const items = data.hotwords || [];
    if (items.length === 0) {
      container.innerHTML = "<em>Chưa có từ khóa nào trong danh sách ZipFormer.</em>";
      return;
    }
    container.innerHTML = items.map(item => `
      <div class="list-item">
        <span>
          <strong style="color:var(--primary); font-size:14px;">${item.phrase}</strong>
          <span style="display:inline-block; background:#e0e7ff; color:#3730a3; font-size:11px; font-weight:600; padding:2px 8px; border-radius:12px; margin-left:8px;">
            Bonus: +${item.score}
          </span>
        </span>
        <button class="btn btn-danger btn-sm" onclick="deleteHotword('${item.phrase}')" title="Xóa từ khóa">
          <i class="fa-solid fa-trash"></i>
        </button>
      </div>
    `).join("");
  } catch (err) {
    container.innerHTML = `<span style="color:red">Lỗi tải: ${err.message}</span>`;
  }
}

window.deleteHotword = async function(phrase) {
  try {
    const res = await fetch(`/api/hotwords/${encodeURIComponent(phrase)}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    showNotification(`Đã xóa từ khóa '${phrase}'!`, "success", 3000);
    loadHotwordsList();
  } catch (err) {
    showNotification(`Lỗi xóa từ khóa: ${err.message}`, "error", 5000);
  }
};

document.getElementById("btnAddHotword").addEventListener("click", async () => {
  const btn = document.getElementById("btnAddHotword");
  const phrase = document.getElementById("hwPhraseInput").value.trim();
  const score = parseFloat(document.getElementById("hwScoreInput").value) || 2.0;

  if (!phrase) {
    showNotification("Vui lòng nhập từ hoặc cụm từ khóa chuyên ngành!", "warning", 3000);
    return;
  }

  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang thêm...`;
  if (hotwordProgressWrapper) hotwordProgressWrapper.style.display = "block";
  hideInlineAlert("hotwordModalAlert");

  try {
    const res = await fetch(`/api/hotwords?phrase=${encodeURIComponent(phrase)}&score=${score}`, {
      method: "POST"
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || "Lỗi không xác định");

    document.getElementById("hwPhraseInput").value = "";
    document.getElementById("hwScoreInput").value = "2.0";
    showInlineAlert("hotwordModalAlert", `Đã thêm từ khóa "${phrase}" (+${score}) thành công!`, "success");
    showNotification(`Đã thêm từ khóa "${phrase}" (+${score}) vào ZipFormer!`, "success", 3500);
    loadHotwordsList();
  } catch (err) {
    showInlineAlert("hotwordModalAlert", "Lỗi thêm từ khóa: " + err.message, "error");
    showNotification("Lỗi thêm từ khóa: " + err.message, "error", 5000);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-plus"></i> Thêm Từ Khóa`;
    if (hotwordProgressWrapper) hotwordProgressWrapper.style.display = "none";
  }
});

// ==============================================================================
// 9. INLINE MODEL PARAMETERS (TINH CHỈNH NGAY TẠI TỪNG CHỨC NĂNG MODEL)
// ==============================================================================
function initInlineModelParameters() {
  // 1. Nút bấm ẩn/hiện bảng thông số cho từng model
  document.querySelectorAll(".btn-toggle-config").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const targetId = btn.dataset.target;
      const targetPanel = document.getElementById(targetId);
      if (!targetPanel) return;
      const isCollapsed = targetPanel.classList.toggle("collapsed");
      btn.classList.toggle("active", !isCollapsed);
    });
  });

  // 2. Gắn sự kiện cập nhật Live Badge tức thì khi kéo Slider hoặc nhập số
  const inlineBindings = [
    { id: "param_denoise_atten_lim_db", badgeId: "badge_denoise_atten_lim_db", unit: "dB", isFloat: false },
    { id: "param_asr_max_active_paths", badgeId: "badge_asr_max_active_paths", unit: "paths", isFloat: false },
    { id: "param_diarization_collar", badgeId: "badge_diarization_collar", unit: "s", isFloat: true, decimals: 2 },
    { id: "param_diarization_min_duration", badgeId: "badge_diarization_min_duration", unit: "s", isFloat: true, decimals: 2 },
    { id: "param_diarization_chunk_len", badgeId: "badge_diarization_chunk_len", unit: "frames", isFloat: false },
    { id: "param_verification_threshold", badgeId: "badge_verification_threshold", unit: "", isFloat: true, decimals: 2 },
    { id: "param_min_speech_duration_for_id", badgeId: "badge_min_speech_duration_for_id", unit: "s", isFloat: true, decimals: 2 },
    { id: "param_asr_hotwords_score", badgeId: "badge_asr_hotwords_score", unit: "pts", prefix: "+", isFloat: true, decimals: 1 },
    { id: "param_max_recent_words_per_speaker", badgeId: "badge_max_recent_words_per_speaker", unit: "từ", isFloat: false },
    { id: "param_speaker_inactivity_timeout_s", badgeId: "badge_speaker_inactivity_timeout_s", unit: "s", isFloat: false },
    { id: "param_interim_max_words", badgeId: "badge_interim_max_words", unit: "từ", isFloat: false },
    { id: "param_interim_idle_timeout_s", badgeId: "badge_interim_idle_timeout_s", unit: "s", isFloat: true, decimals: 1 },
    { id: "param_interim_timeout_s", badgeId: "badge_interim_timeout_s", unit: "s", isFloat: true, decimals: 1 },
  ];

  inlineBindings.forEach(b => {
    const el = document.getElementById(b.id);
    const badge = document.getElementById(b.badgeId);
    if (!el || !badge) return;

    const update = () => {
      let val = parseFloat(el.value);
      if (isNaN(val)) val = 0;
      let displayVal = b.isFloat ? val.toFixed(b.decimals) : val;
      const prefix = b.prefix || "";
      const unit = b.unit ? ` ${b.unit}` : "";

      if (b.id === "param_denoise_atten_lim_db") {
        let label = "Không khử";
        if (val >= 50) label = "Khử tối đa";
        else if (val >= 35) label = "Khử mạnh";
        else if (val >= 20) label = "Khử vừa";
        else if (val > 0) label = "Khử nhẹ";
        badge.textContent = `${val} dB (${label})`;
      } else {
        badge.textContent = `${prefix}${displayVal}${unit}`;
      }

      // Cập nhật giá trị vào biến state
      const key = b.id.replace("param_", "");
      currentModelConfig[key] = val;

      if (b.id.includes("interim")) {
        if (b.id.includes("idle") || b.id.includes("timeout")) {
          if (interimIdleTimer && !isInterimIdleHidden && lastWordAddedTimestamp > 0) {
            const idleTimeoutMs = getInterimIdleTimeoutMs();
            const elapsed = Date.now() - lastWordAddedTimestamp;
            const remaining = Math.max(50, idleTimeoutMs - elapsed);
            clearTimeout(interimIdleTimer);
            interimIdleTimer = setTimeout(onInterimIdleTimeout, remaining);
          }
        }
        renderInterimText();
      }
    };

    el.addEventListener("input", update);
    el.addEventListener("change", update);
  });

  const decodingEl = document.getElementById("param_asr_decoding_method");
  if (decodingEl) {
    decodingEl.addEventListener("change", () => {
      currentModelConfig["asr_decoding_method"] = decodingEl.value;
    });
  }

  const diarModelEl = document.getElementById("param_diarization_model");
  if (diarModelEl) {
    diarModelEl.addEventListener("change", () => {
      currentModelConfig["diarization_model"] = diarModelEl.value;
    });
  }

  const toggleRtDiarEl = document.getElementById("toggleRealtimeDiarization");
  if (toggleRtDiarEl) {
    toggleRtDiarEl.addEventListener("change", () => {
      const livePanel = document.getElementById("liveMultiSpeakerPanel");
      if (livePanel && !toggleRtDiarEl.checked) {
        livePanel.style.display = "none";
      }
      sendToggleConfig();
    });
  }
}

function getInlineModelParameters() {
  return {
    verification_threshold: parseFloat(document.getElementById("param_verification_threshold")?.value) || 0.55,
    min_speech_duration_for_id: parseFloat(document.getElementById("param_min_speech_duration_for_id")?.value) || 0.80,
    diarization_model: document.getElementById("param_diarization_model")?.value || "nvidia/Nemotron-3-Diarization",
    diarization_collar: parseFloat(document.getElementById("param_diarization_collar")?.value) || 0.25,
    diarization_min_duration: parseFloat(document.getElementById("param_diarization_min_duration")?.value) || 0.20,
    diarization_chunk_len: parseInt(document.getElementById("param_diarization_chunk_len")?.value, 10) || 340,
    asr_decoding_method: document.getElementById("param_asr_decoding_method")?.value || "modified_beam_search",
    asr_max_active_paths: parseInt(document.getElementById("param_asr_max_active_paths")?.value, 10) || 4,
    asr_hotwords_score: parseFloat(document.getElementById("param_asr_hotwords_score")?.value) || 2.0,
    denoise_atten_lim_db: parseFloat(document.getElementById("param_denoise_atten_lim_db")?.value) ?? 40.0,
    enable_realtime_diarization: document.getElementById("toggleRealtimeDiarization") ? document.getElementById("toggleRealtimeDiarization").checked : true,
    max_recent_words_per_speaker: parseInt(document.getElementById("param_max_recent_words_per_speaker")?.value, 10) || 25,
    speaker_inactivity_timeout_s: parseFloat(document.getElementById("param_speaker_inactivity_timeout_s")?.value) || 7.0,
    interim_max_words: parseInt(document.getElementById("param_interim_max_words")?.value, 10) || 20,
    interim_idle_timeout_s: parseFloat(document.getElementById("param_interim_idle_timeout_s")?.value) || 3.0,
    enable_2pass_streaming: document.getElementById("toggle2PassStreaming") ? document.getElementById("toggle2PassStreaming").checked : true,
    turn_silence_timeout_s: parseFloat(document.getElementById("param_turn_silence_timeout_s")?.value) || 0.7,
  };
}

function updateInlineModelParametersUI(config) {
  if (!config) return;
  const setVal = (id, badgeId, val, unit = "", prefix = "", isFloat = false, decimals = 2) => {
    const el = document.getElementById(id);
    const badge = document.getElementById(badgeId);
    if (el && val !== undefined) {
      el.value = val;
    }
    if (badge && val !== undefined) {
      if (id === "param_denoise_atten_lim_db") {
        let label = "Không khử";
        if (val >= 50) label = "Khử tối đa";
        else if (val >= 35) label = "Khử mạnh";
        else if (val >= 20) label = "Khử vừa";
        else if (val > 0) label = "Khử nhẹ";
        badge.textContent = `${val} dB (${label})`;
      } else {
        let d = val;
        if (isFloat && typeof val === "number") d = val.toFixed(decimals);
        badge.textContent = `${prefix}${d}${unit ? ' ' + unit : ''}`;
      }
    }
  };

  if (config.enable_denoise !== undefined && toggleDenoise) {
    toggleDenoise.checked = config.enable_denoise;
  }

  setVal("param_verification_threshold", "badge_verification_threshold", config.verification_threshold, "", "", true, 2);
  setVal("param_min_speech_duration_for_id", "badge_min_speech_duration_for_id", config.min_speech_duration_for_id, "s", "", true, 2);
  setVal("param_diarization_model", null, config.diarization_model);
  setVal("param_diarization_collar", "badge_diarization_collar", config.diarization_collar, "s", "", true, 2);
  setVal("param_diarization_min_duration", "badge_diarization_min_duration", config.diarization_min_duration, "s", "", true, 2);
  setVal("param_diarization_chunk_len", "badge_diarization_chunk_len", config.diarization_chunk_len, "frames", "", false);
  setVal("param_asr_decoding_method", null, config.asr_decoding_method);
  setVal("param_asr_max_active_paths", "badge_asr_max_active_paths", config.asr_max_active_paths, "paths", "", false);
  setVal("param_asr_hotwords_score", "badge_asr_hotwords_score", config.asr_hotwords_score, "pts", "+", true, 1);
  setVal("param_denoise_atten_lim_db", "badge_denoise_atten_lim_db", config.denoise_atten_lim_db, "dB", "", false);

  if (config.enable_realtime_diarization !== undefined) {
    const tg = document.getElementById("toggleRealtimeDiarization");
    if (tg) tg.checked = config.enable_realtime_diarization;
  }
  if (config.enable_2pass_streaming !== undefined) {
    const tg2 = document.getElementById("toggle2PassStreaming");
    if (tg2) tg2.checked = config.enable_2pass_streaming;
  }
  setVal("param_turn_silence_timeout_s", "badge_turn_silence_timeout_s", config.turn_silence_timeout_s, "s", "", true, 1);
  setVal("param_max_recent_words_per_speaker", "badge_max_recent_words_per_speaker", config.max_recent_words_per_speaker, "từ", "", false);
  setVal("param_speaker_inactivity_timeout_s", "badge_speaker_inactivity_timeout_s", config.speaker_inactivity_timeout_s, "s", "", false);
  setVal("param_interim_max_words", "badge_interim_max_words", config.interim_max_words, "từ", "", false);
  setVal("param_interim_idle_timeout_s", "badge_interim_idle_timeout_s", config.interim_idle_timeout_s, "s", "", true, 1);
}

// ==============================================================================
// 10. MODAL: MODEL HYPERPARAMETERS (TÙY CHỌN MỞ RỘNG)
// ==============================================================================
document.getElementById("btnOpenModelSettings").addEventListener("click", () => {
  modalModelSettings.classList.add("active");
  hideInlineAlert("modelSettingsAlert");
  loadModelSettings(true);
});

async function loadModelSettings(forceRender = false) {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    modelSettingsMeta = data;
    currentModelConfig = data.current_config || data.current || {};

    updateInlineModelParametersUI(currentModelConfig);

    if (forceRender || !modelSettingsGroupsContainer.querySelector(".setting-group-card")) {
      renderModelSettingsUI(data);
    } else {
      updateModelSettingsValues(currentModelConfig);
    }
  } catch (err) {
    console.error("Lỗi tải thông số mô hình:", err);
    if (forceRender) {
      modelSettingsGroupsContainer.innerHTML = `
        <div class="settings-loading-placeholder" style="color:var(--danger)">
          <i class="fa-solid fa-triangle-exclamation fa-2x"></i>
          <p style="margin-top:8px;">Lỗi tải thông số từ hệ thống: ${err.message}</p>
        </div>
      `;
    }
  }
}

function renderModelSettingsUI(meta) {
  if (!meta || !meta.groups) return;
  const current = meta.current_config || meta.current || {};
  const groupsHtml = meta.groups.map(group => {
    const paramsHtml = (group.params || []).map(p => {
      const val = current[p.key] !== undefined ? current[p.key] : p.default;
      let controlHtml = "";

      if (p.type === "slider") {
        controlHtml = `
          <div class="param-slider-wrapper">
            <span class="param-min-max">${p.min}${p.unit ? ' ' + p.unit : ''}</span>
            <input type="range" 
                   id="setting_${p.key}" 
                   name="${p.key}" 
                   min="${p.min}" 
                   max="${p.max}" 
                   step="${p.step}" 
                   value="${val}" 
                   data-unit="${p.unit || ''}"
                   data-type="float"
                   class="param-range-input">
            <span class="param-min-max text-right">${p.max}${p.unit ? ' ' + p.unit : ''}</span>
          </div>
        `;
      } else if (p.type === "number") {
        controlHtml = `
          <input type="number" 
                 id="setting_${p.key}" 
                 name="${p.key}" 
                 min="${p.min !== undefined ? p.min : 0}" 
                 max="${p.max !== undefined ? p.max : 1000}" 
                 step="${p.step || 1}" 
                 value="${val}" 
                 data-unit="${p.unit || ''}"
                 data-type="int"
                 class="param-number-input">
        `;
      } else if (p.type === "select") {
        const optionsHtml = (p.options || []).map(opt => `
          <option value="${opt.value}" ${opt.value === val ? 'selected' : ''}>${opt.label}</option>
        `).join("");
        controlHtml = `
          <select id="setting_${p.key}" name="${p.key}" class="param-select">
            ${optionsHtml}
          </select>
        `;
      } else if (p.type === "toggle") {
        const isChecked = (val === true || val === "true" || val === 1) ? 'checked' : '';
        controlHtml = `
          <div style="margin-top: 6px;">
            <label class="switch">
              <input type="checkbox" id="setting_${p.key}" name="${p.key}" ${isChecked}>
              <span class="slider"></span>
            </label>
          </div>
        `;
      }

      const unitText = p.unit ? ` ${p.unit}` : '';
      return `
        <div class="setting-param-item">
          <div class="param-label-row">
            <label class="param-label" for="setting_${p.key}">${p.label}</label>
            <span id="badge_${p.key}" class="param-badge">${val}${unitText}</span>
          </div>
          ${controlHtml}
          <div class="param-desc">${p.description || ''}</div>
        </div>
      `;
    }).join("");

    return `
      <div class="setting-group-card">
        <div class="setting-group-header">
          <i class="fa-solid ${group.icon || 'fa-sliders'}"></i>
          <span>${group.title}</span>
        </div>
        <div class="setting-params-list">
          ${paramsHtml}
        </div>
      </div>
    `;
  }).join("");

  modelSettingsGroupsContainer.innerHTML = groupsHtml;

  // Realtime badge sync listeners
  meta.groups.forEach(group => {
    (group.params || []).forEach(p => {
      const inputEl = document.getElementById(`setting_${p.key}`);
      const badgeEl = document.getElementById(`badge_${p.key}`);
      if (!inputEl || !badgeEl) return;

      const updateBadge = () => {
        const unit = inputEl.dataset.unit ? ` ${inputEl.dataset.unit}` : '';
        let displayVal = inputEl.value;
        if (p.type === "slider" && p.step && p.step < 1) {
          const decimals = (String(p.step).split(".")[1] || "").length;
          displayVal = Number(inputEl.value).toFixed(decimals);
        }
        badgeEl.textContent = `${displayVal}${unit}`;
      };

      inputEl.addEventListener("input", updateBadge);
      inputEl.addEventListener("change", updateBadge);
    });
  });
}

function updateModelSettingsValues(config) {
  if (!config) return;
  for (const [k, v] of Object.entries(config)) {
    const inputEl = document.getElementById(`setting_${k}`);
    const badgeEl = document.getElementById(`badge_${k}`);
    if (inputEl) {
      inputEl.value = v;
      if (badgeEl) {
        const unit = inputEl.dataset.unit ? ` ${inputEl.dataset.unit}` : '';
        let displayVal = v;
        if (inputEl.dataset.type === "float" && typeof v === "number" && !Number.isInteger(v)) {
          displayVal = v.toFixed(2);
        }
        badgeEl.textContent = `${displayVal}${unit}`;
      }
    }
  }
}

formModelSettings.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideInlineAlert("modelSettingsAlert");
  modelSettingsProgressWrapper.style.display = "block";
  modelSettingsProgressStatus.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang áp dụng thông số vào các mô hình AI...';

  const payload = {};
  const formData = new FormData(formModelSettings);
  for (const [k, v] of formData.entries()) {
    const strVal = String(v).trim();
    if (!isNaN(strVal) && strVal !== "") {
      payload[k] = Number(strVal);
    } else {
      payload[k] = strVal;
    }
  }

  // Thu thập các switch/checkbox Bật/Tắt
  formModelSettings.querySelectorAll("input[type=checkbox]").forEach(cb => {
    payload[cb.name] = cb.checked;
  });

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không thể cập nhật cấu hình.");

    currentModelConfig = data.config || payload;
    updateInlineModelParametersUI(currentModelConfig);
    updateModelSettingsValues(currentModelConfig);
    sendToggleConfig();
    showInlineAlert("modelSettingsAlert", data.message || "Đã cập nhật các thông số model thành công!", "success");
    showNotification("Đã áp dụng các thông số Model thành công vào hệ thống!", "success", 3500);
  } catch (err) {
    showInlineAlert("modelSettingsAlert", `Lỗi: ${err.message}`, "error");
    showNotification(`Lỗi cập nhật thông số: ${err.message}`, "error", 5000);
  } finally {
    modelSettingsProgressWrapper.style.display = "none";
  }
});

btnResetModelSettings.addEventListener("click", async () => {
  if (!confirm("Bạn có chắc chắn muốn khôi phục toàn bộ thông số mô hình về giá trị tối ưu mặc định?")) return;

  hideInlineAlert("modelSettingsAlert");
  modelSettingsProgressWrapper.style.display = "block";
  modelSettingsProgressStatus.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang khôi phục cài đặt mặc định...';

  try {
    const res = await fetch("/api/config/reset", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không thể khôi phục cấu hình.");

    currentModelConfig = data.config || {};
    updateInlineModelParametersUI(currentModelConfig);
    updateModelSettingsValues(currentModelConfig);
    sendToggleConfig();
    showInlineAlert("modelSettingsAlert", data.message || "Đã khôi phục toàn bộ thông số mô hình về mặc định!", "success");
    showNotification("Đã khôi phục toàn bộ thông số về mặc định!", "info", 3500);
  } catch (err) {
    showInlineAlert("modelSettingsAlert", `Lỗi: ${err.message}`, "error");
    showNotification(`Lỗi khôi phục thông số: ${err.message}`, "error", 5000);
  } finally {
    modelSettingsProgressWrapper.style.display = "none";
  }
});

// Kiểm tra nếu có bản ghi âm microphone từ phiên trước đó chưa xóa
async function checkExistingRecording() {
  try {
    const res = await fetch("/api/recording/latest", { method: "HEAD" });
    if (res.ok) {
      const cacheBuster = Date.now();
      const recUrl = `/api/recording/latest?t=${cacheBuster}`;
      if (micAudioPreview && !micAudioPreview.src) {
        micAudioPreview.src = recUrl;
      }
      if (btnDownloadMicAudio) {
        btnDownloadMicAudio.href = recUrl;
        btnDownloadMicAudio.download = "recording_online.wav";
      }
      if (micAudioPreviewWrapper) {
        micAudioPreviewWrapper.style.display = "block";
      }
    }
  } catch (e) {
    // Không có bản ghi âm hoặc server chưa sẵn sàng
  }
}

// Start on Page Load
window.addEventListener("DOMContentLoaded", () => {
  initInlineModelParameters();
  initWebSocket();
  loadModelSettings(false);
  checkExistingRecording();
});
