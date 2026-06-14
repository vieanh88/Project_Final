/* ============================================================
   Audiobook Studio — frontend logic (vanilla JS)
   ============================================================ */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const ROLES = ["narrator", "character_male", "character_female"];

const state = {
  script: [],        // [{id, role, text, pause_after_ms}]
  voices: [],
  config: null,
};

/* ---------- helpers ---------- */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.detail || JSON.stringify(j); } catch (_) {}
    throw new Error(msg);
  }
  return res;
}
const apiJson = async (path, opts) => (await api(path, opts)).json();
const postJson = (path, body) =>
  apiJson(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.innerHTML = `<b>${type === "err" ? "Lỗi" : type === "ok" ? "Xong" : "Thông báo"}</b>${msg}`;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transform = "translateY(8px)"; }, 4200);
  setTimeout(() => el.remove(), 4700);
}
function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}
function escapeHtml(s) { return (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
const accentColor = () => getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#7C5CFF";
function setLoading(btn, on) { btn.classList.toggle("loading", on); btn.disabled = on; }
function setSelectValue(sel, val) {
  // Chỉ đồng bộ nếu model trong config nằm trong list options.
  // Nếu không (vd config dùng model ngoài 5 lựa chọn) -> GIỮ option mặc định
  // đã đánh dấu `selected` trong HTML.
  if (val && [...sel.options].some(o => o.value === val)) sel.value = val;
}

/* ============================================================
   INIT
   ============================================================ */
document.addEventListener("DOMContentLoaded", () => {
  wireTabs();
  wireStoryInputs();
  wireScriptCard();
  wireVoiceUploads();
  wireSynth();
  wirePlayground();
  bootstrap();
  window.addEventListener("resize", () => players.forEach(p => p.redraw()));
});

async function bootstrap() {
  await Promise.allSettled([loadHealth(), loadConfig(), loadVoices(), loadSamples()]);
}

/* ---------- health ---------- */
async function loadHealth(tries = 0) {
  try {
    const h = await apiJson("/api/health");
    const dot = $("#statusDot");
    dot.className = "dot " + (h.ready ? "ready" : "error");
    $("#statusText").textContent = h.ready ? "Engine sẵn sàng" : "Engine chưa sẵn sàng";
    $("#statusSub").textContent = h.cuda ? (h.gpu_name || "CUDA") : "CPU";
    const ep = h.trained_epoch != null ? `epoch ${h.trained_epoch}` : "";
    $("#footInfo").textContent = [h.device, ep, h.n_params_million ? `${h.n_params_million}M params` : ""]
      .filter(Boolean).join(" · ");
  } catch (e) {
    if (tries < 8) return setTimeout(() => loadHealth(tries + 1), 1200);
    $("#statusDot").className = "dot error";
    $("#statusText").textContent = "Không kết nối được engine";
  }
}

/* ---------- config ---------- */
async function loadConfig() {
  const c = await apiJson("/api/config");
  state.config = c;
  setSelectValue($("#nlpModel"), c.nlp.model);
  $("#nlpThinking").checked = !!c.nlp.thinking;
  $("#nlpChunk").value = 10000;
  $("#denoise").value = c.tts.denoise ?? 0.3;
  $("#splitDur").value = c.tts.split_dur ?? 2.0;
  $("#skipErr").checked = c.tts.skip_on_error ?? true;
  $("#normalize").checked = c.tts.normalize ?? true;
  setSlider($("#narSpeed"), $("#narValLabel"), c.tts.narrator_speed ?? 1);
  setSlider($("#chrSpeed"), $("#chrValLabel"), c.tts.character_speed ?? 1);
}

/* ---------- voices ---------- */
async function loadVoices() {
  const { voices } = await apiJson("/api/voices");
  state.voices = voices;
  fillVoiceSelect($("#maleVoice"), "male");
  fillVoiceSelect($("#femaleVoice"), "female");
  fillVoiceSelect($("#pgVoice"), null);
}
function fillVoiceSelect(sel, roleFilter) {
  const prev = sel.value;
  sel.innerHTML = "";
  state.voices.forEach(v => {
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = v.label + (v.duration ? ` (${v.duration}s)` : "");
    sel.appendChild(o);
  });
  // chọn mặc định theo role nếu chưa có lựa chọn trước
  if (prev && state.voices.some(v => v.id === prev)) { sel.value = prev; return; }
  const pref = state.voices.find(v => roleFilter ? v.role === roleFilter : true);
  if (pref) sel.value = pref.id;
}

/* ---------- sample stories ---------- */
async function loadSamples() {
  const { stories } = await apiJson("/api/sample-stories");
  const sel = $("#sampleSelect");
  stories.forEach(s => {
    const o = document.createElement("option");
    o.value = s.name;
    o.textContent = `${s.name} (${s.chars.toLocaleString()} ký tự)`;
    sel.appendChild(o);
  });
  sel.addEventListener("change", async () => {
    if (!sel.value) return;
    try {
      const { text } = await apiJson(`/api/sample-stories/${encodeURIComponent(sel.value)}`);
      $("#storyText").value = text;
      updateCharCount();
    } catch (e) { toast(e.message, "err"); }
  });
}

/* ============================================================
   TABS
   ============================================================ */
function wireTabs() {
  $$(".tab").forEach(tab => tab.addEventListener("click", () => {
    $$(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    $("#tab-audiobook").hidden = tab.dataset.tab !== "audiobook";
    $("#tab-playground").hidden = tab.dataset.tab !== "playground";
  }));
}

/* ============================================================
   STORY INPUTS
   ============================================================ */
function wireStoryInputs() {
  const ta = $("#storyText");
  ta.addEventListener("input", updateCharCount);
  $("#txtUpload").addEventListener("change", async e => {
    const f = e.target.files[0]; if (!f) return;
    ta.value = await f.text();
    updateCharCount();
    e.target.value = "";
  });
  $("#genScriptBtn").addEventListener("click", generateScript);
}
function updateCharCount() {
  $("#charCount").textContent = `${$("#storyText").value.length.toLocaleString()} ký tự`;
}

async function generateScript() {
  const text = $("#storyText").value.trim();
  if (!text) return toast("Hãy nhập văn bản truyện trước.", "err");
  const btn = $("#genScriptBtn");
  setLoading(btn, true);
  toast("Đang gọi Gemini sinh kịch bản…");
  try {
    const body = {
      text,
      model: $("#nlpModel").value || null,
      thinking: $("#nlpThinking").checked,
      chunk_size: parseInt($("#nlpChunk").value) || null,
    };
    const res = await postJson("/api/generate-script", body);
    state.script = res.lines.map((l, i) => ({ ...l, id: i + 1 }));
    renderScript();
    showStep("#scriptCard"); showStep("#voiceCard"); showStep("#synthCard");
    const s = res.stats;
    $("#scriptStats").textContent =
      `${s.n_lines} câu · ~${s.estimated_audio_min} phút · ` +
      Object.entries(s.role_counts).map(([k, v]) => `${k}:${v}`).join("  ");
    toast(`Sinh ${s.n_lines} câu kịch bản.`, "ok");
    $("#scriptCard").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    toast(e.message, "err");
  } finally { setLoading(btn, false); }
}
function showStep(sel) { $(sel).hidden = false; }

/* ============================================================
   SCRIPT TABLE (editable)
   ============================================================ */
function wireScriptCard() {
  $("#addLineBtn").addEventListener("click", () => {
    state.script.push({ id: state.script.length + 1, role: "narrator", text: "", pause_after_ms: 500 });
    renderScript();
  });
}
function makeRoleSelect(current) {
  const sel = document.createElement("select");
  sel.className = "role-sel role-" + current;
  ROLES.forEach(r => {
    const o = document.createElement("option");
    o.value = r; o.textContent = r; if (r === current) o.selected = true;
    sel.appendChild(o);
  });
  return sel;
}
function renderScript() {
  const body = $("#scriptBody");
  body.innerHTML = "";
  state.script.forEach((line, i) => {
    line.id = i + 1;
    const tr = document.createElement("tr");

    const tdIdx = document.createElement("td");
    tdIdx.className = "idx"; tdIdx.textContent = i + 1;

    const tdRole = document.createElement("td");
    const roleSel = makeRoleSelect(line.role);
    roleSel.addEventListener("change", () => {
      line.role = roleSel.value;
      roleSel.className = "role-sel role-" + line.role;
    });
    tdRole.appendChild(roleSel);

    const tdText = document.createElement("td");
    const txt = document.createElement("textarea");
    txt.className = "line-text"; txt.rows = 1; txt.value = line.text;
    autoGrow(txt);
    txt.addEventListener("input", () => { line.text = txt.value; autoGrow(txt); });
    tdText.appendChild(txt);

    const tdPause = document.createElement("td");
    const pause = document.createElement("input");
    pause.className = "pause"; pause.type = "number"; pause.min = "0"; pause.max = "5000"; pause.step = "50";
    pause.value = line.pause_after_ms;
    pause.addEventListener("input", () => { line.pause_after_ms = parseInt(pause.value) || 0; });
    tdPause.appendChild(pause);

    const tdDel = document.createElement("td");
    const del = document.createElement("button");
    del.className = "del-btn"; del.title = "Xoá dòng"; del.textContent = "✕";
    del.addEventListener("click", () => { state.script.splice(i, 1); renderScript(); });
    tdDel.appendChild(del);

    tr.append(tdIdx, tdRole, tdText, tdPause, tdDel);
    body.appendChild(tr);
  });
}
function autoGrow(ta) { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 120) + "px"; }

/* ============================================================
   VOICE UPLOAD
   ============================================================ */
function wireVoiceUploads() {
  $$(".voice-upload").forEach(inp => inp.addEventListener("change", async e => {
    const f = e.target.files[0]; if (!f) return;
    const role = inp.dataset.role;
    const note = $(`.voice-note[data-note="${role}"]`);
    note.textContent = "Đang kiểm tra file…"; note.className = "voice-note";
    const fd = new FormData(); fd.append("file", f); fd.append("role", role);
    try {
      const res = await apiJson("/api/voices/upload", { method: "POST", body: fd });
      await loadVoices();
      const sel = role === "male" ? $("#maleVoice") : $("#femaleVoice");
      sel.value = res.voice.id;
      const w = (res.health.warnings || []).length;
      note.textContent = `✓ ${res.health.duration_sec}s · ${w} cảnh báo`;
      note.className = "voice-note " + (w ? "warn" : "ok");
      toast(`Đã thêm giọng ${role === "male" ? "nam" : "nữ"}.`, "ok");
    } catch (err) {
      note.textContent = "✕ " + err.message; note.className = "voice-note warn";
      toast(err.message, "err");
    }
    e.target.value = "";
  }));
}

/* ============================================================
   SYNTHESIZE (audiobook job + polling)
   ============================================================ */
function wireSynth() {
  bindSlider($("#narSpeed"), $("#narValLabel"));
  bindSlider($("#chrSpeed"), $("#chrValLabel"));
  $("#synthBtn").addEventListener("click", synthesize);
}
function setSlider(input, label, val) { input.value = val; label.textContent = Number(val).toFixed(2); }
function bindSlider(input, label) { input.addEventListener("input", () => label.textContent = Number(input.value).toFixed(2)); }

async function synthesize() {
  if (!state.script.length) return toast("Chưa có kịch bản.", "err");
  const btn = $("#synthBtn");
  setLoading(btn, true);
  $("#resultArea").hidden = true;
  $("#progressArea").hidden = false;
  setProgress(0, "Đang gửi yêu cầu…", "", "");
  try {
    const body = {
      script: state.script,
      male_voice_id: $("#maleVoice").value,
      female_voice_id: $("#femaleVoice").value,
      narrator_speed: parseFloat($("#narSpeed").value),
      character_speed: parseFloat($("#chrSpeed").value),
      denoise: parseFloat($("#denoise").value),
      split_dur: parseFloat($("#splitDur").value),
      skip_on_error: $("#skipErr").checked,
      normalize: $("#normalize").checked,
    };
    const { job_id } = await postJson("/api/synthesize", body);
    await pollJob(job_id);
  } catch (e) {
    toast(e.message, "err");
    $("#progressArea").hidden = true;
  } finally { setLoading(btn, false); }
}

function setProgress(ratio, stage, count, last) {
  $("#progBar").style.width = Math.round(ratio * 100) + "%";
  $("#progStage").textContent = stage;
  $("#progCount").textContent = count;
  $("#progLast").innerHTML = last;
}

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const job = await apiJson(`/api/jobs/${jobId}`);
        const last = job.last
          ? `<span class="tag role-${job.last.role}" style="color:var(--${roleVar(job.last.role)})">${job.last.role}</span> · ${escapeHtml(job.last.text_preview || "")}`
          : "";
        setProgress(job.progress || 0, job.stage || "", job.total ? `${job.done}/${job.total}` : "", last);
        if (job.status === "done") { onSynthDone(job.result); return resolve(); }
        if (job.status === "error") { $("#progressArea").hidden = true; toast(job.error || "Job lỗi", "err"); return reject(new Error(job.error)); }
        setTimeout(tick, 700);
      } catch (e) { setTimeout(tick, 1200); }
    };
    tick();
  });
}
function roleVar(role) { return role === "narrator" ? "narrator" : role === "character_female" ? "female" : "male"; }

function onSynthDone(result) {
  $("#progressArea").hidden = true;
  $("#resultArea").hidden = false;
  $("#dlAudiobook").href = result.audio_url;
  const s = result.summary;
  $("#abSummary").innerHTML = [
    ["Thời lượng", `${s.total_audio_min} phút`],
    ["Số câu", s.total_lines],
    ["Thành công", s.ok_lines],
    ["Lỗi", s.failed_lines],
    ["Dung lượng", `${s.size_mb} MB`],
  ].map(([k, v]) => `<div class="stat-box"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  buildPlayer($("#abPlayer"), result.audio_url, "ab");
  $("#resultArea").scrollIntoView({ behavior: "smooth", block: "center" });
  toast("Audiobook đã sẵn sàng!", "ok");
}

/* ============================================================
   PLAYGROUND
   ============================================================ */
const PG_SAMPLES = [
  "Đêm hôm ấy, trời tối đen như mực, không một tiếng động.",
  "Anh ơi, anh có nghe thấy tiếng gì ngoài hành lang không?",
  "Trời ơi, ai vừa gọi tên tôi vậy? Lạnh quá!",
  "Tôi quay đầu lại, và nó đã đứng ngay sau lưng tôi.",
];
function wirePlayground() {
  bindSlider($("#pgSpeed"), $("#pgSpeedLabel"));
  const chips = $("#pgChips");
  PG_SAMPLES.forEach(t => {
    const b = document.createElement("button");
    b.className = "chip-btn"; b.textContent = t.length > 42 ? t.slice(0, 42) + "…" : t;
    b.addEventListener("click", () => { $("#pgText").value = t; });
    chips.appendChild(b);
  });
  $("#pgBtn").addEventListener("click", playground);
}
async function playground() {
  const text = $("#pgText").value.trim();
  if (!text) return toast("Nhập câu cần đọc.", "err");
  const voice = $("#pgVoice").value;
  if (!voice) return toast("Chưa có giọng nào.", "err");
  const btn = $("#pgBtn");
  setLoading(btn, true);
  try {
    const res = await api("/api/playground", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text, voice_id: voice, speed: parseFloat($("#pgSpeed").value),
        denoise: parseFloat($("#denoise").value) || 0.3,
        split_dur: parseFloat($("#splitDur").value) || 2.0,
      }),
    });
    const blob = await res.blob();
    $("#pgResult").hidden = false;
    buildPlayer($("#pgPlayer"), URL.createObjectURL(blob), "pg");
  } catch (e) { toast(e.message, "err"); }
  finally { setLoading(btn, false); }
}

/* ============================================================
   AUDIO PLAYER + WAVEFORM
   ============================================================ */
const players = [];   // {redraw}
async function buildPlayer(container, url, key) {
  container.innerHTML = "";
  const wrap = document.createElement("div"); wrap.className = "player";
  const playBtn = document.createElement("button"); playBtn.className = "play-btn"; playBtn.textContent = "▶";
  const waveWrap = document.createElement("div"); waveWrap.className = "wave-wrap";
  const canvas = document.createElement("canvas"); canvas.className = "wave";
  const time = document.createElement("div"); time.className = "time"; time.textContent = "0:00 / 0:00";
  waveWrap.appendChild(canvas);
  wrap.append(playBtn, waveWrap, time);
  container.appendChild(wrap);

  const audio = new Audio(url);
  audio.preload = "auto";
  let peaks = null, progress = 0;

  const redraw = () => { if (peaks) drawWave(canvas, peaks, progress); };
  const slot = { redraw };
  players[key === "ab" ? 0 : 1] = slot;

  // decode for waveform
  try {
    const buf = await (await fetch(url)).arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await ctx.decodeAudioData(buf);
    peaks = computePeaks(audioBuf.getChannelData(0), 360);
    ctx.close();
    redraw();
  } catch (_) { /* fallback: no waveform */ }

  playBtn.addEventListener("click", () => { audio.paused ? audio.play() : audio.pause(); });
  audio.addEventListener("play", () => playBtn.textContent = "⏸");
  audio.addEventListener("pause", () => playBtn.textContent = "▶");
  audio.addEventListener("ended", () => { playBtn.textContent = "▶"; progress = 0; redraw(); });
  audio.addEventListener("loadedmetadata", () => time.textContent = `0:00 / ${fmtTime(audio.duration)}`);
  audio.addEventListener("timeupdate", () => {
    progress = audio.duration ? audio.currentTime / audio.duration : 0;
    time.textContent = `${fmtTime(audio.currentTime)} / ${fmtTime(audio.duration)}`;
    redraw();
  });
  canvas.addEventListener("click", e => {
    const r = canvas.getBoundingClientRect();
    if (audio.duration) audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
  });
}
function computePeaks(data, buckets) {
  const block = Math.floor(data.length / buckets) || 1;
  const peaks = new Array(buckets); let max = 1e-4;
  for (let b = 0; b < buckets; b++) {
    let m = 0; const start = b * block;
    for (let i = 0; i < block; i++) { const v = Math.abs(data[start + i] || 0); if (v > m) m = v; }
    peaks[b] = m; if (m > max) max = m;
  }
  return peaks.map(p => p / max);
}
function drawWave(canvas, peaks, progress) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 56;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const n = peaks.length, bw = W / n, mid = H / 2, acc = accentColor();
  for (let i = 0; i < n; i++) {
    const h = Math.max(2, peaks[i] * H * 0.92);
    ctx.fillStyle = (i / n) <= progress ? acc : "rgba(255,255,255,0.13)";
    const x = i * bw + 0.5, w = Math.max(1, bw - 1.5);
    const r = Math.min(w / 2, 2), y = mid - h / 2;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
    ctx.fill();
  }
}
