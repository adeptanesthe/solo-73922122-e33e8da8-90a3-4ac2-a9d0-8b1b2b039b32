/* 野生动物追踪串号校正与计数台 —— 前端逻辑 */
"use strict";

const PALETTE = [
  "#ff4d4f", "#36cfc9", "#ffa940", "#73d13d", "#9254de",
  "#40a9ff", "#f759ab", "#a0d911", "#ffc53d", "#597ef7",
  "#ff7a45", "#5cdbd3", "#b37feb", "#95de64", "#69c0ff",
];

const state = {
  sessionId: null,
  version: 0,
  nFrames: 0,
  detections: [],
  judgments: [],          // [{a, b, type}]，a/b 为框号
  params: { w_dist: 1, w_app: 10000, penalty: 15000 },
  images: [],             // HTMLImageElement
  currentFrame: 0,
  selection: [],          // 已选框号（至多 2 个）
  result: null,           // 当前展示的求解结果
  jobId: null,
  pollTimer: null,
  // 点选分割（独立于追踪求解；不改动检测 JSON、判断、轨迹与计数）
  seg: {
    active: false,        // 是否处于点选分割模式
    targetId: null,       // 当前选中的检测框 id
    frame: -1,            // 目标框所在帧
    fg: [],               // [{x, y}] 原图像素坐标
    bg: [],
    order: [],            // 标点添加顺序（"fg"/"bg"），供撤销
    mask: null,           // 已接受的分割结果
    seq: 0,               // 标点令牌：仅与当前令牌一致的结果可落图
    busy: false,
    timer: null,          // 去抖定时器
    pollTimer: null,      // 任务轮询定时器（与去抖分开，互不误清）
  },
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const detById = (id) => state.detections.find((d) => d.id === id);
const pairKey = (a, b) => [a, b].sort().join("||");

function trajIndexOf(boxId) {
  if (!state.result || !state.result.feasible) return null;
  for (let i = 0; i < state.result.trajectories.length; i++) {
    if (state.result.trajectories[i].boxes.some((b) => b.id === boxId)) return i;
  }
  return null;
}

/* ---------------------------------------------------------------- API */

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return resp.json();
}

function showError(targetId, err) {
  const el = $(targetId);
  el.textContent = `错误：${err.message}`;
  el.classList.remove("muted");
  el.style.color = "#ff9d97";
}

/* ------------------------------------------------------------ 会话载入 */

function loadSession(payload) {
  state.sessionId = payload.session_id;
  state.version = payload.version;
  state.nFrames = payload.n_frames;
  state.detections = payload.detections;
  state.judgments = payload.judgments;
  state.params = payload.params;
  state.currentFrame = 0;
  state.selection = [];
  state.result = null;
  state.jobId = null;
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
  resetSegmentation(true);

  state.images = [];
  for (let i = 0; i < state.nFrames; i++) {
    const img = new Image();
    img.src = `/api/sessions/${state.sessionId}/frames/${i}`;
    img.onload = () => { if (i === state.currentFrame) draw(); };
    state.images.push(img);
  }

  $("p-wdist").value = payload.params.w_dist;
  $("p-wapp").value = payload.params.w_app;
  $("p-penalty").value = payload.params.penalty;

  const info = $("session-info");
  info.classList.add("muted");
  info.style.color = "";
  info.innerHTML =
    `会话 <b>${payload.session_id}</b>：${payload.n_frames} 帧、` +
    `${payload.detections.length} 个检测框。` +
    (payload.description ? `<br>${payload.description}` : "");

  $("solve-btn").disabled = false;
  $("no-session").style.display = "none";
  renderAll();
}

async function uploadSession() {
  const imgs = $("img-input").files;
  const jsons = $("json-input").files;
  if (imgs.length === 0 || imgs.length > 10) {
    alert("请选择 1~10 张帧图片。");
    return;
  }
  if (jsons.length !== 1) {
    alert("请选择 1 个检测 JSON 文件。");
    return;
  }
  const fd = new FormData();
  for (const f of imgs) fd.append("images", f);
  fd.append("detections", jsons[0]);
  try {
    const payload = await api("/api/sessions", { method: "POST", body: fd });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

async function loadExample() {
  try {
    const payload = await api("/api/example", { method: "POST" });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

/* ------------------------------------------------------------ 画面渲染 */

function detsInFrame(f) {
  return state.detections.filter((d) => d.frame === f);
}

function draw() {
  const img = state.images[state.currentFrame];
  if (!img || !img.complete || img.naturalWidth === 0) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  for (const d of detsInFrame(state.currentFrame)) {
    const [x, y, w, h] = d.rect;
    const ti = trajIndexOf(d.id);
    const color = ti !== null ? PALETTE[ti % PALETTE.length] : "#ffe600";
    const selected = state.selection.includes(d.id);

    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeStyle = selected ? "#ffffff" : color;
    ctx.strokeRect(x, y, w, h);

    const label = `#${d.id}` + (ti !== null ? ` T${ti + 1}` : "");
    ctx.font = "bold 12px sans-serif";
    const tw = ctx.measureText(label).width;
    const ly = y >= 16 ? y - 16 : y + h + 2;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(x, ly, tw + 6, 15);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 3, ly + 12);
  }

  drawSegmentOverlay();
}

/* ------------------------------------------------------------ 点选分割渲染 */

function decodeMaskRLE(rle, width, height) {
  const n = width * height;
  const mask = new Uint8Array(n);
  let pos = 0;
  for (let i = 0; i < rle.length; i++) {
    const len = rle[i];
    if (i % 2 === 1) { // off/on 交替，奇数段为前景
      for (let k = 0; k < len; k++) mask[pos + k] = 1;
    }
    pos += len;
  }
  if (pos !== n) throw new Error("掩膜数据长度与裁剪尺寸不一致。");
  return mask;
}

function drawSegmentOverlay() {
  const seg = state.seg;
  if (!seg.active || !seg.targetId) return;
  const det = detById(seg.targetId);
  if (!det) return;
  const [bx, by] = det.rect;

  // 目标检测框高亮
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#19e6c9";
  ctx.setLineDash([7, 5]);
  ctx.strokeRect(bx, by, det.rect[2], det.rect[3]);
  ctx.restore();

  // 掩膜半透明叠加
  if (seg.mask && seg.mask.ok) {
    const { crop, mask_rle: rle } = seg.mask;
    const cw = crop.width, ch = crop.height;
    const mask = decodeMaskRLE(rle, cw, ch);
    const overlay = ctx.createImageData(cw, ch);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) {
        overlay.data[i * 4] = 25;
        overlay.data[i * 4 + 1] = 230;
        overlay.data[i * 4 + 2] = 201;
        overlay.data[i * 4 + 3] = 120;
      }
    }
    ctx.putImageData(overlay, crop.x, crop.y);

    // 轮廓：掩膜边界像素（与裁剪外/相邻背景像素接边的掩膜像素）描边，
    // 外边界与内孔边界都会显示。
    ctx.save();
    ctx.fillStyle = "#a7ffef";
    for (let yy = 0; yy < ch; yy++) {
      for (let xx = 0; xx < cw; xx++) {
        if (!mask[yy * cw + xx]) continue;
        let boundary = false;
        for (let dy = -1; dy <= 1 && !boundary; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = xx + dx, ny = yy + dy;
            if (nx < 0 || ny < 0 || nx >= cw || ny >= ch || !mask[ny * cw + nx]) {
              boundary = true;
              break;
            }
          }
        }
        if (boundary) ctx.fillRect(crop.x + xx, crop.y + yy, 1, 1);
      }
    }
    ctx.restore();
  }

  // 标点
  for (const p of seg.fg) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = "#ff5a52";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
  for (const p of seg.bg) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = "#59a8ff";
    ctx.fillRect(-3.4, -3.4, 6.8, 6.8);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#fff";
    ctx.strokeRect(-3.4, -3.4, 6.8, 6.8);
    ctx.restore();
  }
}

function renderFrameNav() {
  const n = state.nFrames || 0;
  $("frame-indicator").textContent =
    n === 0 ? "— / —" : `第 ${state.currentFrame + 1} / ${n} 帧（本帧 ${detsInFrame(state.currentFrame).length} 框）`;
  $("prev-frame").disabled = n === 0 || state.currentFrame === 0;
  $("next-frame").disabled = n === 0 || state.currentFrame >= n - 1;
}

function switchFrame(delta) {
  const next = state.currentFrame + delta;
  if (next < 0 || next >= state.nFrames) return;
  state.currentFrame = next;
  // 换帧即撤下旧掩膜与标点（目标框属于旧帧）
  resetSegmentation(true);
  renderFrameNav();
  draw();
}

/* ------------------------------------------------------------ 选择与标记 */

function renderSelection() {
  const info = $("selection-info");
  if (state.selection.length === 0) {
    info.textContent = "未选择框。点击画面中的检测框进行选择（可跨帧选两个）。";
  } else {
    const parts = state.selection.map((id) => {
      const d = detById(id);
      return d ? `帧 ${d.frame} 的 #${id}` : `#${id}`;
    });
    let text = `已选：${parts.join(" 与 ")}`;
    if (state.selection.length === 2) {
      const j = findJudgment(state.selection[0], state.selection[1]);
      if (j) text += `（当前判断：${j.type === "same" ? "同只" : "异只"}）`;
    }
    info.textContent = text;
  }
  const two = state.selection.length === 2;
  $("mark-same").disabled = !two;
  $("mark-diff").disabled = !two;
  $("mark-clear").disabled = !two || !findJudgment(state.selection[0], state.selection[1]);
}

function findJudgment(a, b) {
  const key = pairKey(a, b);
  return state.judgments.find((j) => pairKey(j.a, j.b) === key) || null;
}

function onCanvasClick(ev) {
  if (!state.sessionId) return;
  const { px, py } = eventPixels(ev);
  const dets = detsInFrame(state.currentFrame);

  if (state.seg.active) {
    handleSegPick(px, py, "fg");
    return;
  }

  for (let i = dets.length - 1; i >= 0; i--) {
    const [x, y, w, h] = dets[i].rect;
    if (px >= x && px <= x + w && py >= y && py <= y + h) {
      toggleSelect(dets[i].id);
      return;
    }
  }
}

function eventPixels(ev) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return {
    px: (ev.clientX - rect.left) * sx,
    py: (ev.clientY - rect.top) * sy,
  };
}

function onCanvasContextMenu(ev) {
  if (!state.sessionId || !state.seg.active) return;
  ev.preventDefault();
  const { px, py } = eventPixels(ev);
  handleSegPick(px, py, "bg");
}

/* ------------------------------------------------------------ 点选分割逻辑 */

function resetSegmentation(keepMode) {
  const seg = state.seg;
  if (seg.timer) { clearTimeout(seg.timer); seg.timer = null; }
  if (seg.pollTimer) { clearTimeout(seg.pollTimer); seg.pollTimer = null; }
  // 令牌递增：任何在途/迟到结果都不再被接受，不能覆盖新标点
  seg.seq += 1;
  seg.busy = false;
  seg.targetId = null;
  seg.frame = -1;
  seg.fg = [];
  seg.bg = [];
  seg.order = [];  // 添加顺序，供撤销使用
  seg.mask = null;
  if (!keepMode) seg.active = false;
  renderSegmentPanel();
}

function toggleSegmentMode() {
  const seg = state.seg;
  seg.active = !seg.active;
  resetSegmentation(true); // 进入/退出都撤下旧掩膜与标点
  renderSegmentPanel();
  draw();
}

function pickDetectionAt(px, py) {
  const dets = detsInFrame(state.currentFrame);
  for (let i = dets.length - 1; i >= 0; i--) {
    const [x, y, w, h] = dets[i].rect;
    if (px >= x && px <= x + w && py >= y && py <= y + h) return dets[i];
  }
  return null;
}

function handleSegPick(px, py, kind) {
  const seg = state.seg;
  const det = pickDetectionAt(px, py);
  if (!det) return; // 框外点击忽略

  // 换框即撤下旧掩膜与标点
  if (seg.targetId !== det.id) {
    if (seg.timer) { clearTimeout(seg.timer); seg.timer = null; }
    if (seg.pollTimer) { clearTimeout(seg.pollTimer); seg.pollTimer = null; }
    seg.seq += 1;
    seg.busy = false;
    seg.targetId = det.id;
    seg.frame = det.frame;
    seg.fg = [];
    seg.bg = [];
    seg.order = [];
    seg.mask = null;
  }

  if (kind === "fg") {
    seg.fg.push({ x: px, y: py });
  } else {
    seg.bg.push({ x: px, y: py });
  }
  seg.order.push(kind);
  $("seg-result").style.display = "none"; // 新标点：撤下旧结论/旧报错
  scheduleSegmentation();
  renderSegmentPanel();
  draw();
}

function undoLastPoint() {
  const seg = state.seg;
  const kind = seg.order.pop();
  if (kind === "fg") seg.fg.pop();
  else if (kind === "bg") seg.bg.pop();
  else return;
  invalidateSegmentMask();
  renderSegmentPanel();
  draw();
}

function clearSegmentPoints() {
  resetSegmentation(true);
  draw();
}

function invalidateSegmentMask() {
  const seg = state.seg;
  if (seg.timer) { clearTimeout(seg.timer); seg.timer = null; }
  if (seg.pollTimer) { clearTimeout(seg.pollTimer); seg.pollTimer = null; }
  // 改点即撤下旧掩膜，并使在途结果作废
  seg.seq += 1;
  seg.busy = false;
  seg.mask = null;
}

function scheduleSegmentation() {
  const seg = state.seg;
  invalidateSegmentMask();
  if (seg.fg.length < 2 || seg.bg.length < 1) {
    renderSegmentPanel();
    return; // 标点不足，不发请求
  }
  const seq = seg.seq;
  seg.timer = setTimeout(() => submitSegmentation(seq), 250);
  renderSegmentPanel();
}

async function submitSegmentation(seq) {
  const seg = state.seg;
  if (!seg.targetId || seq !== seg.seq) return;
  seg.busy = true;
  renderSegmentPanel();
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/segment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        detection_id: seg.targetId,
        foreground: seg.fg,
        background: seg.bg,
      }),
    });
    pollSegmentJob(resp.job_id, seq);
  } catch (err) {
    if (seq !== seg.seq) return; // 已被新标点取代
    seg.busy = false;
    showSegmentBanner(`分割请求失败：${err.message}`, "err");
    renderSegmentPanel();
  }
}

async function pollSegmentJob(jobId, seq) {
  const seg = state.seg;
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (seq !== seg.seq) return; // 迟到结果：标点/框/帧已变化，丢弃不覆盖
    if (job.status === "running") {
      seg.pollTimer = setTimeout(() => pollSegmentJob(jobId, seq), 300);
      return;
    }
    seg.busy = false;
    if (job.status === "error") {
      showSegmentBanner(job.error || "分割失败。", "err");
      return;
    }
    const result = job.result;
    if (!result || !result.ok) {
      // 不能可靠分割：不显示面积
      seg.mask = null;
      showSegmentBanner(result ? result.message : "分割失败，不提供面积。", "err");
      draw();
      return;
    }
    seg.mask = result;
    showSegmentBanner(
      `分割成功：动物实际可见像素面积（原图）= <b>${result.area_px.toLocaleString()}</b> px` +
      `（裁剪 ${result.crop.width}×${result.crop.height}，掩膜为单块连通区域）。`,
      "ok",
    );
    draw();
  } catch (err) {
    if (seq !== seg.seq) return;
    seg.busy = false;
    showSegmentBanner(`分割查询失败：${err.message}`, "err");
    renderSegmentPanel();
  }
}

function showSegmentBanner(message, kind) {
  const el = $("seg-result");
  el.style.display = "";
  el.className = `banner ${kind === "ok" ? "ok" : "err"}`;
  el.innerHTML = message;
}

function renderSegmentPanel() {
  const seg = state.seg;
  const hasSession = Boolean(state.sessionId);
  $("seg-mode-btn").disabled = !hasSession;
  $("seg-mode-btn").textContent = seg.active ? "退出点选分割" : "进入点选分割";
  $("seg-mode-btn").classList.toggle("primary", seg.active);
  $("seg-undo").disabled = !hasSession || !seg.active || (!seg.fg.length && !seg.bg.length);
  $("seg-clear").disabled = !hasSession || !seg.active ||
    (!seg.fg.length && !seg.bg.length && !seg.mask);

  const status = $("seg-status");
  const banner = $("seg-result");
  if (!seg.active) {
    status.textContent = "未开启点选分割。";
    banner.style.display = "none";
    return;
  }
  if (!seg.targetId) {
    status.textContent = "请点击当前帧中的一个检测框作为分割目标。";
    banner.style.display = "none";
    return;
  }
  const needFg = Math.max(0, 2 - seg.fg.length);
  const needBg = Math.max(0, 1 - seg.bg.length);
  const missing = [];
  if (needFg) missing.push(`还需 ${needFg} 个前景点`);
  if (needBg) missing.push(`还需 ${needBg} 个背景点`);
  let text = `目标：帧 ${seg.frame} 的 #${seg.targetId}；前景点 ${seg.fg.length} 个、背景点 ${seg.bg.length} 个。`;
  if (missing.length) {
    text += missing.join("，") + "（左键加前景，右键加背景）。";
  } else if (seg.busy) {
    text += " 正在分割…";
  } else if (!seg.mask) {
    text += " 等待分割结果。";
  }
  status.textContent = text;
}

function toggleSelect(id) {
  const i = state.selection.indexOf(id);
  if (i >= 0) {
    state.selection.splice(i, 1);
  } else {
    if (state.selection.length >= 2) state.selection.shift();
    state.selection.push(id);
  }
  renderSelection();
  draw();
}

async function setJudgment(type) {
  if (state.selection.length !== 2) return;
  const [a, b] = state.selection;
  const key = pairKey(a, b);
  const rest = state.judgments.filter((j) => pairKey(j.a, j.b) !== key);
  const next = type ? [...rest, { a, b, type }] : rest;
  invalidateResult(); // 修改即撤下旧结果
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/judgments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judgments: next }),
    });
    state.version = resp.version;
    state.judgments = next;
    renderJudgments();
    renderSelection();
    draw();
  } catch (err) {
    alert(`保存判断失败：${err.message}`);
  }
}

async function removeJudgment(a, b) {
  const key = pairKey(a, b);
  const next = state.judgments.filter((j) => pairKey(j.a, j.b) !== key);
  invalidateResult();
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/judgments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judgments: next }),
    });
    state.version = resp.version;
    state.judgments = next;
    renderJudgments();
    renderSelection();
    draw();
  } catch (err) {
    alert(`删除判断失败：${err.message}`);
  }
}

function renderJudgments() {
  const ul = $("judgment-list");
  ul.innerHTML = "";
  $("judgment-count").textContent = `（${state.judgments.length} 条）`;
  for (const j of state.judgments) {
    const da = detById(j.a);
    const db = detById(j.b);
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = `tag ${j.type === "same" ? "same" : "diff"}`;
    tag.textContent = j.type === "same" ? "同只" : "异只";
    const text = document.createElement("span");
    text.textContent = `帧${da ? da.frame : "?"} #${j.a} ↔ 帧${db ? db.frame : "?"} #${j.b}`;
    const del = document.createElement("button");
    del.textContent = "删除";
    del.onclick = () => removeJudgment(j.a, j.b);
    li.append(tag, text, del);
    ul.appendChild(li);
  }
}

/* ------------------------------------------------------------ 参数 */

async function applyParams() {
  if (!state.sessionId) return;
  const wDist = parseFloat($("p-wdist").value);
  const wApp = parseFloat($("p-wapp").value);
  const penalty = parseFloat($("p-penalty").value);
  if (![wDist, wApp, penalty].every((v) => Number.isFinite(v) && v >= 0)) {
    alert("参数必须是有限非负数。");
    return;
  }
  invalidateResult(); // 修改即撤下旧结果
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/params`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ w_dist: wDist, w_app: wApp, penalty }),
    });
    state.version = resp.version;
    state.params = { w_dist: wDist, w_app: wApp, penalty };
  } catch (err) {
    alert(`保存参数失败：${err.message}`);
  }
}

/* ------------------------------------------------------------ 求解 */

function invalidateResult() {
  if (state.result !== null) {
    state.result = null;
    const banner = $("result-banner");
    banner.className = "banner stale";
    banner.textContent = "输入已修改，旧结果已撤下。请重新计算。";
    $("result-summary").innerHTML = "";
    $("trajectory-list").innerHTML = "";
    draw();
  }
}

async function solve() {
  if (!state.sessionId) return;
  const status = $("solve-status");
  status.classList.add("muted");
  status.style.color = "";
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/solve`, { method: "POST" });
    state.jobId = resp.job_id;
    status.textContent = `计算中（输入版本 v${resp.version}）… 可继续修改判断与参数。`;
    if (state.pollTimer) clearTimeout(state.pollTimer);
    pollJob(resp.job_id);
  } catch (err) {
    showError("solve-status", err);
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (jobId !== state.jobId) return; // 已被更新的任务取代
    if (job.status === "running") {
      state.pollTimer = setTimeout(() => pollJob(jobId), 400);
      return;
    }
    const status = $("solve-status");
    if (job.status === "error") {
      status.textContent = job.error || "求解失败。";
      status.style.color = "#ff9d97";
      return;
    }
    if (job.version !== state.version) {
      // 迟到结果：输入在计算期间已变化，丢弃不展示
      status.textContent =
        `任务完成，但其基于旧输入（v${job.version}，当前 v${state.version}），结果已作废。请重新计算。`;
      status.style.color = "#ffd887";
      const banner = $("result-banner");
      banner.className = "banner stale";
      banner.textContent = "存在基于旧输入的迟到结果，已自动丢弃，未覆盖当前输入。";
      return;
    }
    status.textContent = `计算完成（输入版本 v${job.version}）。`;
    status.style.color = "";
    state.result = job.result;
    renderResult();
    draw();
  } catch (err) {
    showError("solve-status", err);
  }
}

function renderResult() {
  const banner = $("result-banner");
  const summary = $("result-summary");
  const list = $("trajectory-list");
  const r = state.result;
  if (!r) return;
  if (!r.feasible) {
    banner.className = "banner err";
    banner.textContent = `无解：${r.message || "约束冲突。"}`;
    summary.innerHTML = "<b>不提供动物计数。</b>";
    list.innerHTML = "";
    return;
  }
  banner.className = "banner ok";
  banner.textContent = r.message || "求解成功。";
  summary.innerHTML =
    `<div><span class="count">${r.animal_count}</span>只动物（按轨迹条数计）</div>` +
    `<div>轨迹条数：${r.n_trajectories}</div>` +
    `<div>总代价：${r.total_cost.toFixed(3)}` +
    `（关联代价 ${r.assoc_cost.toFixed(3)} + 轨迹罚分 ${r.penalty_cost.toFixed(3)}）</div>`;
  list.innerHTML = "";
  r.trajectories.forEach((t, i) => {
    const div = document.createElement("div");
    div.className = "traj";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = PALETTE[i % PALETTE.length];
    const boxes = document.createElement("span");
    boxes.className = "boxes";
    boxes.textContent = `轨迹 T${i + 1}：` + t.boxes.map((b) => `帧${b.frame} #${b.id}`).join(" → ");
    const len = document.createElement("span");
    len.className = "len";
    len.textContent = `（${t.length} 框）`;
    div.append(sw, boxes, len);
    list.appendChild(div);
  });
}

/* ------------------------------------------------------------ 汇总渲染 */

function renderAll() {
  renderFrameNav();
  renderSelection();
  renderJudgments();
  renderSegmentPanel();
  draw();
  const banner = $("result-banner");
  banner.className = "banner muted";
  banner.textContent = "尚无结果。";
  $("result-summary").innerHTML = "";
  $("trajectory-list").innerHTML = "";
}

/* ------------------------------------------------------------ 事件绑定 */

$("upload-btn").addEventListener("click", uploadSession);
$("example-btn").addEventListener("click", loadExample);
$("prev-frame").addEventListener("click", () => switchFrame(-1));
$("next-frame").addEventListener("click", () => switchFrame(1));
$("mark-same").addEventListener("click", () => setJudgment("same"));
$("mark-diff").addEventListener("click", () => setJudgment("different"));
$("mark-clear").addEventListener("click", () => setJudgment(null));
$("solve-btn").addEventListener("click", solve);
canvas.addEventListener("click", onCanvasClick);
canvas.addEventListener("contextmenu", onCanvasContextMenu);
$("seg-mode-btn").addEventListener("click", toggleSegmentMode);
$("seg-undo").addEventListener("click", undoLastPoint);
$("seg-clear").addEventListener("click", clearSegmentPoints);
for (const id of ["p-wdist", "p-wapp", "p-penalty"]) {
  $(id).addEventListener("change", applyParams);
}
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === "ArrowLeft") switchFrame(-1);
  if (ev.key === "ArrowRight") switchFrame(1);
});

// 初始刷新：无会话时点选分割按钮保持禁用
renderSegmentPanel();
