# -*- coding: utf-8 -*-
"""
Flask Web UI 服务
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request as flask_request

import webui_state as ws


WEBUI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO 检测 · 实时监控</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border-bottom: 1px solid #2d3748;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #f7fafc; }
  header .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #48bb78; margin-left: auto;
    box-shadow: 0 0 6px #48bb78;
    animation: pulse 2s infinite;
  }
  header .status-dot.offline { background: #fc8181; box-shadow: 0 0 6px #fc8181; animation: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .main-layout {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 340px;
    grid-template-rows: 1fr auto;
    height: calc(100vh - 57px);
  }

  .video-panel {
    background: #0a0d14;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #2d3748;
  }
  .video-header {
    padding: 10px 16px;
    background: #161b27;
    border-bottom: 1px solid #2d3748;
    font-size: 12px;
    color: #718096;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .video-header span { color: #a0aec0; font-weight: 500; }
  #fps-badge {
    margin-left: auto;
    background: #2d3748;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    color: #68d391;
  }
  .video-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    overflow: hidden;
  }
  #live-stream {
    max-width: 100%;
    max-height: 100%;
    border-radius: 8px;
    border: 1px solid #2d3748;
    object-fit: contain;
  }
  .no-signal {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: #4a5568;
  }
  .no-signal .icon { font-size: 48px; }

  .side-panel {
    display: flex;
    flex-direction: column;
    background: #13192a;
    overflow: hidden;
  }

  .detections-section {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid #2d3748;
  }
  .section-title {
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: #161b27;
    border-bottom: 1px solid #2d3748;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .section-title .count-badge {
    background: #2b6cb0;
    color: #bee3f8;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 11px;
    margin-left: auto;
  }
  #detections-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  .det-item {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .det-item:hover { border-color: #4a90d9; background: #1e2a45; }
  .det-item .det-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  .det-item .class-dot {
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0;
  }
  .det-item .class-name {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    flex: 1;
  }
  .det-item .conf-badge {
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 8px;
    background: #276749;
    color: #9ae6b4;
  }
  .det-item .conf-badge.low { background: #744210; color: #fbd38d; }
  .det-item .det-pos {
    font-size: 11px;
    color: #718096;
    font-family: 'Courier New', monospace;
  }
  .det-item .track-id {
    font-size: 10px;
    color: #4a5568;
    margin-top: 2px;
  }
  .det-item .det-caption {
    font-size: 11px;
    color: #a0aec0;
    margin-top: 3px;
    line-height: 1.4;
    font-style: italic;
  }
  .empty-hint {
    text-align: center;
    color: #4a5568;
    font-size: 13px;
    padding: 32px 16px;
  }

  .go2-section {
    background: #161b27;
    border-bottom: 1px solid #2d3748;
  }
  .go2-toggle {
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: #161b27;
    border: none;
    border-bottom: 1px solid #2d3748;
    width: 100%;
    text-align: left;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .go2-toggle:hover { background: #1a2035; }
  .go2-toggle .arrow { transition: transform 0.2s; font-size: 10px; margin-left: auto; color: #4a5568; }
  .go2-toggle .arrow.open { transform: rotate(90deg); }
  .go2-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #fc8181;
    display: inline-block;
  }
  .go2-status-dot.ready { background: #48bb78; }
  .go2-panel {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
  }
  .go2-panel.open { max-height: 500px; }
  .go2-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    padding: 10px 12px;
  }
  .go2-btn {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 8px 4px;
    font-size: 11px;
    color: #a0aec0;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
    line-height: 1.3;
  }
  .go2-btn:hover { border-color: #ed8936; background: #1e2a45; color: #fbd38d; }
  .go2-btn:active { transform: scale(0.95); }
  .go2-btn.executing { border-color: #ed8936; background: #2d3748; color: #fbd38d; pointer-events: none; }
  .go2-btn .btn-icon { font-size: 16px; display: block; margin-bottom: 2px; }
  .go2-btn .btn-label { display: block; }

  .query-section {
    background: #161b27;
    border-top: 1px solid #2d3748;
  }
  .query-log {
    height: 180px;
    overflow-y: auto;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.6;
  }
  .log-entry { margin-bottom: 6px; }
  .log-entry .log-time { color: #4a5568; margin-right: 6px; }
  .log-entry.user .log-text { color: #90cdf4; }
  .log-entry.system .log-text { color: #68d391; }
  .log-entry.error .log-text { color: #fc8181; }
  .log-entry.info .log-text { color: #a0aec0; }

  .query-input-row {
    display: flex;
    gap: 8px;
    padding: 10px 12px;
    border-top: 1px solid #2d3748;
  }
  #query-input {
    flex: 1;
    background: #0f1117;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 8px 12px;
    color: #e2e8f0;
    font-size: 13px;
    outline: none;
  }
  #query-input:focus { border-color: #4a90d9; }
  #query-btn {
    background: linear-gradient(135deg, #4a90d9, #357abd);
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    color: white;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }

  .quick-btns {
    display: flex;
    gap: 6px;
    padding: 0 12px 8px;
    flex-wrap: wrap;
  }
  .quick-btn {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    color: #a0aec0;
    cursor: pointer;
  }

  #nav-success-banner {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: linear-gradient(135deg, #1a4731, #22543d);
    border-bottom: 2px solid #48bb78;
    box-shadow: 0 4px 24px rgba(72, 187, 120, 0.35);
  }
  #nav-success-banner.show {
    display: flex;
    animation: slideDown 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  @keyframes slideDown {
    from { transform: translateY(-100%); opacity: 0; }
    to   { transform: translateY(0); opacity: 1; }
  }
  @keyframes slideUp {
    from { transform: translateY(0); opacity: 1; }
    to   { transform: translateY(-100%); opacity: 0; }
  }
  #nav-success-banner.hide {
    animation: slideUp 0.35s ease-in forwards;
  }
  .banner-inner {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 14px 24px;
  }
  .banner-icon { font-size: 28px; }
  .banner-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .banner-title {
    font-size: 18px;
    font-weight: 700;
    color: #68d391;
  }
  .banner-sub {
    font-size: 13px;
    color: #9ae6b4;
    opacity: 0.85;
  }
  .banner-close {
    margin-left: auto;
    background: none;
    border: 1px solid #276749;
    border-radius: 6px;
    color: #68d391;
    padding: 4px 12px;
    font-size: 12px;
    cursor: pointer;
  }

  #live-stream.nav-success {
    border-color: #48bb78;
    box-shadow: 0 0 0 3px rgba(72, 187, 120, 0.4);
    transition: border-color 0.3s, box-shadow 0.3s;
  }
</style>
</head>
<body>

<div id="nav-success-banner">
  <div class="banner-inner">
    <div class="banner-icon">&#x2705;</div>
    <div class="banner-text">
      <div class="banner-title">导航成功！</div>
      <div class="banner-sub" id="banner-sub-text">已到达目标位置</div>
    </div>
    <button class="banner-close" onclick="dismissBanner()">关闭</button>
  </div>
</div>

<header>
  <div class="logo">&#x1F916;</div>
  <h1>RealVLN 实时检测监控</h1>
  <div id="status-dot" class="status-dot offline"></div>
</header>

<div class="main-layout">
  <div class="video-panel">
    <div class="video-header">
      &#x1F4F9; <span>实时画面</span>
      <span id="fps-badge">-- FPS</span>
    </div>
    <div class="video-container">
      <img id="live-stream" alt="视频流" style="display:none">
      <div class="no-signal" id="no-signal">
        <div class="icon">&#x1F4F7;</div>
        <div>等待视频流...</div>
      </div>
    </div>
  </div>

  <div class="side-panel">
    <div class="detections-section">
      <div class="section-title">
        &#x1F50D; 检测结果
        <span class="count-badge" id="det-count">0</span>
      </div>
      <div id="detections-list">
        <div class="empty-hint">暂无检测结果</div>
      </div>
    </div>

    <div class="go2-section" id="go2-section">
      <button class="go2-toggle" onclick="toggleGo2Panel()">
        &#x1F43E; Go2 运动控制
        <span class="go2-status-dot" id="go2-status-dot"></span>
        <span class="arrow" id="go2-arrow">&#x25B6;</span>
      </button>
      <div class="go2-panel" id="go2-panel">
        <div class="go2-grid" id="go2-grid"></div>
      </div>
    </div>

    <div class="query-section">
      <div class="section-title">&#x1F4AC; 语义导航查询</div>
      <div class="query-log" id="query-log">
        <div class="log-entry info">
          <span class="log-text">系统已就绪，输入目标类别/自然语言发布导航目标，或输入动作指令（如 turn_left）控制机器狗</span>
        </div>
      </div>
      <div class="quick-btns">
        <button class="quick-btn" onclick="quickQuery('list')">list 列出物体</button>
        <button class="quick-btn" onclick="quickQuery('chair')">chair</button>
        <button class="quick-btn" onclick="quickQuery('box')">box</button>
        <button class="quick-btn" onclick="quickQuery('person')">person</button>
      </div>
      <div class="query-input-row">
        <input id="query-input" type="text" placeholder="输入目标类别、自然语言，或动作指令（如 turn_left、stand_up）" />
        <button id="query-btn" onclick="submitQuery()">发送</button>
      </div>
    </div>
  </div>
</div>

<script>
const COLORS = [
  '#48bb78','#ed8936','#4299e1','#ed64a6',
  '#a0aec0','#68d391','#fc8181','#f6e05e'
];

let lastFrameTime = Date.now();
let frameCount = 0;
let fps = 0;

const img = document.getElementById('live-stream');
const noSignal = document.getElementById('no-signal');
const statusDot = document.getElementById('status-dot');
let polling = false;
let hasFrame = false;

function pollSnapshot() {
  if (polling) return;
  polling = true;
  fetch('/snapshot')
    .then(r => { if (!r.ok) throw new Error(r.status); return r.blob(); })
    .then(blob => {
      const url = URL.createObjectURL(blob);
      img.onload = function() {
        URL.revokeObjectURL(url);
        if (!hasFrame) {
          hasFrame = true;
          noSignal.style.display = 'none';
          img.style.display = 'block';
        }
        statusDot.className = 'status-dot';
        frameCount++;
        const now = Date.now();
        if (now - lastFrameTime >= 1000) {
          fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
          document.getElementById('fps-badge').textContent = fps + ' FPS';
          frameCount = 0;
          lastFrameTime = now;
        }
        polling = false;
      };
      img.onerror = function() {
        URL.revokeObjectURL(url);
        polling = false;
      };
      img.src = url;
    })
    .catch(() => {
      statusDot.className = 'status-dot offline';
      polling = false;
    });
}
setInterval(pollSnapshot, 100);

let latestCaptions = {};

async function fetchDetections() {
  try {
    const resp = await fetch('/api/detections');
    if (!resp.ok) return;
    const data = await resp.json();
    renderDetections(data.detections || []);
  } catch(e) {}
}

async function fetchCaptions() {
  try {
    const resp = await fetch('/api/captions');
    if (!resp.ok) return;
    const data = await resp.json();
    latestCaptions = data.captions || {};
  } catch(e) {}
}

function renderDetections(dets) {
  const list = document.getElementById('detections-list');
  const badge = document.getElementById('det-count');
  badge.textContent = dets.length;

  if (dets.length === 0) {
    list.innerHTML = '<div class="empty-hint">暂无检测结果</div>';
    return;
  }

  list.innerHTML = dets.map((d, i) => {
    const color = COLORS[i % COLORS.length];
    const conf = d.confidence || 0;
    const confClass = conf < 0.5 ? 'low' : '';
    const pos = d.world_fused
      ? `(${d.world_fused[0].toFixed(2)}, ${d.world_fused[1].toFixed(2)}, ${d.world_fused[2].toFixed(2)})`
      : '位置未知';
    const cap = d.caption || latestCaptions[String(d.track_id)] || '';
    const capHtml = cap ? `<div class="det-caption">${cap}</div>` : '';
    return `<div class="det-item" onclick="quickQuery('${d.class_name}')">
      <div class="det-header">
        <div class="class-dot" style="background:${color}"></div>
        <span class="class-name">${d.class_name}</span>
        <span class="conf-badge ${confClass}">${(conf*100).toFixed(0)}%</span>
      </div>
      <div class="det-pos">世界坐标: ${pos}</div>
      <div class="track-id">Track #${d.track_id}</div>
      ${capHtml}
    </div>`;
  }).join('');
}

setInterval(fetchDetections, 500);
setInterval(fetchCaptions, 3000);

function addLog(text, type='info') {
  const log = document.getElementById('query-log');
  const now = new Date().toLocaleTimeString('zh-CN', {hour12:false});
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `<span class="log-time">[${now}]</span><span class="log-text">${text}</span>`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
  while (log.children.length > 100) log.removeChild(log.firstChild);
}

async function submitQuery() {
  const input = document.getElementById('query-input');
  const btn = document.getElementById('query-btn');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  btn.disabled = true;
  addLog(text, 'user');

  try {
    const resp = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: text})
    });
    const data = await resp.json();
    if (data.error) {
      addLog('错误: ' + data.error, 'error');
    } else {
      for (const line of (data.messages || [])) {
        addLog(line, 'system');
      }
    }
  } catch(e) {
    addLog('请求失败: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

function quickQuery(text) {
  document.getElementById('query-input').value = text;
  submitQuery();
}

document.getElementById('query-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') submitQuery();
});

let bannerTimer = null;
let navPolling = false;

function showNavSuccessBanner(target) {
  const banner = document.getElementById('nav-success-banner');
  const sub = document.getElementById('banner-sub-text');
  sub.textContent = target ? `已到达目标：${target}` : '已到达目标位置';

  banner.classList.remove('show', 'hide');
  void banner.offsetWidth;
  banner.classList.add('show');

  const liveImg = document.getElementById('live-stream');
  liveImg.classList.add('nav-success');

  addLog(`✅ 导航成功！已到达 "${target || '目标'}"`, 'system');

  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(dismissBanner, 5000);
}

function dismissBanner() {
  const banner = document.getElementById('nav-success-banner');
  banner.classList.add('hide');
  setTimeout(() => {
    banner.classList.remove('show', 'hide');
    document.getElementById('live-stream').classList.remove('nav-success');
  }, 380);
  if (bannerTimer) { clearTimeout(bannerTimer); bannerTimer = null; }
}

async function pollNavStatus() {
  if (navPolling) return;
  navPolling = true;
  try {
    const resp = await fetch('/api/nav_status');
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.succeeded) {
      showNavSuccessBanner(data.target || '');
      fetch('/api/nav_status/ack', {method: 'POST'}).catch(() => {});
    }
  } catch(e) {}
  finally { navPolling = false; }
}
setInterval(pollNavStatus, 800);

const GO2_ICONS = {
  stand_up: '🧍', stand_down: '🛌', move_forward: '⬆',
  move_backward: '⬇', turn_left: '⬅', turn_right: '➡',
  stop: '🛑', hello: '👋', sit: '🧘', stretch: '🤸',
  recovery: '🔄', balance: '⚖', damp: '🛡',
  move_lateral: '↔', handstand: '🤸', left_flip: '🔀',
  back_flip: '🔄', free_walk: '🚶', free_bound: '🏃',
  free_avoid: '🚧', walk_upright: '🧍', cross_step: '💃',
  free_jump: '🏋',
};

let go2PanelOpen = false;
let go2Actions = [];
let go2Enabled = false;

function toggleGo2Panel() {
  go2PanelOpen = !go2PanelOpen;
  document.getElementById('go2-panel').classList.toggle('open', go2PanelOpen);
  document.getElementById('go2-arrow').classList.toggle('open', go2PanelOpen);
  if (go2PanelOpen && go2Actions.length === 0) fetchGo2Actions();
}

async function fetchGo2Actions() {
  try {
    const resp = await fetch('/api/go2_actions');
    if (!resp.ok) return;
    const data = await resp.json();
    go2Actions = data.actions || [];
    go2Enabled = data.enabled || false;
    renderGo2Grid();
    document.getElementById('go2-status-dot').classList.toggle('ready', go2Enabled);
  } catch(e) {}
}

function renderGo2Grid() {
  const grid = document.getElementById('go2-grid');
  if (go2Actions.length === 0) {
    grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:#4a5568;padding:12px;font-size:12px;">Go2 控制器未连接</div>';
    return;
  }
  grid.innerHTML = go2Actions.map(a => {
    const icon = GO2_ICONS[a.name] || '🤖';
    return `<button class="go2-btn" id="go2-btn-${a.name}" onclick="execGo2('${a.name}')" title="${a.desc}"><span class="btn-icon">${icon}</span><span class="btn-label">${a.desc}</span></button>`;
  }).join('');
}

async function execGo2(actionName) {
  const btn = document.getElementById('go2-btn-' + actionName);
  if (btn) btn.classList.add('executing');
  addLog(`[Go2] ${actionName}`, 'user');
  try {
    const resp = await fetch('/api/go2_action', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: actionName})
    });
    const data = await resp.json();
    if (data.error) {
      addLog('[Go2] 错误: ' + data.error, 'error');
    } else {
      for (const line of (data.messages || [])) addLog('[Go2] ' + line, 'system');
    }
  } catch(e) {
    addLog('[Go2] 请求失败: ' + e.message, 'error');
  } finally {
    if (btn) btn.classList.remove('executing');
  }
}

fetchGo2Actions();
setInterval(() => {
  if (!go2PanelOpen) fetchGo2Actions();
}, 10000);
</script>
</body>
</html>
"""


app = Flask(__name__)
app.logger.disabled = True
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def webui_index():
    return WEBUI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/stream")
def webui_stream():
    def generate():
        while True:
            frame_jpg = ws.get_latest_frame_jpg()
            if frame_jpg is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Waiting for camera...",
                    (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (80, 80, 80),
                    2,
                )
                frame_jpg = ws.encode_frame_to_jpg(placeholder)
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_jpg + b"\r\n"
            )
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/snapshot")
def webui_snapshot():
    frame_jpg = ws.get_latest_frame_jpg()
    if frame_jpg is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for camera...",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 80, 80),
            2,
        )
        frame_jpg = ws.encode_frame_to_jpg(placeholder)

    return Response(
        frame_jpg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@app.route("/api/detections")
def webui_detections():
    dets = ws.get_latest_detections()
    return jsonify({"detections": dets, "count": len(dets)})


@app.route("/api/nav_status")
def webui_nav_status():
    return jsonify(ws.get_nav_status())


@app.route("/api/nav_status/ack", methods=["POST"])
def webui_nav_status_ack():
    ws.ack_nav_status()
    return jsonify({"ok": True})


@app.route("/api/captions")
def webui_captions():
    return jsonify(ws.fetch_captions())


@app.route("/api/query", methods=["POST"])
def webui_query():
    data = flask_request.get_json(force=True, silent=True) or {}
    user_input = str(data.get("query", "")).strip()

    if ws.is_go2_action(user_input):
        messages, error = ws.handle_go2_action(user_input)
        if error is not None:
            return jsonify({"error": error}), 500 if "失败" in error else 400
        return jsonify({"messages": messages})

    messages, error = ws.handle_query_request(user_input)
    if error is not None:
        return jsonify({"error": error}), 500 if "失败" in error else 400
    return jsonify({"messages": messages})


@app.route("/api/go2_action", methods=["POST"])
def webui_go2_action():
    data = flask_request.get_json(force=True, silent=True) or {}
    action_name = str(data.get("action", "")).strip()
    angle = data.get("angle")
    if angle is not None:
        try:
            angle = float(angle)
        except (ValueError, TypeError):
            angle = None
    if not action_name:
        return jsonify({"error": "action 不能为空"}), 400
    messages, error = ws.handle_go2_action(action_name, angle=angle)
    if error is not None:
        return jsonify({"error": error}), 500 if "失败" in error else 400
    return jsonify({"messages": messages})


@app.route("/api/go2_actions", methods=["GET"])
def webui_go2_actions_list():
    from go2_control import get_action_list
    controller = ws.get_go2_controller()
    return jsonify({
        "actions": get_action_list(),
        "enabled": controller is not None and controller.is_ready,
    })


def start_webui(port: int) -> threading.Thread:
    def webui_thread():
        print(f"[WebUI] 启动 Web UI，访问 http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)

    t = threading.Thread(target=webui_thread, daemon=True)
    t.start()
    return t
