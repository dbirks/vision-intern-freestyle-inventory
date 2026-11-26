/* Freestyle review app — vanilla JS, no build step. */
'use strict';

const params = new URLSearchParams(location.search);
const GRID_MODE = params.get('mode') === 'grid';

const state = {
  images: [],
  classes: [],
  idx: 0,
  data: null,          // current image payload (ann + attrs)
  sel: null,           // selected annotation id
  zoom: 1,
  pan: { x: 0, y: 0 },
  space: false,
  drawing: false,
  drag: null,
  saveTimer: null,
  gridView: GRID_MODE,
};

const $ = (s) => document.querySelector(s);
const byId = (id) => document.getElementById(id);
const img = $('#img');
const overlay = $('#overlay');
const wrap = $('#canvas-wrap');

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// --- data -------------------------------------------------------------------

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

async function refreshImages() {
  const data = await fetchJSON('/api/images');
  state.images = data.images;
  renderImageList();
}

async function loadImage(idx) {
  state.idx = idx;
  state.data = await fetchJSON(`/api/ann/${idx}`);
  state.sel = null;
  img.src = `/img/${idx}`;
  wrap.classList.toggle('grid-mode', state.gridView);
  img.addEventListener('load', drawGridCells, { once: true });
  // reset view to fit
  state.space = false;
  state.drawing = false;
  await fitView();
  renderAll();
}

async function activeImg() {
  if (!state.data) return null;
  return img;
}

async function fitView() {
  const d = state.data;
  const r = wrap.getBoundingClientRect();
  if (!d || !d.width || !d.height || r.width < 10) return;
  state.zoom = Math.min(r.width / d.width, r.height / d.height) * 0.96;
  state.pan = {
    x: (r.width - d.width * state.zoom) / 2,
    y: (r.height - d.height * state.zoom) / 2,
  };
}

function applyTransform() {
  const t = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
  img.style.transform = t;
  img.style.left = '0';
  img.style.top = '0';
  overlay.style.transform = t;
  overlay.style.transformOrigin = '0 0';
  renderBoxes();
}

// --- render -----------------------------------------------------------------

function renderImageList() {
  const list = byId('image-list');
  list.innerHTML = state.images.map((im) => {
    const cls = im.status === 'verified' ? 'verified' : im.status === 'in_progress' ? 'in-progress' : 'todo';
    return `<button class="img-item ${im.idx === state.idx ? 'active' : ''} ${cls}" data-idx="${im.idx}">
      <span class="status-dot ${cls}"></span>
      <span class="img-name">${esc(im.image_id)}</span>
      <span class="img-meta">${im.n_boxes} box${im.n_boxes === 1 ? '' : 'es'}</span>
    </button>`;
  }).join('');
  list.querySelectorAll('.img-item').forEach((b) =>
    b.addEventListener('click', () => loadImage(parseInt(b.dataset.idx, 10)))
  );
  const verified = state.images.filter((i) => i.verified).length;
  const total = state.images.length;
  byId('left-foot').textContent = `${verified}/${total} verified`;
  byId('left-meta').textContent = `${total} images`;
}

function catOf(name) {
  return state.classes.find((c) => c.name === name) || { name, color: '#868e96', display: name, hotkey: '', generic: true };
}

function renderBoxes() {
  const d = state.data;
  if (!d) { overlay.innerHTML = ''; return; }
  const svgNS = 'http://www.w3.org/2000/svg';
  overlay.setAttribute('viewBox', `0 0 ${d.width} ${d.height}`);
  overlay.setAttribute('width', d.width);
  overlay.setAttribute('height', d.height);
  overlay.innerHTML = '';
  for (const a of d.annotations) {
    const [x, y, w, h] = a.bbox;
    const cat = catOf(a.category);
    const sel = a.id === state.sel;
    const g = document.createElementNS(svgNS, 'g');
    g.setAttribute('class', 'box' + (sel ? ' selected' : ''));
    g.dataset.id = a.id;
    const pad = Math.max(1, 2 / state.zoom);
    const rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('x', x); rect.setAttribute('y', y);
    rect.setAttribute('width', w); rect.setAttribute('height', h);
    rect.setAttribute('fill', cat.color + '33');
    rect.setAttribute('stroke', cat.color);
    rect.setAttribute('stroke-width', sel ? 3.5 / state.zoom : 2 / state.zoom);
    g.appendChild(rect);
    const label = document.createElementNS(svgNS, 'text');
    label.setAttribute('x', x);
    label.setAttribute('y', y - 4 / state.zoom);
    label.setAttribute('font-size', 15 / state.zoom);
    label.setAttribute('class', 'box-label');
    label.textContent = `${a.category.split('-').slice(0, 2).join('-')}`;
    label.setAttribute('fill', cat.color);
    label.setAttribute('paint-order', 'stroke');
    label.setAttribute('stroke', 'rgba(0,0,0,0.75)');
    label.setAttribute('stroke-width', 3 / state.zoom);
    g.appendChild(label);
    if (sel) {
      const handles = [[x, y, 'nw'], [x + w / 2, y, 'n'], [x + w, y, 'ne'],
                       [x + w, y + h / 2, 'e'], [x + w, y + h, 'se'], [x + w / 2, y + h, 's'],
                       [x, y + h, 'sw'], [x, y + h / 2, 'w']];
      for (const [hx, hy, cursor] of handles) {
        const r = document.createElementNS(svgNS, 'rect');
        const s = Math.max(6, 10 / state.zoom);
        r.setAttribute('class', 'handle');
        r.setAttribute('x', hx - s / 2); r.setAttribute('y', hy - s / 2);
        r.setAttribute('width', s); r.setAttribute('height', s);
        r.setAttribute('rx', 2 / state.zoom);
        r.setAttribute('data-cursor', cursor);
        r.setAttribute('data-id', a.id);
        r.setAttribute('data-hx', hx); r.setAttribute('data-hy', hy);
        g.appendChild(r);
      }
    }
    overlay.appendChild(g);
  }
}

function renderAll() {
  renderToolbar();
  renderCounts();
  renderInspector();
  if (state.gridView) renderGrid();
}

function renderToolbar() {
  const d = state.data;
  byId('cur-title').textContent = d ? d.image_id : '—';
  byId('btn-verify').textContent = d && d.verified ? 'verified ✓' : 'mark verified';
  byId('btn-verify').classList.toggle('done', !!(d && d.verified));
  byId('sel-verified').checked = !!(d && d.verified);
  byId('sel-note').value = d ? d.reviewer_note || '' : '';
  byId('sel-scene').value = d ? d.scene || '' : '';
  byId('btn-new').disabled = false;
}

function renderCounts() {
  const d = state.data;
  const counts = new Map();
  if (d) for (const a of d.annotations) counts.set(a.category, (counts.get(a.category) || 0) + 1);
  const parts = [];
  for (const c of state.classes) {
    const n = counts.get(c.name) || 0;
    parts.push(`<span class="count${n ? ' have' : ' zero'}" title="${esc(c.name)}"><i style="background:${c.color}"></i>${esc(c.name)} ×${n}</span>`);
  }
  byId('counts').innerHTML = parts.join('');
}

function renderInspector() {
  const d = state.data;
  const a = d && d.annotations.find((x) => x.id === state.sel);
  const noSel = byId('no-selection'), insp = byId('inspector');
  if (!a) {
    noSel.classList.remove('hidden');
    insp.classList.add('hidden');
    return;
  }
  noSel.classList.add('hidden');
  insp.classList.remove('hidden');
  byId('box-size').textContent = `${a.bbox[2].toFixed(1)} × ${a.bbox[3].toFixed(1)} @ ${a.bbox[0].toFixed(1)},${a.bbox[1].toFixed(1)}`;
  byId('sel-class').value = a.category;
  byId('sel-class').innerHTML = state.classes.map((c) =>
    `<option value="${esc(c.name)}">${esc(c.display)}${c.hotkey ? ` (${c.hotkey})` : ''}</option>`
  ).join('');
  byId('sel-class').value = a.category;
  byId('flag-trunc').checked = !!a.attributes.truncated;
  byId('flag-occ').checked = !!a.attributes.occluded;
  byId('flag-iv').checked = a.attributes.identity_visible !== false;
  const grid = byId('class-grid');
  grid.innerHTML = state.classes.map((c) => `
    <button class="cls-btn${c.name === a.category ? ' active' : ''}" data-cls="${esc(c.name)}" title="hotkey ${esc(c.hotkey)}">
      <i style="background:${c.color}"></i><span class="cls-name">${esc(c.display)}</span><kbd>${esc(c.hotkey)}</kbd>
    </button>`).join('');
  grid.querySelectorAll('.cls-btn').forEach((b) =>
    b.addEventListener('click', () => setCategory(b.dataset.cls))
  );
  drawCrop(a);
}

function drawCrop(a) {
  const d = state.data;
  const canvas = byId('crop');
  const ctx = canvas.getContext('2d');
  const [x, y, w, h] = a.bbox;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#141414';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // letterbox into the crop box; 3x zoom capped so the crop fits the canvas
  const scale = Math.min(canvas.width / Math.max(1, w), canvas.height / Math.max(1, h), 3);
  const dw = w * scale, dh = h * scale;
  const dx = (canvas.width - dw) / 2, dy = (canvas.height - dh) / 2;
  try {
    ctx.drawImage(img, x, y, w, h, dx, dy, dw, dh);
  } catch (e) { /* image not loaded yet */ }
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.strokeRect(dx + 0.5, dy + 0.5, dw - 1, dh - 1);
}

function renderGrid() {
  const grid = byId('grid-view');
  const d = state.data;
  if (!state.gridView || !d) { grid.classList.add('hidden'); return; }
  grid.classList.remove('hidden');
  byId('audit-link').textContent = 'back to canvas';
  byId('audit-link').href = location.pathname + '?mode=edit';
  const byCls = new Map();
  for (const a of d.annotations) {
    if (!byCls.has(a.category)) byCls.set(a.category, []);
    byCls.get(a.category).push(a);
  }
  const sorted = [...byCls.entries()].sort((x, y) => x[0].localeCompare(y[0]));
  let html = `<div class="grid-head">audit — sorted by class (${d.annotations.length} boxes)</div>`;
  for (const [cls, anns] of sorted) {
    const cat = catOf(cls);
    html += `<div class="grid-section"><div class="grid-section-title"><i style="background:${cat.color}"></i>${esc(cat.display)} (${cls}) ×${anns.length}</div><div class="grid-cells">`;
    for (const a of anns) {
      const [x, y, w, h] = a.bbox;
      const boxJson = JSON.stringify([x | 0, y | 0, Math.max(1, w | 0), Math.max(1, h | 0)]);
      html += `<button class="grid-cell${a.id === state.sel ? ' sel' : ''}" data-id="${a.id}" title="${esc(cls)} ${w.toFixed(0)}×${h.toFixed(0)}">
        <canvas width="168" height="112" data-box="${esc(boxJson)}"></canvas><span>${esc(cls)}</span></button>`;
    }
    html += `</div></div>`;
  }
  grid.innerHTML = html;
  grid.querySelectorAll('.grid-cell').forEach((cell) =>
    cell.addEventListener('click', () => {
      state.sel = parseInt(cell.dataset.id, 10);
      renderAll();
      renderBoxes();
    })
  );
  drawGridCells();
}

function drawGridCells() {
  const d = state.data;
  if (!d || !state.gridView) return;
  const loaded = img.complete && img.naturalWidth > 0 && img.naturalWidth === d.width;
  document.querySelectorAll('.grid-cell canvas').forEach((canvas) => {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#101010';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!loaded) return;
    let box;
    try { box = JSON.parse(canvas.dataset.box); } catch (e) { return; }
    const [x, y, w, h] = box;
    const scale = Math.min(canvas.width / w, canvas.height / h);
    const dx = (canvas.width - w * scale) / 2, dy = (canvas.height - h * scale) / 2;
    try { ctx.drawImage(img, x, y, w, h, dx, dy, w * scale, h * scale); } catch (e) {}
  });
}

// --- server round trip ------------------------------------------------------

function currentPayload() {
  const d = state.data;
  return {
    annotations: d.annotations.map((a) => ({
      id: a.id ?? null,
      category: a.category,
      bbox: a.bbox.map((v) => Math.round(v * 100) / 100),
      quad: a.quad ? a.quad.map((pt) => pt.map((v) => Math.round(v * 100) / 100)) : null,
      attributes: { ...a.attributes },
    })),
    verified: d.verified,
    scene: d.scene,
    reviewer_note: d.reviewer_note,
  };
}

async function save() {
  if (!state.data) return;
  try {
    const res = await fetch(`/api/ann/${state.idx}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentPayload()),
    });
    if (!res.ok) {
      console.error('save failed', await res.text());
      byId('left-foot').textContent = 'save failed!';
      return;
    }
    const returned = await res.json();
    // reconcile ids (server may have renumbered)
    state.data = { ...returned };
    if (state.sel != null && !returned.annotations.some((a) => a.id === state.sel)) state.sel = null;
    await refreshImages();
    renderAll();
  } catch (e) {
    console.error('save error', e);
    byId('left-foot').textContent = 'save error';
  }
}

function scheduleSave() {
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(save, 250);
}

// --- interactions -----------------------------------------------------------

function selectBox(id) {
  state.sel = id;
  renderBoxes();
  renderInspector();
  if (state.gridView) renderGrid();
}

function setCategory(name) {
  const a = state.data.annotations.find((x) => x.id === state.sel);
  if (!a || a.category === name) return;
  a.category = name;
  a.attributes.source = 'human';
  renderAll();
  scheduleSave();
}

function setFlag(key, value) {
  const a = state.data.annotations.find((x) => x.id === state.sel);
  if (!a) return;
  if (!!a.attributes[key] === value) return;
  a.attributes[key] = value;
  a.attributes.source = 'human';
  renderAll();
  scheduleSave();
}

function setImageAttr(key, value) {
  state.data[key] = value === '' ? null : value;
  scheduleSave();
}

async function markVerified(value) {
  const d = state.data;
  if (!d) return;
  d.verified = value;
  const nextImg = value
    ? (state.images.find((im) => !im.verified && im.idx !== d.idx) || null)
    : null;
  renderToolbar();
  scheduleSave();
  if (value && nextImg) setTimeout(() => loadImage(nextImg.idx), 350);
}

// toolbar buttons
byId('btn-verify').addEventListener('click', () => markVerified(!state.data.verified));
byId('btn-new').addEventListener('click', startDrawing);
byId('sel-class').addEventListener('change', (e) => setCategory(e.target.value));
byId('flag-trunc').addEventListener('change', (e) => setFlag('truncated', e.target.checked));
byId('flag-occ').addEventListener('change', (e) => setFlag('occluded', e.target.checked));
byId('flag-iv').addEventListener('change', (e) => setFlag('identity_visible', e.target.checked));
byId('sel-scene').addEventListener('change', (e) => setImageAttr('scene', e.target.value));
byId('sel-note').addEventListener('change', (e) => setImageAttr('reviewer_note', e.target.value));
byId('sel-verified').addEventListener('change', (e) => markVerified(e.target.checked));

// --- canvas gestures --------------------------------------------------------

function toImageCoords(e) {
  const r = wrap.getBoundingClientRect();
  return {
    x: (e.clientX - r.left - state.pan.x) / state.zoom,
    y: (e.clientY - r.top - state.pan.y) / state.zoom,
  };
}

function startDrawing() {
  state.drawing = true;
  wrap.classList.add('drawing');
  byId('draw-hint').classList.remove('hidden');
}

function endDrawing() {
  state.drawing = false;
  wrap.classList.remove('drawing');
  byId('draw-hint').classList.add('hidden');
  state.drag = null;
  overlay.querySelectorAll('.draft-rect').forEach((n) => n.remove());
  renderBoxes();
}

wrap.addEventListener('pointerdown', (e) => {
  if (e.button === 2 || e.button === 1 || state.space || e.altKey) {
    // pan
    state.drag = { kind: 'pan', sx: e.clientX, sy: e.clientY, pan: { ...state.pan } };
    wrap.setPointerCapture(e.pointerId);
    wrap.classList.add('panning');
    return;
  }
  if (e.button !== 0) return;
  if (wrap.classList.contains('drawing')) {
    const p = toImageCoords(e);
    state.drag = { kind: 'draw', x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    wrap.setPointerCapture(e.pointerId);
    return;
  }
  const target = e.target;
  const handle = target.closest && target.closest('.handle');
  if (handle) {
    const cursor = handle.getAttribute('data-cursor');
    state.drag = { kind: 'resize', id: parseInt(handle.dataset.id, 10), cursor, sx: e.clientX, sy: e.clientY };
    wrap.setPointerCapture(e.pointerId);
    e.preventDefault();
    return;
  }
  const g = target.closest && target.closest('.box');
  if (g) {
    selectBox(parseInt(g.dataset.id, 10));
    // begin move drag
    const p = toImageCoords(e);
    const a = state.data.annotations.find((x) => x.id === state.sel);
    state.drag = { kind: 'move', id: a.id, ox: p.x, oy: p.y, orig: [...a.bbox] };
    wrap.setPointerCapture(e.pointerId);
    return;
  }
  selectBox(null);
});

wrap.addEventListener('pointermove', (e) => {
  if (!state.drag) return;
  const d = state.data;
  if (!d) return;
  if (state.drag.kind === 'pan') {
    state.pan = {
      x: state.drag.pan.x + (e.clientX - state.drag.sx),
      y: state.drag.pan.y + (e.clientY - state.drag.sy),
    };
    applyTransform();
    return;
  }
  if (state.drag.kind === 'draw') {
    const p = toImageCoords(e);
    state.drag.x1 = p.x; state.drag.y1 = p.y;
    drawDraftRect();
    return;
  }
  const p = toImageCoords(e);
  const a = d.annotations.find((x) => x.id === state.drag.id);
  if (!a) return;
  if (state.drag.kind === 'move') {
    const dx = p.x - state.drag.ox, dy = p.y - state.drag.oy;
    a.bbox = [state.drag.orig[0] + dx, state.drag.orig[1] + dy, state.drag.orig[2], state.drag.orig[3]];
    renderBoxes();
    renderInspector();
    return;
  }
  if (state.drag.kind === 'resize') {
    const [ox, oy, ow, oh] = a.bbox;
    const cursor = state.drag.cursor;
    let x = ox, y = oy, w = ow, h = oh;
    if (cursor.includes('e')) w = Math.max(2, p.x - ox);
    if (cursor.includes('w')) { x = Math.min(p.x, ox + ow - 2); w = ox + ow - x; }
    if (cursor.includes('s')) h = Math.max(2, p.y - oy);
    if (cursor.includes('n')) { y = Math.min(p.y, oy + oh - 2); h = oy + oh - y; }
    a.bbox = [x, y, w, h];
    renderBoxes();
    renderInspector();
  }
});

wrap.addEventListener('pointerup', (e) => {
  if (!state.drag) return;
  const kind = state.drag.kind;
  if (kind === 'draw') {
    const { x0, y0, x1, y1 } = state.drag;
    const x = Math.min(x0, x1), y = Math.min(y0, y1);
    const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
    endDrawing();
    if (w > 4 && h > 4) {
      const maxId = Math.max(0, ...state.data.annotations.map((a) => a.id || 0));
      state.data.annotations.push({
        id: maxId + 1,
        category: state.classes[0] ? state.classes[0].name : 'unknown',
        bbox: [Math.round(x), Math.round(y), Math.round(w), Math.round(h)],
        quad: null,
        attributes: { truncated: false, occluded: false, identity_visible: true, source: 'human' },
      });
      state.sel = state.data.annotations[state.data.annotations.length - 1].id;
      // keep server-assigned ids: re-sync from the returned payload in save()
      renderAll();
      renderBoxes();
      scheduleSave();
    } else {
      renderAll();
    }
    return;
  }
  const id = state.drag.id;
  state.drag = null;
  if (kind === 'move' || kind === 'resize') {
    renderBoxes();
    renderInspector();
    scheduleSave();
  }
  wrap.classList.remove('panning');
});

wrap.addEventListener('pointercancel', () => { endDrawing(); state.drag = null; wrap.classList.remove('panning'); });

function drawDraftRect() {
  const d = state.data;
  const overlay_svg = overlay;
  overlay_svg.querySelectorAll('.draft-rect').forEach((n) => n.remove());
  const svgNS = 'http://www.w3.org/2000/svg';
  const x = Math.min(state.drag.x0, state.drag.x1);
  const y = Math.min(state.drag.y0, state.drag.y1);
  const w = Math.abs(state.drag.x1 - state.drag.x0);
  const h = Math.abs(state.drag.y1 - state.drag.y0);
  const r = document.createElementNS(svgNS, 'rect');
  r.setAttribute('class', 'draft-rect');
  r.setAttribute('x', x); r.setAttribute('y', y); r.setAttribute('width', w); r.setAttribute('height', h);
  r.setAttribute('fill', 'rgba(255,255,255,0.12)');
  r.setAttribute('stroke', '#fff');
  r.setAttribute('stroke-width', 2 / state.zoom);
  r.setAttribute('stroke-dasharray', `${6 / state.zoom} ${6 / state.zoom}`);
  overlay_svg.appendChild(r);
}

wrap.addEventListener('wheel', (e) => {
  if (e.target.closest('#right') || e.target.closest('#left')) return;
  e.preventDefault();
  const r = wrap.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const old = state.zoom;
  const factor = Math.exp(-e.deltaY * 0.0016);
  state.zoom = Math.min(40, Math.max(0.05, old * factor));
  const k = state.zoom / old;
  state.pan = { x: mx - (mx - state.pan.x) * k, y: my - (my - state.pan.y) * k };
  applyTransform();
}, { passive: false });

wrap.addEventListener('contextmenu', (e) => e.preventDefault());

// dblclick: zoom to fit (or toggle)
wrap.addEventListener('dblclick', async () => { await fitView(); applyTransform(); });

// --- keyboard -----------------------------------------------------------------

function sortedBoxes() {
  const d = state.data;
  if (!d) return [];
  return [...d.annotations].sort((a, b) => (a.bbox[1] - b.bbox[1]) || (a.bbox[0] - b.bbox[0]));
}

document.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toLowerCase();
  const typing = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;
  if (e.key === ' ' && !typing) {
    state.space = true;
    wrap.classList.add('space');
    if (!state.drawing) e.preventDefault();
    return;
  }
  if (typing) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  switch (e.key) {
    case 'Escape':
      if (state.drawing) endDrawing();
      else selectBox(null);
      return;
    case 'n':
      e.preventDefault();
      startDrawing();
      return;
    case 'Delete':
    case 'Backspace':
      if (state.sel != null && state.data) {
        e.preventDefault();
        const a = state.data.annotations.find((x) => x.id === state.sel);
        if (a) {
          a.attributes = { ...a.attributes, source: 'human' };
          state.data.annotations = state.data.annotations.filter((x) => x.id !== state.sel);
          state.sel = null;
          renderAll();
          scheduleSave();
        }
      }
      return;
    case 'Tab':
      e.preventDefault();
      cycleBox(e.shiftKey ? -1 : 1);
      return;
    case 'v':
    case 'V':
      e.preventDefault();
      if (state.data) markVerified(!state.data.verified);
      return;
    case 'f':
      e.preventDefault();
      fitView().then(applyTransform);
      return;
    case 'g':
      e.preventDefault();
      state.gridView = !state.gridView;
      history.replaceState(null, '', location.pathname + (state.gridView ? '?mode=grid' : '?mode=edit'));
      wrap.classList.toggle('grid-mode', state.gridView);
      byId('audit-link').textContent = state.gridView ? 'back to canvas' : 'audit';
      byId('audit-link').href = state.gridView ? location.pathname + '?mode=edit' : location.pathname + '?mode=grid';
      renderAll();
      return;
  }
  if (/^[\d]$/.test(e.key) || /^[a-z]$/.test(e.key)) {
    const match = state.classes.find((c) => c.hotkey === e.key);
    if (match && state.sel != null) { e.preventDefault(); setCategory(match.name); }
  }
});

document.addEventListener('keyup', (e) => {
  if (e.key === ' ') { state.space = false; wrap.classList.remove('space'); }
});

function cycleBox(delta) {
  const boxes = sortedBoxes();
  if (!boxes.length) return;
  const idx = boxes.findIndex((b) => b.id === state.sel);
  const next = boxes[(idx + delta + boxes.length) % boxes.length];
  selectBox(next.id);
}

// --- boot --------------------------------------------------------------------

async function boot() {
  try {
    [state.classes] = await Promise.all([
      fetchJSON('/api/classes').then((d) => d.classes),
    ]);
    await refreshImages();
    await loadImage(state.images[0]?.idx ?? 0);
    window.addEventListener('resize', applyTransform);
    // re-render when the image itself loaded (fitView depends on geometry)
    img.addEventListener('load', () => { applyTransform(); drawGridCells(); });
  } catch (e) {
    byId('cur-title').textContent = `load error: ${e.message}`;
    console.error(e);
  }
}

boot();
