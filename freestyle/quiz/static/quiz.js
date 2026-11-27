/* Freestyle GT Quiz — phone-first, one question at a time.
   Answers POST inline; the queue is loaded once and walked locally. */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    status: $('status'), quiz: $('quiz'), done: $('done'), doneText: $('done-text'),
    kind: $('q-kind'), where: $('q-where'),
    countBody: $('q-count'), cropBody: $('q-crop'),
    countAsk: $('count-ask'), countImg: $('count-img'), countFix: $('count-fix'),
    fixTop: $('fix-top'), fixBottom: $('fix-bottom'),
    split: $('q-split'),
    cropImg: $('crop-img'), cropCaption: $('crop-caption'), contextImg: $('context-img'),
    contextCaption: $('context-caption'), guessText: $('guess-text'), guessConf: $('guess-conf'),
    chipText: $('progress-short'),
    partAnswers: $('part-answers'), cropAnswers: $('crop-answers'), classGrid: $('class-grid'),
    correct: $('btn-correct'), undo: $('undo'),
    progressFill: $('progress-fill'), progressBar: $('progress-bar'), chip: $('progress-chip'),
    lightbox: $('lightbox'), lightboxImg: $('lightbox-img'),
    theme: $('theme'), restart: $('restart')
  };

  var queue = null;      // {questions, classes, images, ...}
  var answers = {};      // qid -> answer object
  var index = 0;         // current question index
  var frameOrder = [];   // image ids in queue order (matches /img/frame?i=)
  var partStep = false;  // crop question showing the part chooser first
  var busy = false;
  var fix = { top: 0, bottom: 0 };

  // ---------------------------------------------------------------- helpers

  function toast(text, bad) {
    var el = document.createElement('div');
    el.className = 'toast' + (bad ? ' toast--bad' : '');
    el.textContent = text;
    document.body.appendChild(el);
    requestAnimationFrame(function () { el.classList.add('toast--show'); });
    setTimeout(function () {
      el.classList.remove('toast--show');
      setTimeout(function () { el.remove(); }, 250);
    }, 1100);
  }

  function pct(value) { return Math.round((Number(value) || 0) * 100) + '%'; }

  function imagePosition(imageId) {
    var i = frameOrder.indexOf(imageId);
    return i < 0 ? 1 : i + 1;
  }

  function drawProgress() {
    if (!queue) return;
    var total = queue.questions.length;
    var answered = Object.keys(answers).length;
    var images = queue.images.length;
    var current = index < total ? queue.questions[index].image_id : null;
    var line = answered + ' / ' + total + ' · image ' +
      (current ? imagePosition(current) : images) + ' of ' + images;
    if (els.where) els.where.textContent = line;
    els.chipText.textContent = answered + ' / ' + total;
    var percent = total ? Math.round((answered / total) * 100) : 100;
    els.progressFill.style.width = percent + '%';
    els.progressBar.setAttribute('aria-valuenow', String(percent));
    els.chip.classList.toggle('live-chip--warn', answered === 0 && total > 0);
    els.undo.disabled = Object.keys(answers).length === 0;
  }

  function prefetch() {
    // warm the server cache + the browser cache for the next few crops
    var host = location.origin;
    for (var k = 1; k <= 3; k++) {
      var q = queue.questions[index + k];
      if (!q) break;
      if (q.kind === 'crop') {
        new Image().src = host + '/img/crop?i=' + (index + k);
        new Image().src = host + '/img/context?i=' + (index + k);
      } else if (q.kind === 'count') {
        new Image().src = host + '/img/frame?i=' + (imagePosition(q.image_id) - 1);
      }
    }
  }

  // ---------------------------------------------------------------- render

  function render() {
    if (!queue) return;
    if (index >= queue.questions.length) {
      els.quiz.hidden = true;
      els.status.hidden = true;
      els.done.hidden = false;
      els.doneText.textContent = 'All ' + queue.questions.length + ' questions answered. ' +
        'Export with: python -m freestyle.quiz.to_coco';
      drawProgress();
      return;
    }
    els.done.hidden = true;
    els.status.hidden = true;
    els.quiz.hidden = false;
    var q = queue.questions[index];
    partStep = false;
    window.scrollTo(0, 0);
    if (q.kind === 'count') renderCount(q); else renderCrop(q);
    drawProgress();
    prefetch();
  }

  function renderCount(q) {
    els.kind.textContent = 'count';
    els.kind.className = 'tag-pill tag-pill--cola';
    els.split.textContent = q.split || '';
    els.countAsk.innerHTML = 'Model counts <strong>' + q.model_top + ' top / ' + q.model_bottom +
      ' bottom</strong> cartridges (' + q.model_total + ' total). Correct?';
    els.cropBody.hidden = true;
    els.countBody.hidden = false;
    els.countFix.hidden = true;
    fix = { top: q.model_top, bottom: q.model_bottom };
    syncFix();
    var src = '/img/frame?i=' + (imagePosition(q.image_id) - 1);
    els.countImg.src = src;
    els.countImg.setAttribute('data-zoom', src + '&full=1');
  }

  function syncFix() {
    els.fixTop.textContent = String(fix.top);
    els.fixBottom.textContent = String(fix.bottom);
  }

  function renderCrop(q) {
    els.kind.textContent = 'crop ' + (q.number || '');
    els.kind.className = 'tag-pill tag-pill--sky';
    els.split.textContent = q.split || '';
    els.countBody.hidden = true;
    els.cropBody.hidden = false;
    els.cropImg.src = '/img/crop?i=' + index;
    els.cropImg.setAttribute('data-zoom', '/img/crop?i=' + index);
    els.contextImg.src = '/img/context?i=' + index;
    els.cropCaption.textContent = 'box ' + (q.number || '?') + ' · ' + (q.shelf || '?') + ' shelf';
    els.contextCaption.textContent = q.image_id.split('/').pop();
    els.guessText.textContent = q.guess_text || q.guess;
    els.guessConf.textContent = pct(q.confidence);
    buildClassGrid(q);
    var askPart = !!q.ask_part && !answers[q.qid];
    partStep = askPart;
    els.partAnswers.hidden = !askPart;
    els.cropAnswers.hidden = askPart;
    els.correct.innerHTML = '✓ Correct <span class="class-btn__key">(' + esc(q.guess_text || q.guess) + ')</span>';
  }

  function esc(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function buildClassGrid(q) {
    if (els.classGrid.dataset.built === q.qid) return;
    els.classGrid.dataset.built = q.qid;
    els.classGrid.innerHTML = '';
    var keys = '123456789';
    queue.classes.forEach(function (cls, i) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'class-btn' + (cls.name === q.guess ? ' class-btn--guess' : '');
      btn.dataset.class = cls.name;
      var swatch = document.createElement('span');
      swatch.className = 'class-btn__swatch';
      swatch.style.background = cls.color;
      var label = document.createElement('span');
      label.textContent = cls.name === 'unknown' ? "Can't tell (unknown)" : cls.display;
      btn.appendChild(swatch);
      btn.appendChild(label);
      if (i < keys.length) {
        var key = document.createElement('span');
        key.className = 'class-btn__key';
        key.textContent = keys[i];
        btn.appendChild(key);
      }
      els.classGrid.appendChild(btn);
    });
  }

  // ---------------------------------------------------------------- answers

  function send(qid, answer) {
    if (busy) return;
    busy = true;
    fetch('/api/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ qid: qid, answer: answer })
    }).then(function (response) {
      return response.json().then(function (body) { return { ok: response.ok, body: body }; });
    }).then(function (result) {
      if (!result.ok) {
        busy = false;
        toast(result.body.error || 'not saved', true);
        return;
      }
      answers = result.body.state.answers || {};
      index = result.body.state.next_index;
      busy = false;
      toast(savedLabel(answer));
      render();
    }).catch(function (error) {
      busy = false;
      toast('offline? ' + error, true);
    });
  }

  function savedLabel(answer) {
    var map = {
      correct: '✓ saved', fix: 'count saved', not_cabinet: 'marked unusable',
      class: 'saved', part: 'saved', not_cartridge: 'not a cartridge',
      two_plus: '2+ flagged', partial: 'partial flagged', skip: 'skipped'
    };
    return map[answer.result] || 'saved';
  }

  function advance() {
    while (index < queue.questions.length && answers[queue.questions[index].qid]) index++;
    render();
  }

  function undo() {
    if (busy) return;
    busy = true;
    fetch('/api/undo', { method: 'POST' }).then(function (response) {
      return response.json().then(function (body) { return { ok: response.ok, body: body }; });
    }).then(function (result) {
      busy = false;
      if (!result.ok) { toast(result.body.error || 'nothing to undo', true); return; }
      var removed = result.body.removed.qid;
      var state = result.body.state || {};
      answers = state.answers || {};
      index = typeof state.next_index === 'number' ? state.next_index : 0;
      render();
      if (removed !== undefined) toast('↩ undone');
    }).catch(function () { busy = false; toast('undo failed', true); });
  }

  function answerFromButton(q, dataset) {
    if (dataset.count) return { result: dataset.count };
    if (dataset.crop) return { result: dataset.crop };
    if (dataset.class) return { result: 'class', class: dataset.class };
    if (dataset.part) return { result: 'part', part: Number(dataset.part) };
    return null;
  }

  // ---------------------------------------------------------------- wiring

  function firstUnanswered(state) {
    var jump = new URLSearchParams(location.search).get('i');
    if (jump !== null && /^\d+$/.test(jump)) {
      return Math.min(queue.questions.length - 1, Number(jump));
    }
    var n = typeof state.next_index === 'number' ? state.next_index : 0;
    return n;
  }

  function boot() {
    fetch('/api/queue').then(function (r) { return r.json(); }).then(function (payload) {
      queue = payload;
      answers = payload.answers || {};
      frameOrder = (queue.images || []).map(function (im) { return im.image_id; });
      index = firstUnanswered(payload);
      render();
      if (Object.keys(answers).length) toast('resumed at ' + (index + 1) + ' / ' + queue.questions.length);
    }).catch(function (error) {
      els.status.textContent = 'Could not load the queue: ' + error;
    });
  }

  document.addEventListener('click', function (event) {
    var target = event.target.closest ? event.target.closest('button') : null;
    if (!target || busy) return;
    var q = queue && index < queue.questions.length ? queue.questions[index] : null;
    if (!q) return;
    if (target === els.undo) { undo(); return; }
    if (target === els.theme) { toggleTheme(); return; }
    if (target === els.restart) { index = 0; render(); return; }
    if (target.dataset && target.dataset.step) {
      var key = target.dataset.target;
      if (key) {
        fix[key] = Math.max(0, Math.min(500, fix[key] + Number(target.dataset.step)));
        syncFix();
      }
      return;
    }
    if (target === $('fix-save')) { send(q.qid, { result: 'fix', top: fix.top, bottom: fix.bottom }); return; }
    if (target === $('part-unknown')) {
      partStep = false;
      els.partAnswers.hidden = true;
      els.cropAnswers.hidden = false;
      return;
    }
    if (target === els.correct) { send(q.qid, { result: 'correct' }); return; }
    var answer = answerFromButton(q, target.dataset || {});
    if (answer) send(q.qid, answer);
  });

  document.addEventListener('click', function (event) {
    if (event.target.closest && event.target.closest('[data-count="fix"]')) {
      els.countFix.hidden = !els.countFix.hidden;
    }
  });

  document.addEventListener('keydown', function (event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (!queue || busy || index >= queue.questions.length) return;
    var q = queue.questions[index];
    var key = event.key;
    if (key === 'Escape' && !els.lightbox.hidden) { event.preventDefault(); closeZoom(); return; }
    if (key === 'u') { event.preventDefault(); undo(); return; }
    if (key === 's') { event.preventDefault(); send(q.qid, { result: 'skip' }); return; }
    if (q.kind === 'count') {
      if (key === 'y') { event.preventDefault(); send(q.qid, { result: 'correct' }); }
      else if (key === 'f') { event.preventDefault(); els.countFix.hidden = false; }
      else if (key === 'x') { event.preventDefault(); send(q.qid, { result: 'not_cabinet' }); }
      return;
    }
    if (partStep) {
      if (key === '1' || key === '2') { event.preventDefault(); send(q.qid, { result: 'part', part: Number(key) }); }
      else if (key === 'y' || key === 'c') {
        event.preventDefault();
        partStep = false;
        els.partAnswers.hidden = true;
        els.cropAnswers.hidden = false;
      }
      return;
    }
    if (key === 'y') { event.preventDefault(); send(q.qid, { result: 'correct' }); return; }
    if (key === 'n') { event.preventDefault(); send(q.qid, { result: 'not_cartridge' }); return; }
    var digits = '123456789';
    var slot = digits.indexOf(key);
    if (slot >= 0 && queue.classes[slot]) {
      event.preventDefault();
      send(q.qid, { result: 'class', class: queue.classes[slot].name });
    }
  });

  document.addEventListener('click', function (event) {
    var zoom = event.target.closest ? event.target.closest('[data-zoom]') : null;
    if (zoom) {
      els.lightboxImg.src = zoom.getAttribute('data-zoom') || zoom.src;
      els.lightboxImg.alt = zoom.alt || 'zoomed image';
      els.lightbox.hidden = false;
      document.body.classList.add('scroll-locked');
    }
  });

  function closeZoom() {
    els.lightbox.hidden = true;
    document.body.classList.remove('scroll-locked');
    els.lightboxImg.removeAttribute('src');
  }

  document.addEventListener('click', function (event) {
    if (!els.lightbox.hidden && (event.target === els.lightbox || event.target.id === 'lightbox-close')) {
      closeZoom();
    }
  });

  function toggleTheme() {
    var next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('freestyle-quiz:theme', next); } catch (e) {}
  }

  boot();
})();
