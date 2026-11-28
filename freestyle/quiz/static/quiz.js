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
    correct: $('btn-correct'), next: $('btn-next'), nextPart: $('btn-next-part'), undo: $('undo'),
    confirm: $('confirm'), confirmText: $('confirm-text'),
    progressFill: $('progress-fill'), progressBar: $('progress-bar'), chip: $('progress-chip'),
    lightbox: $('lightbox'), lightboxImg: $('lightbox-img'),
    theme: $('theme'), restart: $('restart')
  };

  var queue = null;      // {questions, classes, images, ...}
  var answers = {};      // qid -> answer object
  var index = 0;         // current question index
  var frameOrder = [];   // image ids in queue order (matches /img/frame?i=)
  var partStep = false;  // crop question showing the part chooser first
  var selected = null;   // picked-but-not-submitted answer for the current question
  var busy = false;
  var confirming = false;
  var confirmTimer = 0;
  var fix = { top: 0, bottom: 0 };

  var CONFIRM_MS = 500;  // how long the big ✓ stays up before the answer is posted

  // Short chip labels for the 3-column class grid (full name stays in the title).
  var SHORT = {
    'coca-cola-part1': 'Coke P1',
    'coca-cola-part2': 'Coke P2',
    'diet-coke-part1': 'Diet P1',
    'diet-coke-part2': 'Diet P2',
    'dr-pepper-part1': 'Dr P P1',
    'dr-pepper-part2': 'Dr P P2',
    'coke-zero-part1': 'Zero P1',
    'coke-zero-part2': 'Zero P2',
    'nns-sweetener': 'NNS',
    'ingredient-aa': 'AA',
    'ingredient-cc': 'CC',
    'fanta-orange': 'BB/Fanta',
    'hi-c': 'Hi-C',
    sprite: 'Sprite',
    'flavor-orange': 'Orange',
    'flavor-vanilla': 'Vanilla',
    'flavor-lemon': 'Lemon',
    'flavor-cherry': 'Cherry',
    unknown: "Can't tell"
  };

  function shortLabel(cls) {
    if (SHORT[cls.name]) return SHORT[cls.name];
    var text = String(cls.display || cls.name);
    return text.length > 11 ? text.slice(0, 10) + '…' : text;
  }

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

  function currentQuestion() {
    return queue && index < queue.questions.length ? queue.questions[index] : null;
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
    selected = null;
    window.scrollTo(0, 0);
    if (q.kind === 'count') renderCount(q); else renderCrop(q);
    syncSelection();
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
    // 'unknown' lives in the secondary row as "Can't tell" to keep the grid to 6 rows
    queue.classes.forEach(function (cls, i) {
      if (cls.name === 'unknown') return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'class-btn' + (cls.name === q.guess ? ' class-btn--guess' : '');
      btn.dataset.class = cls.name;
      btn.title = (cls.display || cls.name) + (i < keys.length ? ' (key ' + keys[i] + ')' : '');
      var swatch = document.createElement('span');
      swatch.className = 'class-btn__swatch';
      swatch.style.background = cls.color;
      var label = document.createElement('span');
      label.textContent = shortLabel(cls);
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

  // ------------------------------------------------- selection + confirm

  function answerKey(answer) {
    if (!answer) return '';
    if (answer.result === 'class') return 'class:' + answer.class;
    if (answer.result === 'part') return 'part:' + answer.part;
    return 'crop:' + answer.result;
  }

  function datasetKey(dataset) {
    if (dataset.part) return 'part:' + dataset.part;
    if (dataset.class) return 'class:' + dataset.class;
    if (dataset.crop) return 'crop:' + dataset.crop;
    return '';
  }

  // Tapping a choice only picks it (tap again to clear); Next submits.
  function toggleSelect(answer) {
    selected = (selected && answerKey(selected) === answerKey(answer)) ? null : answer;
    syncSelection();
  }

  function syncSelection() {
    var stale = document.querySelectorAll('.is-sel');
    for (var i = 0; i < stale.length; i++) stale[i].classList.remove('is-sel');
    var scope = partStep ? els.partAnswers : els.cropAnswers;
    var selKey = answerKey(selected);
    if (scope) {
      var buttons = scope.querySelectorAll('[data-class],[data-crop],[data-part]');
      for (var j = 0; j < buttons.length; j++) {
        var on = !!selKey && datasetKey(buttons[j].dataset) === selKey;
        buttons[j].classList.toggle('is-sel', on);
        buttons[j].setAttribute('aria-pressed', on ? 'true' : 'false');
      }
    }
    var picked = !!selected;
    els.next.hidden = !picked || partStep;
    els.next.disabled = !picked;
    els.correct.hidden = picked || partStep;
    els.nextPart.hidden = !(picked && partStep);
    els.nextPart.disabled = !picked;
  }

  function pressNext() {
    var q = currentQuestion();
    if (!q || !selected) return;
    confirmThenSend(q.qid, selected, q);
  }

  // Big centred ✓ for ~500 ms, then post and advance.
  function confirmThenSend(qid, answer, question) {
    if (busy || confirming) return;
    confirming = true;
    showConfirm(answer, question);
    setTimeout(function () {
      confirming = false;
      send(qid, answer, question);
    }, CONFIRM_MS);
  }

  function showConfirm(answer, question) {
    els.confirmText.textContent = labelFor(answer, question);
    els.confirm.hidden = false;
    clearTimeout(confirmTimer);
    confirmTimer = setTimeout(function () { els.confirm.hidden = true; }, CONFIRM_MS + 200);
  }

  function classShort(name) {
    var classes = (queue && queue.classes) || [];
    for (var i = 0; i < classes.length; i++) {
      if (classes[i].name === name) return shortLabel(classes[i]);
    }
    return SHORT[name] || String(name || '');
  }

  function labelFor(answer, question) {
    var result = answer.result;
    if (result === 'class') return classShort(answer.class);
    if (result === 'part') return 'Part ' + answer.part;
    if (result === 'not_cartridge') return 'Not a cartridge';
    if (result === 'two_plus') return '2+';
    if (result === 'partial') return 'Partial';
    if (result === 'skip') return 'Skipped';
    if (result === 'not_cabinet') return 'Unusable frame';
    if (result === 'fix') return answer.top + ' top / ' + answer.bottom + ' bottom';
    if (result === 'correct') {
      if (question && question.kind === 'count') return 'Count is right';
      return question ? classShort(question.guess) || question.guess_text || 'Correct' : 'Correct';
    }
    return 'Saved';
  }

  // ---------------------------------------------------------------- answers

  function send(qid, answer, question) {
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
      toast(savedLabel(answer, question));
      render();
    }).catch(function (error) {
      busy = false;
      toast('offline? ' + error, true);
    });
  }

  function savedLabel(answer, question) {
    var label = labelFor(answer, question);
    if (answer.result === 'skip') return '↷ ' + label;
    if (answer.result === 'not_cabinet') return '↻ ' + label;
    return '✓ ' + label;
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

  // Re-offer the current question even if it was skipped/hidden (POST /api/revisit).
  // A revisit is not an answer: it only resets the skip budget server-side.
  function revisit(question) {
    if (busy || !question) return;
    busy = true;
    fetch('/api/revisit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ qid: question.qid })
    }).then(function (response) {
      return response.json().then(function (body) { return { ok: response.ok, body: body }; });
    }).then(function (result) {
      busy = false;
      if (!result.ok) { toast(result.body.error || 'revisit failed', true); return; }
      var state = result.body.state || {};
      answers = state.answers || answers;
      index = typeof state.next_index === 'number' ? state.next_index : index;
      render();
      toast('\u21ba back to it \u2014 tap \u21b5 or \u0079 to answer');
    }).catch(function () { busy = false; toast('revisit failed', true); });
  }

  function answerFromDataset(q, dataset) {
    if (dataset.class) return { result: 'class', class: dataset.class };
    if (dataset.crop) return dataset.crop === 'skip' ? { result: 'skip' } : { result: dataset.crop };
    if (dataset.part && q.ask_part) return { result: 'part', part: Number(dataset.part) };
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
    if (!target || busy || confirming) return;
    var q = currentQuestion();
    if (!q) return;
    var d = target.dataset || {};
    if (target === els.undo) { undo(); return; }
    if (target === els.theme) { toggleTheme(); return; }
    if (target === els.restart) { index = 0; render(); return; }
    if (d.step) {
      var key = d.target;
      if (key) {
        fix[key] = Math.max(0, Math.min(500, fix[key] + Number(d.step)));
        syncFix();
      }
      return;
    }
    if (target === els.next || target === els.nextPart) { pressNext(); return; }
    // count question: the only one-tap answers here are Skip and “not usable”
    if (d.count === 'skip') { send(q.qid, { result: 'skip' }, q); return; }
    if (d.count === 'not_cabinet') { send(q.qid, { result: 'not_cabinet' }, q); return; }
    if (d.count === 'correct') { confirmThenSend(q.qid, { result: 'correct' }, q); return; }
    // "Fix the numbers" only opens the stepper panel; the answer is sent by Save.
    if (d.count === 'fix') return;
    if (target === $('fix-save')) {
      confirmThenSend(q.qid, { result: 'fix', top: fix.top, bottom: fix.bottom }, q);
      return;
    }
    if (target === $('part-unknown')) {
      partStep = false;
      selected = null;
      els.partAnswers.hidden = true;
      els.cropAnswers.hidden = false;
      syncSelection();
      return;
    }
    if (target === els.correct) { confirmThenSend(q.qid, { result: 'correct' }, q); return; }
    // every remaining choice selects only — Next (or Enter) posts it
    var answer = answerFromDataset(q, d);
    if (!answer) return;
    if (answer.result === 'skip') { send(q.qid, answer, q); return; }
    toggleSelect(answer);
  });

  document.addEventListener('click', function (event) {
    if (event.target.closest && event.target.closest('[data-count="fix"]')) {
      els.countFix.hidden = !els.countFix.hidden;
    }
  });

  document.addEventListener('keydown', function (event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (!queue || busy || confirming || index >= queue.questions.length) return;
    var q = queue.questions[index];
    var key = event.key;
    if (key === 'Escape' && !els.lightbox.hidden) { event.preventDefault(); closeZoom(); return; }
    if (key === 'u') { event.preventDefault(); undo(); return; }
    if (key === 'r') { event.preventDefault(); revisit(q); return; }
    if (key === 's') { event.preventDefault(); send(q.qid, { result: 'skip' }, q); return; }
    if (key === 'Enter') {
      if (selected) { event.preventDefault(); pressNext(); }
      return;
    }
    if (q.kind === 'count') {
      if (key === 'y') { event.preventDefault(); confirmThenSend(q.qid, { result: 'correct' }, q); }
      else if (key === 'f') { event.preventDefault(); els.countFix.hidden = false; }
      else if (key === 'x') { event.preventDefault(); send(q.qid, { result: 'not_cabinet' }, q); }
      return;
    }
    var choice = null;
    if (partStep) {
      if (key === '1' || key === '2') choice = { result: 'part', part: Number(key) };
      else if (key === 'y' || key === 'c') {
        // "Neither": drop down to the full class grid
        event.preventDefault();
        partStep = false;
        selected = null;
        els.partAnswers.hidden = true;
        els.cropAnswers.hidden = false;
        syncSelection();
      }
    } else if (key === 'y') {
      event.preventDefault();
      confirmThenSend(q.qid, { result: 'correct' }, q);
      return;
    } else {
      var digits = '123456789';
      var slot = digits.indexOf(key);
      if (slot >= 0 && queue.classes[slot]) choice = { result: 'class', class: queue.classes[slot].name };
      else if (key === 'n') choice = { result: 'not_cartridge' };
      else if (key === 'k') choice = { result: 'class', class: 'unknown' };
      else if (key === 'p') choice = { result: 'partial' };
      else if (key === 'm') choice = { result: 'two_plus' };
    }
    if (choice) {
      event.preventDefault();
      toggleSelect(choice);
    }
  });

  document.addEventListener('click', function (event) {
    var zoom = event.target.closest ? event.target.closest('[data-zoom]') : null;
    if (zoom) {
      els.lightboxImg.src = zoom.getAttribute('data-zoom') || zoom.src;
      els.lightboxImg.alt = zoom.alt || 'zoomed image';
      els.lightbox.hidden = false;
      document.body.classList.add('scroll-locked');
      zoomTap.x = null;
      zoomTap.y = null;
    }
  });

  function closeZoom() {
    els.lightbox.hidden = true;
    document.body.classList.remove('scroll-locked');
    els.lightboxImg.removeAttribute('src');
    zoomTap.x = null;
    zoomTap.y = null;
  }

  // Tap anywhere in the lightbox closes it — but a pinch/drag to zoom must not.
  var zoomTap = { x: null, y: null };

  els.lightbox.addEventListener('pointerdown', function (event) {
    zoomTap.x = event.clientX;
    zoomTap.y = event.clientY;
  });

  els.lightbox.addEventListener('click', function (event) {
    var moved = zoomTap.y === null ? 0
      : Math.max(Math.abs(event.clientY - zoomTap.y), Math.abs(event.clientX - zoomTap.x));
    zoomTap.x = null;
    zoomTap.y = null;
    if (moved > 12) return;
    closeZoom();
  });

  function toggleTheme() {
    var next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('freestyle-quiz:theme', next); } catch (e) {}
  }

  boot();
})();
