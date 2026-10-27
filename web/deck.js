/* Sentinel web deck: charts from data.js, navigation, and the timeline rail. */
(() => {
  'use strict';
  const D = window.SENTINEL;
  const root = document.documentElement;
  const stage = document.getElementById('stage');
  const slides = [...document.querySelectorAll('.slide')];
  const params = new URLSearchParams(location.search);
  const still = params.has('still');
  if (still) root.classList.add('still');
  try { const t = localStorage.getItem('sentinel-deck-theme'); if (t) root.dataset.theme = t; } catch (e) { /* storage blocked */ }
  if (params.get('theme')) root.dataset.theme = params.get('theme');

  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const svg = (parent, w, h) => el('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}` }, parent);
  const div = (parent, cls) => { const d = document.createElement('div'); d.className = cls; parent.appendChild(d); return d; };
  const text = (parent, x, y, str, attrs = {}) => { const t = el('text', { x, y, ...attrs }, parent); t.textContent = str; return t; };
  const fmt = (x, dec = 0) => x.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
  const WATCHER = { level: 'spike', seasonal: 'routine', shift: 'new level', volatility: 'jitter', novelty: 'new shape' };
  let uid = 0;

  function sized(host, dur, delay) {
    const W = +host.dataset.w, H = +host.dataset.h;
    host.style.width = W + 'px';
    host.style.height = H + 'px';
    host.style.setProperty('--dur', dur + 'ms');
    host.style.setProperty('--delay', delay + 'ms');
    return [W, H];
  }

  // A real series that draws itself left to right; alarms light up as the scan passes them.
  function signal(host, s, o = {}) {
    const dur = o.dur ?? 2400, delay = o.delay ?? 500;
    const [W, H] = sized(host, dur, delay);
    const pt = o.pt ?? 30, pb = o.pb ?? 4, g = o.gamma ?? 1;
    const X = x => x * W;
    const Y = y => pt + (1 - Math.pow(Math.max(0, y), g)) * (H - pt - pb);

    const base = svg(host, W, H);
    for (const [a, b] of s.windows) {
      const x = X(a), w = Math.max(4, X(b) - X(a));
      el('rect', { x, y: 0, width: w, height: H - pb, class: 'band' }, base);
      el('line', { x1: x, x2: x + w, y1: 1, y2: 1, class: 'band-edge' }, base);
    }
    el('line', { x1: 0, x2: W, y1: H - pb + .5, y2: H - pb + .5, class: 'axis' }, base);

    const sv = svg(div(host, 'reveal'), W, H);
    const d = 'M' + s.pts.map(([x, y]) => X(x).toFixed(1) + ',' + Y(y).toFixed(1)).join('L');
    if (o.glow !== false) el('path', { d, class: 'glow' }, sv);
    el('path', { d, class: 'trace' }, sv);
    if (o.sigma) for (const x of s.sigma) el('line', { x1: X(x), x2: X(x), y1: H - pb + 10, y2: H - pb + 30, class: 'tick' }, sv);
    div(host, 'scan');

    if (o.alarms === 'none') return;
    const ms = svg(host, W, H);
    for (const a of s.alarms) {
      const x = X(a.x), y = Y(a.y), top = o.stemTop ?? 0;
      const gg = el('g', { class: 'beacon' + (a.hit ? '' : ' false'), style: `--t:${Math.round(delay + a.x * dur)}ms` }, ms);
      if (o.stems !== false && y - top > 18) el('line', { x1: x, x2: x, y1: top, y2: y - 10, class: 'stem' }, gg);
      if (a.hit) {
        el('circle', { cx: x, cy: y, r: (o.r ?? 5.5) * 2, class: 'halo' }, gg);
        el('circle', { cx: x, cy: y, r: 9, class: 'ring' }, gg);
      }
      el('circle', { cx: x, cy: y, r: o.r ?? 5.5, class: 'core' }, gg);
      const lab = o.label && o.label(a);
      if (lab) {
        const t = text(gg, x + 14, top + 12, lab[0], { class: 'blabel' });
        const sub = el('tspan', { class: 'sub' }, t);
        sub.textContent = '  ·  ' + lab[1];
      }
    }
  }

  // Rarity score (-log10 p) with the y axis written as "1 in N".
  function scoreChart(host, sc, o = {}) {
    const dur = o.dur ?? 2400, delay = o.delay ?? 500;
    const [W, H] = sized(host, dur, delay);
    const CAP = 10, pt = 6, pb = 4;
    const X = x => x * W;
    const Y = v => pt + (1 - Math.min(v, CAP) / CAP) * (H - pt - pb);
    const base = svg(host, W, H);
    const labels = { 1: '1 in 10', 3: '1 in 1,000', 5: '1 in 100,000', 7: '1 in 10 million', 9: '1 in a billion' };
    for (const [v, t] of Object.entries(labels)) {
      const hot = +v === 5;
      el('line', { x1: 0, x2: W, y1: Y(+v), y2: Y(+v), class: hot ? 'alarm-line' : 'grid-line' }, base);
      const l = div(host, 'ylab' + (hot ? ' hot' : ''));
      l.style.top = Y(+v) + 'px';
      l.textContent = t;
    }
    text(base, W, Y(5) - 12, 'alarm line', { class: 'blabel', 'text-anchor': 'end' });
    el('line', { x1: 0, x2: W, y1: H - pb + .5, y2: H - pb + .5, class: 'axis' }, base);

    const sv = svg(div(host, 'reveal'), W, H);
    const id = 'clip' + uid++;
    el('rect', { x: -10, y: -60, width: W + 20, height: Y(5) + 60 }, el('clipPath', { id }, el('defs', {}, sv)));
    const d = 'M' + sc.map(([x, v]) => X(x).toFixed(1) + ',' + Y(v).toFixed(1)).join('L');
    el('path', { d: `${d}L${W},${H - pb}L0,${H - pb}Z`, class: 'area-hot', 'clip-path': `url(#${id})` }, sv);
    el('path', { d, class: 'score' }, sv);
    el('path', { d, class: 'score-hot', 'clip-path': `url(#${id})` }, sv);
    div(host, 'scan');
  }

  // Slide 6: best-so-far F1 across phase 1 versions and phase 2 runs.
  function trajectory(host) {
    const [W, H] = sized(host, 0, 0);
    const P = [['v1', .343, 'p1'], ['v2', .473, 'p1'], ['v3', .500, 'p1'], ['v4', .535, 'p1'], ['v5', .551, 'p1'],
      ['run 2', .561, 'keep'], ['run 3', .544, 'drop'], ['run 4', .584, 'keep'], ['run 5', .556, 'drop'],
      ['run 6', .594, 'keep'], ['run 7', .571, 'drop']];
    const l = 60, r = 24, t = 100, b = 50;
    const X = i => l + 26 + i * (W - l - r - 52) / (P.length - 1);
    const Y = v => t + (1 - (v - .3) / (.64 - .3)) * (H - t - b);
    const s = svg(host, W, H);
    for (const v of [.3, .4, .5, .6]) {
      el('line', { x1: l, x2: W - r, y1: Y(v), y2: Y(v), class: 'grid-line' }, s);
      text(s, l - 14, Y(v) + 5, v.toFixed(1), { class: 'lbl', 'text-anchor': 'end' });
    }
    const xd = (X(4) + X(5)) / 2;
    el('line', { x1: xd, x2: xd, y1: 14, y2: H - b, class: 'divider' }, s);
    text(s, l, 30, 'Phase 1 · building it', { class: 'phase' });
    text(s, l, 56, '124 logged runs, v1 to v5', { class: 'phase-sub' });
    text(s, xd + 22, 30, 'Phase 2 · 15-run budget', { class: 'phase' });
    text(s, xd + 22, 56, '7 used · filled dot = kept', { class: 'phase-sub' });
    let best = -1, d = '';
    P.forEach(([, v], i) => {
      best = Math.max(best, v);
      d += i === 0 ? `M${X(0) - 26},${Y(best)}` : `H${X(i)}V${Y(best)}`;
    });
    el('path', { d: d + `H${W - r}`, class: 'step', pathLength: 1 }, s);
    P.forEach(([n, v, k], i) => {
      el('circle', { cx: X(i), cy: Y(v), r: k === 'p1' ? 6.5 : 8, class: `pt ${k}`, style: `--t:${500 + i * 170}ms` }, s);
      text(s, X(i), H - b + 32, n, { class: 'lbl' + (i === 9 ? ' hot' : ''), 'text-anchor': 'middle' });
    });
    text(s, X(0), Y(.343) + 30, '0.343', { class: 'lbl', 'text-anchor': 'middle' });
    text(s, X(4), Y(.551) + 32, '0.551', { class: 'lbl', 'text-anchor': 'middle' });
    text(s, X(9), Y(.594) - 26, '0.594', { class: 'final', 'text-anchor': 'middle' });
  }

  // Slide 7: a drawn example of caught / false alarm / missed.
  function explainer(host) {
    const [W, H] = sized(host, 2200, 500);
    const pb = 50, y0 = 150;
    const A = [120, 300], B = [640, 820];
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const bump = (x, c, w, h) => h * Math.exp(-((x - c) ** 2) / (2 * w * w));
    const pts = [];
    for (let x = 0; x <= W; x += 5) {
      const y = y0 + Math.sin(x * .05) * 4 + (rnd() - .5) * 9
        - bump(x, 205, 7, 118) - bump(x, 455, 6, 84) - bump(x, 735, 16, 26);
      pts.push([x, y]);
    }
    const base = svg(host, W, H);
    for (const [a, b] of [A, B]) {
      el('rect', { x: a, y: 0, width: b - a, height: H - pb, class: 'band' }, base);
      el('line', { x1: a, x2: b, y1: 1, y2: 1, class: 'band-edge' }, base);
    }
    el('line', { x1: 0, x2: W, y1: H - pb + .5, y2: H - pb + .5, class: 'axis' }, base);
    const sv = svg(div(host, 'reveal'), W, H);
    const d = 'M' + pts.map(([x, y]) => x + ',' + y.toFixed(1)).join('L');
    el('path', { d, class: 'glow' }, sv);
    el('path', { d, class: 'trace' }, sv);
    div(host, 'scan');
    const ms = svg(host, W, H);
    const mark = (x, hit) => {
      const y = Math.min(...pts.filter(p => Math.abs(p[0] - x) < 8).map(p => p[1]));
      const g = el('g', { class: 'beacon' + (hit ? '' : ' false'), style: `--t:${Math.round(500 + x / W * 2200)}ms` }, ms);
      if (hit) { el('circle', { cx: x, cy: y, r: 11, class: 'halo' }, g); el('circle', { cx: x, cy: y, r: 9, class: 'ring' }, g); }
      el('circle', { cx: x, cy: y, r: 6, class: 'core' }, g);
    };
    mark(205, true);
    mark(455, false);
    const lbl = (x, str, color, t) => {
      const g = el('g', { class: 'beacon', style: `--t:${t}ms` }, ms);
      text(g, x, H - 14, str, { class: 'explainer-lbl', 'text-anchor': 'middle', style: `fill:${color}` });
    };
    lbl(210, 'caught', 'var(--beacon)', 1000);
    lbl(455, 'false alarm', 'var(--ink-2)', 1100);
    lbl(730, 'missed', 'var(--ink-3)', 2400);
  }

  // ----------------------------------------------------------- HTML bar lists
  function bars() {
    const C3 = [['Twitter mentions', 10, 158631], ['Known machine failures', 7, 69561], ['Cloud servers', 17, 67740],
      ['Synthetic tests', 11, 44352], ['Road traffic', 7, 15664], ['Online ad prices', 6, 9610]];
    document.querySelector('.s3 .cats').innerHTML =
      '<div class="cat head"><span>category · signals</span><span></span><em>values</em></div>' +
      C3.map(([n, k, v], i) => `<div class="cat"><span>${n}<i class="c">${k}</i></span>
        <div class="bar"><i class="trace" style="--w:${(v / 158631 * 100).toFixed(1)}%;--d:${450 + i * 70}"></i></div>
        <em>${Math.round(v / 1000)}k</em></div>`).join('');

    const NAMES = { realTweets: 'Twitter mentions', realAWSCloudwatch: 'Cloud servers', realKnownCause: 'Known failures',
      realAdExchange: 'Online ad prices', realTraffic: 'Road traffic', artificialWithAnomaly: 'Synthetic tests' };
    const cats = Object.entries(D.categories).sort((a, b) => b[1].f1 - a[1].f1);
    document.querySelector('.s7 .catbars').innerHTML = cats.map(([k, m], i) =>
      `<div class="catbar"><span>${NAMES[k]}</span><div class="bar"><i class="trace" style="--w:${(m.f1 * 100).toFixed(1)}%;--d:${800 + i * 80}"></i></div><b>${m.f1.toFixed(2)}</b></div>`).join('');

    const BN = { lof: 'Local Outlier Factor', isolation_forest: 'Isolation Forest', ocsvm: 'One-Class SVM', rolling_z: 'Rolling z-score',
      autoencoder: 'Autoencoder', temporal_clustering: 'Temporal clustering', ensemble_mean: 'Ensemble of all 6' };
    const rows = [{ name: 'Sentinel (ours)', f1: D.overall.f1, fp: D.overall.fp, ours: true }]
      .concat(D.baselines.map(b => ({ name: BN[b.method], f1: b.f1, fp: b.fp })).sort((a, b) => b.f1 - a.f1));
    const cmp = (key, max, dec) => rows.map((r, i) =>
      `<div class="cmprow${r.ours ? ' ours' : ''}"><span>${r.name}</span><div class="bar"><i class="${r.ours ? 'hot' : 'dim'}" style="--w:${Math.max(.6, r[key] / max * 100).toFixed(1)}%;--d:${600 + i * 90}"></i></div><b>${fmt(r[key] + 1e-9, dec)}</b></div>`).join('');
    document.querySelector('[data-cmp="f1"]').innerHTML = cmp('f1', .65, 3);
    document.querySelector('[data-cmp="fp"]').innerHTML = cmp('fp', 620, 0);
  }

  function dates() {
    const f = s => new Date(s + 'T00:00:00Z').toLocaleDateString('en-GB', { month: 'short', year: 'numeric', timeZone: 'UTC' });
    document.querySelectorAll('[data-start]').forEach(e => { e.textContent = f(D[e.dataset.start].start); });
    document.querySelectorAll('[data-end]').forEach(e => { e.textContent = f(D[e.dataset.end].end); });
  }

  function charts() {
    const C = name => document.querySelector(`[data-chart="${name}"]`);
    signal(C('hero'), D.hero, { gamma: .55, pt: 56, stemTop: 16, dur: 3600, delay: 700, label: a => ['caught', WATCHER[a.expert]] });
    signal(C('machine'), D.machine, { pt: 16, dur: 2600, delay: 450, alarms: 'none' });
    signal(C('fb'), D.fb, { gamma: .6, pt: 30, pb: 40, dur: 2800, delay: 700, sigma: true });
    signal(C('clean'), D.clean, { gamma: .9, pt: 14, dur: 2800, delay: 600, r: 5 });
    scoreChart(C('score'), D.clean.score, { dur: 2800, delay: 600 });
    signal(C('relearn_before'), D.relearn_before, { gamma: .7, pt: 6, pb: 2, dur: 1500, delay: 1000, glow: false, stems: false, r: 4 });
    signal(C('relearn_after'), D.relearn_after, { gamma: .7, pt: 6, pb: 2, dur: 1500, delay: 1300, glow: false, stems: false, r: 4 });
    trajectory(C('trajectory'));
    explainer(C('explainer'));
    signal(C('ibm'), D.ibm, { gamma: .5, pt: 30, dur: 2600, delay: 500 });
  }

  // ----------------------------------------------------------- rail (progress as a live trace)
  const RW = 1696, RH = 50, RY = 25;
  const nx = i => 8 + i * (RW - 16) / (slides.length - 1);
  const rail = document.getElementById('rail');
  function buildRail() {
    let d = '', seed = 3;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let x = 0; x <= RW; x += 3) {
      const near = Math.min(...slides.map((_, i) => Math.abs(x - nx(i))));
      const amp = Math.min(1, near / 26);
      const y = RY + amp * (Math.sin(x * .07) * 1.6 + Math.sin(x * .021 + 1) * 2.2 + (rnd() - .5) * 3.4);
      d += (x ? 'L' : 'M') + x + ',' + y.toFixed(1);
    }
    el('path', { d, class: 'dim' }, svg(rail, RW, RH));
    const wrap = div(rail, 'litwrap');
    const top = svg(wrap, RW, RH);
    el('path', { d, class: 'lit' }, top);
    const ns = svg(rail, RW, RH);
    slides.forEach((_, i) => {
      el('circle', { cx: nx(i), cy: RY, r: 4.5, class: 'node' }, ns);
      el('circle', { cx: nx(i), cy: RY, r: 18, class: 'hit' }, ns).addEventListener('click', () => go(i));
    });
    div(rail, 'dot');
    const CH = [['Start', 0, 0], ['The problem', 1, 2], ['Our answer', 3, 5], ['The proof', 6, 8], ["What's next", 9, 9]];
    CH.forEach(([name, a, b], k) => {
      const c = div(rail, 'chap');
      c.textContent = name;
      c.dataset.from = a; c.dataset.to = b;
      if (k === CH.length - 1) { c.classList.add('end'); c.style.right = (RW - nx(b) - 4) + 'px'; }
      else { c.style.left = (nx(a) - 4) + 'px'; if (b > a) c.style.width = (nx(b) - nx(a) + 8) + 'px'; }
      c.addEventListener('click', () => go(a));
    });
  }

  // ----------------------------------------------------------- navigation
  let cur = -1;
  const notes = document.getElementById('notes');
  function countUp(slide) {
    for (const e of slide.querySelectorAll('[data-count]')) {
      const to = +e.dataset.count, dec = +(e.dataset.dec || 0);
      if (still) { e.textContent = fmt(to, dec); continue; }
      const host = e.closest('[data-a]');
      const t0 = performance.now() + (host ? parseFloat(host.style.getPropertyValue('--d')) || 0 : 0);
      e.textContent = fmt(0, dec);
      const step = now => {
        const p = Math.min(1, Math.max(0, (now - t0) / 1400));
        e.textContent = fmt(to * (1 - Math.pow(1 - p, 3)), dec);
        if (p < 1 && slide.classList.contains('active')) requestAnimationFrame(step);
        else e.textContent = fmt(to, dec);
      };
      requestAnimationFrame(step);
    }
  }
  function go(i) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    if (i === cur) return;
    slides.forEach((s, k) => s.classList.toggle('active', k === i));
    cur = i;
    countUp(slides[i]);
    stage.classList.toggle('on-title', i === 0);
    document.getElementById('cur').textContent = String(i + 1).padStart(2, '0');
    rail.style.setProperty('--p', nx(i) + 'px');
    rail.querySelectorAll('.node').forEach((n, k) => n.classList.toggle('past', k <= i));
    rail.querySelectorAll('.chap').forEach(c => c.classList.toggle('on', i >= +c.dataset.from && i <= +c.dataset.to));
    document.getElementById('notes-text').textContent = slides[i].querySelector('.notes')?.textContent.trim() || '';
    document.getElementById('notes-n').textContent = `${i + 1} / ${slides.length}`;
    history.replaceState(null, '', location.pathname + location.search + '#' + (i + 1));
  }

  function fit() {
    const s = Math.min(innerWidth / 1920, innerHeight / 1080);
    stage.style.transform = `translate(${(innerWidth - 1920 * s) / 2}px, ${(innerHeight - 1080 * s) / 2}px) scale(${s})`;
  }

  const hint = document.getElementById('hint');
  const hideHint = () => hint.classList.add('gone');
  function toggleTheme() {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    try { localStorage.setItem('sentinel-deck-theme', root.dataset.theme); } catch (e) { /* storage blocked */ }
  }
  document.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(k)) { e.preventDefault(); go(cur + 1); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(k)) { e.preventDefault(); go(cur - 1); }
    else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (/^[0-9]$/.test(k)) go(k === '0' ? 9 : +k - 1);
    else if (k === 'f' || k === 'F') { document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen?.(); }
    else if (k === 't' || k === 'T') toggleTheme();
    else if (k === 'n' || k === 'N') notes.hidden = !notes.hidden;
    else return;
    hideHint();
  });
  let x0 = null;
  addEventListener('touchstart', e => { x0 = e.touches[0].clientX; }, { passive: true });
  addEventListener('touchend', e => {
    if (x0 === null) return;
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 50) { go(cur + (dx < 0 ? 1 : -1)); hideHint(); }
    x0 = null;
  });
  addEventListener('hashchange', () => go((parseInt(location.hash.slice(1), 10) || 1) - 1));
  addEventListener('resize', fit);

  charts();
  bars();
  dates();
  buildRail();
  fit();
  const start = () => go((parseInt(location.hash.slice(1), 10) || 1) - 1);
  if (still) { hideHint(); start(); }
  else {
    setTimeout(hideHint, 6000);
    Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 1200))])
      .then(() => requestAnimationFrame(() => requestAnimationFrame(start)));
  }
})();
