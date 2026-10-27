/* Sentinel live demo: streams a signal through the JavaScript detector (sentinel.js) point by
 * point, draws it, and shows every alarm with its evidence. Nothing is precomputed. */
(() => {
  'use strict';
  const { Sentinel, stepSeconds } = window.SentinelEngine;
  const $ = s => document.querySelector(s);
  const root = document.documentElement;
  const query = new URLSearchParams(location.search);

  const WATCHERS = [
    ['level', 'Spike', 'this value vs. its learned normal'],
    ['seasonal', 'Routine', 'vs. the same time yesterday'],
    ['shift', 'New level', 'last 10 values vs. last 200'],
    ['volatility', 'Jitter', 'how jumpy or frozen it is'],
    ['novelty', 'New shape', 'last 8 values vs. the past 2,000'],
  ];
  const WNAME = Object.fromEntries(WATCHERS.map(([k, n]) => [k, n]));
  const CATS = {
    upload: 'Your data', realTweets: 'Twitter mentions', realAWSCloudwatch: 'Cloud servers (AWS)',
    realKnownCause: 'Known failures', realTraffic: 'Road traffic', realAdExchange: 'Online ad prices',
    artificialWithAnomaly: 'Synthetic tests', artificialNoAnomaly: 'Synthetic, no anomalies',
  };
  const SHOWCASE = ['realTweets/Twitter_volume_CVS.csv', 'realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv',
    'realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv', 'realTweets/Twitter_volume_FB.csv',
    'realKnownCause/machine_temperature_system_failure.csv'];
  const TICKERS = { AAPL: 'Apple', AMZN: 'Amazon', CRM: 'Salesforce', CVS: 'CVS', FB: 'Facebook', GOOG: 'Google',
    IBM: 'IBM', KO: 'Coca-Cola', PFE: 'Pfizer', UPS: 'UPS' };
  const NAMED = {
    'realKnownCause/ambient_temperature_system_failure.csv': 'Office temperature, before a failure',
    'realKnownCause/cpu_utilization_asg_misconfiguration.csv': 'Server CPU, misconfigured auto-scaling',
    'realKnownCause/ec2_request_latency_system_failure.csv': 'Server latency, before a failure',
    'realKnownCause/machine_temperature_system_failure.csv': 'Machine temperature, before a failure',
    'realKnownCause/nyc_taxi.csv': 'New York taxi demand',
    'realKnownCause/rogue_agent_key_hold.csv': 'Rogue agent: key hold',
    'realKnownCause/rogue_agent_key_updown.csv': 'Rogue agent: key up and down',
    'realAWSCloudwatch/grok_asg_anomaly.csv': 'Auto-scaling group load',
  };
  const ART = {
    art_daily_flatmiddle: 'Daily pattern that goes flat', art_daily_jumpsdown: 'Daily pattern that jumps down',
    art_daily_jumpsup: 'Daily pattern that jumps up', art_daily_nojump: 'Daily pattern, subtle change',
    art_increase_spike_density: 'Spikes getting denser', art_load_balancer_spikes: 'Load balancer spikes',
    art_daily_no_noise: 'Daily pattern, no noise', art_daily_perfect_square_wave: 'Perfect square wave',
    art_daily_small_noise: 'Daily pattern, small noise', art_flatline: 'Flat line', art_noisy: 'Pure noise',
  };
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const MONO = '"Geist Mono", ui-monospace, Consolas, monospace';
  const FOLLOW = 2000, GUT = 86, CAP = 10;

  // ------------------------------------------------------------ formatting
  const nf = new Intl.NumberFormat('en-US');
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const sig = v => (v < 10 ? v.toFixed(1).replace(/\.0$/, '') : String(Math.round(v)));
  function oneIn(score) {
    if (!(score > 0.05)) return 'ordinary';
    const n = 10 ** score;
    if (n < 1000) return `1 in ${Math.round(n)}`;
    if (n < 1e6) return `1 in ${sig(n / 1e3)}k`;
    if (n < 1e9) return `1 in ${sig(n / 1e6)}M`;
    return '1 in 1B+';
  }
  function compact(v) {
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e4) return Math.round(v / 1e3) + 'k';
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(1);
    return a === 0 ? '0' : v.toPrecision(2);
  }
  function parts(s) { return String(s).match(/^(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d)/); }
  function fmtTs(s) { const m = parts(s); return m ? `${+m[3]} ${MON[+m[2] - 1]} ${m[1]} · ${m[4]}:${m[5]}` : String(s); }
  function fmtShort(s) { const m = parts(s); return m ? `${+m[3]} ${MON[+m[2] - 1]} · ${m[4]}:${m[5]}` : String(s); }
  function fmtDate(s) { const m = parts(s); return m ? `${+m[3]} ${MON[+m[2] - 1]} ${m[1]}` : String(s); }
  function fmtStep(sec) {
    if (!Number.isFinite(sec) || sec <= 0) return '';
    if (sec % 3600 === 0) return sec === 3600 ? 'one value an hour' : `one value every ${sec / 3600} hours`;
    if (sec % 60 === 0) return `one value every ${sec / 60} min`;
    return `one value every ${sec} s`;
  }
  function pretty(name) {
    if (NAMED[name]) return NAMED[name];
    const [cat, file] = name.split('/');
    const f = file.replace(/\.csv$/, '');
    let m;
    if (ART[f]) return ART[f];
    if ((m = f.match(/^Twitter_volume_(\w+)$/))) return `Tweets about ${TICKERS[m[1]] || m[1]}`;
    if ((m = f.match(/^exchange-(\d)_(cpc|cpm)_results$/))) return `Ad exchange ${m[1]}, ${m[2] === 'cpc' ? 'cost per click' : 'cost per 1,000 views'}`;
    if ((m = f.match(/^(occupancy|speed|TravelTime)_(\w+)$/))) return `${{ occupancy: 'Road occupancy', speed: 'Traffic speed', TravelTime: 'Travel time' }[m[1]]} · sensor ${m[2]}`;
    if ((m = f.match(/^iio_.*_(i-\w+)_NetworkIn$/))) return `Network traffic in · ${m[1]}`;
    if ((m = f.match(/^(ec2|rds|elb)_(.+)_([0-9a-f]{6})$/))) {
      const what = { cpu_utilization: 'CPU', disk_write_bytes: 'disk writes', network_in: 'network in', request_count: 'requests' }[m[2]] || m[2].replace(/_/g, ' ');
      return `${{ ec2: 'Server', rds: 'Database', elb: 'Load balancer' }[m[1]]} ${what} · ${m[3]}`;
    }
    return (cat ? '' : '') + f.replace(/_/g, ' ');
  }

  // ------------------------------------------------------------ state
  const S = {
    manifest: [], byName: new Map(), meta: null, d: null, eng: null, i: 0, N: 0, step: 300,
    scores: null, alarms: [], runs: [], playing: false, speed: 600, view: 'follow', acc: 0, target: null,
    params: { risk: 1e-5, cooldown: 300, novelty_gate: 0 }, engMs: 0, engN: 0, hover: null, focus: null,
    dirty: true, done: false, token: 0,
  };
  const cache = new Map();
  let C = {}, SIG, SCO;

  function readColors() {
    const cs = getComputedStyle(root);
    for (const k of ['bg', 'ink', 'ink-2', 'ink-3', 'line', 'line-2', 'trace', 'trace-soft', 'trace-glow', 'beacon',
      'beacon-soft', 'beacon-glow', 'band', 'band-edge', 'warm']) C[k] = cs.getPropertyValue('--' + k).trim();
    S.dirty = true;
  }
  const isDefault = () => S.params.risk === 1e-5 && S.params.cooldown === 300 && S.params.novelty_gate === 0;
  const hasLabels = () => S.meta && !S.meta.upload;

  // ------------------------------------------------------------ data
  function parseCSV(txt) {
    const lines = txt.split(/\r?\n/).filter(l => l.trim());
    const head = lines[0].split(',').map(s => s.trim().replace(/^"|"$/g, '').toLowerCase());
    let ti = head.indexOf('timestamp'), vi = head.indexOf('value'), start = 1;
    if (vi < 0) {
      const numeric = head.map(h => Number.isFinite(parseFloat(h)));
      if (numeric.some(Boolean)) start = 0;
      vi = head.length > 1 ? 1 : 0;
      ti = head.length > 1 ? 0 : -1;
    }
    const ts = [], vs = [], secs = [];
    for (let k = start; k < lines.length; k++) {
      const c = lines[k].split(',');
      const t = ti >= 0 ? (c[ti] || '').trim().replace(/^"|"$/g, '') : String(k - start);
      ts.push(t);
      vs.push(parseFloat(c[vi]));
      const iso = /^\d{4}-\d\d-\d\d \d\d:\d\d(:\d\d)?$/.test(t) ? t.replace(' ', 'T') + 'Z' : t;
      secs.push(ti >= 0 ? Date.parse(iso) / 1000 : NaN);
    }
    return { ts, vs: Float64Array.from(vs), secs };
  }
  async function loadSeries(meta) {
    if (meta.data) return meta.data;
    if (!cache.has(meta.name)) {
      const r = await fetch('data/series/' + meta.name);
      if (!r.ok) throw new Error(r.status);
      cache.set(meta.name, parseCSV(await r.text()));
    }
    return cache.get(meta.name);
  }

  // ------------------------------------------------------------ streaming
  function reset() {
    S.playing = false;
    S.step = S.d.secs.length > 1 && S.d.secs.every(Number.isFinite) ? stepSeconds(S.d.secs) : 300;
    S.eng = new Sentinel(S.params, S.step);
    Object.assign(S, { i: 0, scores: new Float64Array(S.N), alarms: [], runs: [], engMs: 0, engN: 0,
      target: null, done: false, focus: null, hover: null, acc: 0 });
    $('#cards').innerHTML = `<p class="empty">No alarms yet. Sentinel spends the first <em>300 values</em> learning what normal looks like, then speaks up only when a value is rarer than <em>${oneIn(S.eng.threshold).replace('1 in ', '1 in ')}</em> for this signal. Every alarm appears here with its reason.</p>`;
    $('#done').hidden = true;
    $('#thr-label').textContent = `alarm line · ${oneIn(S.eng.threshold)}`;
    progressMarks();
    renderWatchers(true);
    S.dirty = true;
  }

  function advance(n, budget) {
    const e = S.eng, d = S.d, end = Math.min(S.N, S.i + n), warm = e.p.warmup, i0 = S.i;
    const t0 = performance.now();
    let i = S.i;
    while (i < end) {
      const o = e.update(d.ts[i], d.vs[i]);
      S.scores[i] = o.score;
      if (o.label === 'ANOMALOUS') {
        const r = S.runs[S.runs.length - 1];
        if (r && r[1] === i - 1) r[1] = i; else S.runs.push([i, i]);
      }
      if (o.evidence) addAlarm(i, o.evidence);
      i++;
      if ((i & 31) === 0 && performance.now() - t0 > budget) break;
    }
    const counted = i - Math.max(i0, warm);
    if (counted > 0) { S.engMs += (performance.now() - t0) * counted / (i - i0); S.engN += counted; }
    S.i = i;
    S.dirty = true;
    if (S.i >= S.N) finish();
  }

  const runHit = r => S.meta.windows.some(([a, b]) => r[0] <= b && r[1] >= a);
  function addAlarm(i, ev) {
    const a = { i, ev, run: S.runs[S.runs.length - 1], n: S.alarms.length + 1, born: performance.now() };
    a.hit = hasLabels() ? runHit(a.run) : null;
    S.alarms.push(a);
    if (S.alarms.length === 1) $('#cards').innerHTML = '';
    a.el = card(a);
    $('#cards').prepend(a.el);
    a.mark = document.createElement('u');
    a.mark.style.left = (i / S.N * 100) + '%';
    $('#prog-marks').appendChild(a.mark);
    syncHit(a);
  }
  function syncHit(a) {
    if (hasLabels()) a.hit = runHit(a.run);
    a.el.classList.toggle('false', a.hit === false);
    a.mark.classList.toggle('false', a.hit === false);
    const v = a.el.querySelector('.verdict');
    if (v) {
      v.className = 'verdict' + (a.hit ? ' real' : '');
      v.textContent = a.hit ? 'real anomaly' : 'false alarm';
    }
  }

  function card(a) {
    const ev = a.ev, el = document.createElement('article');
    el.className = 'card';
    const nE = Object.keys(ev.experts).length, perThr = ev.threshold + Math.log10(nE), scale = 12;
    const rows = WATCHERS.filter(([k]) => ev.experts[k]).map(([k, nm]) => {
      const s = -Math.log10(Math.max(ev.experts[k].p, 1e-300));
      return `<div class="${k === ev.expert ? 'top' : ''}"><span>${nm}</span><div class="bar"><i style="width:${Math.min(100, s / scale * 100).toFixed(1)}%"></i><u style="left:${(perThr / scale * 100).toFixed(1)}%"></u></div><span>${oneIn(s)}</span></div>`;
    }).join('');
    el.innerHTML = `<div class="card-h"><span class="card-w">${esc(WNAME[ev.expert] || ev.expert)}</span><span>#${a.n} · ${esc(fmtShort(ev.timestamp))}</span><span class="sc">score ${ev.score.toFixed(1)}</span></div>
      <p>${esc(ev.explanation)}</p>${hasLabels() ? '<span class="verdict"></span>' : ''}
      <div class="mini">${rows}</div>
      <details><summary>Raw evidence (JSON)</summary><pre>${esc(JSON.stringify(ev, null, 1))}</pre></details>`;
    el.addEventListener('click', e => { if (!e.target.closest('details')) focusAlarm(a); });
    return el;
  }
  function focusAlarm(a) {
    const [x0, x1] = domain();
    if (a.i < x0 || a.i >= x1) setView('all');
    S.focus = { i: a.i, t: performance.now() };
    $('#cv-sig').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function finish() {
    if (S.done) return;
    S.done = true;
    S.playing = false;
    setView('all');
    const nA = S.alarms.length, nF = S.alarms.filter(a => a.hit === false).length, m = S.meta;
    const W = m.windows.length, caught = m.windows.filter(([a, b]) => S.runs.some(r => r[0] <= b && r[1] >= a)).length;
    let line = `<b>Finished.</b><span>${nA} alarm${nA === 1 ? '' : 's'}`;
    if (hasLabels()) line += ` · ${caught} of ${W} anomal${W === 1 ? 'y' : 'ies'} caught · ${nF} false alarm${nF === 1 ? '' : 's'}`;
    line += '</span>';
    if (hasLabels() && isDefault()) {
      line += S.alarms.map(a => a.i).join() === m.ref.alarms.join()
        ? '<span class="ok">Identical to our submitted predictions</span>'
        : '<span>Differs from the submitted run</span>';
    } else if (hasLabels()) {
      line += `<span>Custom settings. The submitted settings raise ${m.ref.alarms.length} here.</span>`;
    }
    $('#done').innerHTML = line;
    $('#done').hidden = false;
  }

  // ------------------------------------------------------------ charts
  function setupCanvas(sel) {
    const host = $(sel), cv = host.querySelector('canvas'), ctx = cv.getContext('2d');
    const o = { host, cv, ctx, w: 0, h: 0 };
    const fit = () => {
      const r = host.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
      o.w = r.width; o.h = r.height;
      cv.width = Math.max(1, Math.round(r.width * dpr));
      cv.height = Math.max(1, Math.round(r.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      S.dirty = true;
    };
    new ResizeObserver(fit).observe(host);
    fit();
    return o;
  }
  function domain() {
    if (S.view === 'all' || S.N <= FOLLOW) return [0, Math.max(S.N, 1)];
    const hi = Math.max(S.i, FOLLOW);
    return [hi - FOLLOW, hi];
  }
  function trace(ctx, arr, x0, end, X, Y) {
    ctx.beginPath();
    const n = end - x0, cols = Math.max(1, Math.floor(X(end) - X(x0)));
    let pen = false;
    if (n <= cols * 1.5) {
      for (let k = x0; k < end; k++) {
        const v = arr[k];
        if (v !== v) { pen = false; continue; }
        pen ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v));
        pen = true;
      }
      return;
    }
    const per = n / cols;
    for (let c = 0; c < cols; c++) {
      const a = x0 + Math.floor(c * per), b = Math.min(end, x0 + Math.floor((c + 1) * per));
      let mn = Infinity, mx = -Infinity;
      for (let k = a; k < b; k++) { const v = arr[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mn === Infinity) continue;
      const x = X(a);
      pen ? ctx.lineTo(x, Y(mn)) : ctx.moveTo(x, Y(mn));
      ctx.lineTo(x, Y(mx));
      pen = true;
    }
  }
  function hline(ctx, x0, x1, y, color, dash) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x0, Math.round(y) + .5);
    ctx.lineTo(x1, Math.round(y) + .5);
    ctx.stroke();
    ctx.restore();
  }
  function bands(ctx, x0, x1, X, L, R, bottom) {
    for (const [a, b] of S.meta.windows) {
      if (b < x0 || a >= x1) continue;
      const xa = Math.max(L, X(a)), xb = Math.min(R, X(b + 1));
      ctx.fillStyle = C.band;
      ctx.fillRect(xa, 0, Math.max(2, xb - xa), bottom);
      ctx.fillStyle = C['band-edge'];
      ctx.fillRect(xa, 0, Math.max(2, xb - xa), 2);
    }
  }
  function ring(ctx, x, y, age, life, color) {
    const p = age / life;
    ctx.save();
    ctx.globalAlpha = Math.max(0, 1 - p);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 6 + p * 22, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawSignal(now) {
    const { ctx, w, h } = SIG;
    ctx.clearRect(0, 0, w, h);
    if (!S.d) return;
    const [x0, x1] = domain(), span = x1 - x0, L = GUT, R = w - 12, top = 34, bot = h - 22;
    const X = i => L + (i - x0) / span * (R - L);
    const end = Math.min(S.i, x1);
    let lo = Infinity, hi = -Infinity;
    for (let k = x0; k < end; k++) { const v = S.d.vs[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    const Y = v => bot - (v - lo) / (hi - lo) * (bot - top);

    ctx.font = `11px ${MONO}`;
    ctx.textBaseline = 'middle';
    for (let g = 0; g <= 2; g++) {
      const v = lo + (hi - lo) * g / 2, y = Y(v);
      hline(ctx, L, R, y, C['line-2']);
      if (end > x0) { ctx.fillStyle = C['ink-3']; ctx.textAlign = 'right'; ctx.fillText(compact(v), L - 14, y); }
    }
    const warm = S.eng.p.warmup;
    if (x0 < warm) {
      const xa = X(x0), xb = X(Math.min(warm, x1));
      ctx.fillStyle = C.warm;
      ctx.fillRect(xa, 0, xb - xa, bot);
      hline(ctx, xa, xb, 0, C.line, [3, 4]);
      ctx.fillStyle = C['ink-3'];
      ctx.textAlign = 'left';
      ctx.fillText('LEARNING NORMAL', xa + 10, 16);
    }
    bands(ctx, x0, x1, X, L, R, bot);
    hline(ctx, L, R, bot, C.line);

    if (end > x0) {
      ctx.save();
      ctx.lineJoin = 'round';
      trace(ctx, S.d.vs, x0, end, X, Y);
      ctx.strokeStyle = C['trace-glow'];
      ctx.lineWidth = 4;
      ctx.globalAlpha = .28;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = C.trace;
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.restore();
    }

    for (const a of S.alarms) {
      if (a.i < x0 || a.i >= x1) continue;
      const x = X(a.i), y = Y(S.d.vs[a.i]), real = a.hit !== false;
      ctx.save();
      ctx.strokeStyle = real ? C.beacon : C['ink-3'];
      ctx.globalAlpha = .75;
      ctx.setLineDash([2, 4]);
      if (y - 6 > 14) { ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, y - 9); ctx.stroke(); }
      ctx.restore();
      if (real) {
        ctx.fillStyle = C['beacon-glow'];
        ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = C.beacon;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.fillStyle = C.bg; ctx.strokeStyle = C['ink-2']; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      const age = now - a.born;
      if (age < 1400) ring(ctx, x, y, age, 1400, real ? C.beacon : C['ink-2']);
      if (S.focus && S.focus.i === a.i && now - S.focus.t < 1800) ring(ctx, x, y, (now - S.focus.t) % 900, 900, C.ink);
    }

    if (S.i > x0 && S.i <= x1 && S.i < S.N) {
      const x = X(S.i - 1), y = Y(S.d.vs[S.i - 1]);
      const g = ctx.createLinearGradient(0, 0, 0, bot);
      g.addColorStop(0, 'transparent'); g.addColorStop(.5, C.trace); g.addColorStop(1, 'transparent');
      ctx.save();
      ctx.globalAlpha = .45;
      ctx.fillStyle = g;
      ctx.fillRect(x, 0, 1.5, bot);
      ctx.restore();
      ctx.fillStyle = C.ink;
      ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
    }
    if (S.hover !== null && S.hover >= x0 && S.hover < end) {
      const x = X(S.hover);
      ctx.save();
      ctx.strokeStyle = C['ink-3']; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, bot); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = C.ink;
      ctx.beginPath(); ctx.arc(x, Y(S.d.vs[S.hover]), 4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = C['ink-3'];
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillText(fmtDate(S.d.ts[x0]), L, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(fmtDate(S.d.ts[Math.min(x1, S.N) - 1]), R, h - 4);
  }

  function drawScore() {
    const { ctx, w, h } = SCO;
    ctx.clearRect(0, 0, w, h);
    if (!S.d) return;
    const [x0, x1] = domain(), span = x1 - x0, L = GUT, R = w - 12, top = 8, bot = h - 6;
    const X = i => L + (i - x0) / span * (R - L);
    const Y = s => bot - Math.min(Math.max(s, 0), CAP) / CAP * (bot - top);
    const end = Math.min(S.i, x1), thr = S.eng.threshold;
    bands(ctx, x0, x1, X, L, R, bot);
    ctx.font = `11px ${MONO}`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    const LBL = { 1: '1 in 10', 3: '1 in 1,000', 5: '1 in 100k', 7: '1 in 10M', 9: '1 in 1B' };
    for (const [s, t] of Object.entries(LBL)) {
      hline(ctx, L, R, Y(+s), C['line-2']);
      if (Math.abs(+s - thr) > .6) { ctx.fillStyle = C['ink-3']; ctx.fillText(t, L - 14, Y(+s)); }
    }
    hline(ctx, L, R, Y(thr), C.beacon, [6, 5]);
    ctx.fillStyle = C.beacon;
    ctx.fillText(oneIn(thr).replace('1 in ', '1 in ').replace('100k', '100k'), L - 14, Y(thr));
    hline(ctx, L, R, bot, C.line);
    if (end > x0) {
      ctx.save();
      ctx.lineJoin = 'round';
      trace(ctx, S.scores, x0, end, X, Y);
      ctx.strokeStyle = C.trace;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.beginPath();
      ctx.rect(L, 0, R - L, Y(thr));
      ctx.clip();
      trace(ctx, S.scores, x0, end, X, Y);
      ctx.strokeStyle = C.beacon;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      ctx.restore();
    }
    if (S.hover !== null && S.hover >= x0 && S.hover < end) {
      const x = X(S.hover);
      ctx.save();
      ctx.strokeStyle = C['ink-3']; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, bot); ctx.stroke();
      ctx.restore();
    }
  }

  // ------------------------------------------------------------ panels
  function renderWatchers(init) {
    const g = $('#wgrid');
    if (init) {
      g.innerHTML = WATCHERS.map(([k, nm, d]) => `<div class="w off" data-k="${k}"><h3>${nm}</h3><p>${d}</p><div class="wbar"><i></i><u></u></div><div class="wv">waiting</div></div>`).join('');
    }
    const e = S.eng;
    if (!e) return;
    const names = e.experts.map(x => x.name), nE = Math.max(1, names.length || 4);
    const perThr = e.threshold + Math.log10(nE);
    $('#watch-note').textContent = `orange tick: one watcher alone alarms past ${oneIn(perThr)}`;
    for (const el of g.children) {
      const k = el.dataset.k, j = names.indexOf(k), bar = el.querySelector('i'), v = el.querySelector('.wv');
      el.querySelector('u').style.left = Math.min(100, perThr / CAP * 100) + '%';
      if (e.t < e.p.warmup || !e.lastPs) {
        el.classList.add('off'); bar.style.width = '0';
        v.textContent = S.i ? `learning · ${nf.format(Math.min(S.i, e.p.warmup))} / ${e.p.warmup}` : 'waiting';
        continue;
      }
      if (j < 0) {
        el.classList.add('off'); bar.style.width = '0';
        v.textContent = k === 'seasonal' ? 'off · no daily rhythm found' : 'off';
        continue;
      }
      el.classList.remove('off');
      const s = -Math.log10(Math.max(e.lastPs[j], 1e-300));
      bar.style.width = Math.min(100, s / CAP * 100) + '%';
      bar.classList.toggle('hot', s >= perThr);
      v.textContent = oneIn(s);
    }
  }

  function kpis() {
    const lab = hasLabels(), nA = S.alarms.length;
    const nF = S.alarms.filter(a => a.hit === false).length;
    $('#k-n').textContent = nf.format(S.i);
    $('#k-of').textContent = `of ${nf.format(S.N)}`;
    $('#k-al').textContent = nA;
    $('#feed-n').textContent = nA;
    $('#k-al').classList.toggle('hot', nA > 0);
    $('#k-fa').textContent = !lab ? 'no labels to check' : nA ? (nF ? `${nF} false` : 'all real') : ' ';
    const started = S.meta.windows.filter(([a]) => a < S.i);
    const caught = started.filter(([a, b]) => S.runs.some(r => r[0] <= b && r[1] >= a)).length;
    $('#k-win').textContent = lab && started.length ? `${caught} / ${started.length}` : '–';
    $('#k-win-s').textContent = lab ? (S.meta.windows.length ? 'labelled windows so far' : 'this signal has none') : 'no labels';
    $('#k-prec').textContent = lab && nA ? Math.round((nA - nF) / nA * 100) + '%' : '–';
    $('#k-us').textContent = S.engN > 50 ? (S.engMs * 1000 / S.engN).toFixed(0) : '–';
    $('#prog-fill').style.width = (S.N ? S.i / S.N * 100 : 0) + '%';
  }
  function progressMarks() {
    $('#prog-marks').innerHTML = S.meta.windows.map(([a, b]) =>
      `<i style="left:${(a / S.N * 100).toFixed(3)}%;width:${Math.max(.3, (b - a + 1) / S.N * 100).toFixed(3)}%"></i>`).join('');
  }
  function status() {
    const live = $('#live');
    live.classList.toggle('on', S.playing || S.target !== null);
    let s;
    if (!S.d) s = 'Loading…';
    else if (S.target !== null) s = `<b>Fast-forwarding</b> · ${nf.format(S.i)} / ${nf.format(S.N)}`;
    else if (S.done) s = `<b>Finished</b> · ${esc(S.meta.label)}`;
    else if (S.playing) s = S.i < S.eng.p.warmup ? `<b>Learning normal</b> · ${S.i} / ${S.eng.p.warmup}` : `<b>Streaming</b> · ${esc(S.meta.label)} · ${nf.format(S.i)} / ${nf.format(S.N)}`;
    else s = S.i ? `<b>Paused</b> · ${nf.format(S.i)} / ${nf.format(S.N)}` : 'Press play (or Space) to start the stream';
    $('#status').innerHTML = s;
    $('#play').classList.toggle('on', S.playing);
  }

  function render(now) {
    drawSignal(now);
    drawScore();
    kpis();
    renderWatchers(false);
    for (const a of S.alarms) if (a.run[1] >= S.i - 2) syncHit(a);
    status();
  }

  // ------------------------------------------------------------ loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(100, now - last);
    last = now;
    if (S.d && S.i < S.N) {
      if (S.target !== null) {
        advance(S.target - S.i, 30);
        if (S.i >= S.target) S.target = null;
      } else if (S.playing) {
        S.acc += S.speed * dt / 1000;
        const n = Math.floor(S.acc);
        if (n > 0) { S.acc -= n; advance(n, 14); if (S.speed > 1e6) S.acc = 0; }
      }
    }
    const anim = S.alarms.some(a => now - a.born < 1400) || (S.focus && now - S.focus.t < 1800);
    if (S.dirty || anim) { S.dirty = false; render(now); }
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ list + selection
  function renderList() {
    const q = $('#q').value.trim().toLowerCase();
    const match = m => !q || (m.label + ' ' + m.name).toLowerCase().includes(q);
    const groups = [];
    if (!q) groups.push(['Start here', SHOWCASE.map(n => S.byName.get(n)).filter(Boolean)]);
    for (const c of Object.keys(CATS)) {
      const g = S.manifest.filter(m => m.cat === c && match(m));
      if (g.length) groups.push([CATS[c], g]);
    }
    $('#list').innerHTML = groups.map(([t, g]) => `<div class="grp"><span>${esc(t)}</span><span>${g.length}</span></div>` +
      g.map(m => `<button class="item${S.meta === m ? ' on' : ''}" data-name="${esc(m.name)}"><span class="nm">${esc(m.label)}</span><span class="mt">${compact(m.n)} values · ${m.upload ? 'no labels' : `${m.windows.length} anomal${m.windows.length === 1 ? 'y' : 'ies'}`}</span></button>`).join('')).join('')
      || '<p class="empty" style="margin:10px">No signal matches.</p>';
  }

  async function select(meta, opts = {}) {
    const token = ++S.token;
    S.playing = false;
    S.meta = meta;
    renderList();
    $('#cat').textContent = CATS[meta.cat] || meta.cat;
    $('#title').textContent = meta.label;
    $('#meta').innerHTML = 'Loading…';
    let d;
    try { d = await loadSeries(meta); } catch (err) { toast('Could not load this signal. Serve the site over http (see docs).'); return; }
    if (token !== S.token) return;
    S.d = d;
    S.N = d.vs.length;
    reset();
    const W = meta.windows.length;
    $('#meta').innerHTML = [
      `<span><b>${fmtDate(d.ts[0])}</b> → <b>${fmtDate(d.ts[S.N - 1])}</b></span>`,
      `<span><b>${nf.format(S.N)}</b> values</span>`,
      fmtStep(S.step) && `<span>${fmtStep(S.step)}</span>`,
      meta.upload ? '<span>your file · no labels</span>' : `<span><b>${W}</b> labelled anomal${W === 1 ? 'y' : 'ies'}</span>`,
      meta.upload ? '' : `<span>${esc(meta.name)}</span>`,
    ].filter(Boolean).join('');
    if (!meta.upload) {
      const q = new URLSearchParams({ series: meta.name });
      if (S.view === 'all') q.set('view', 'all');
      history.replaceState(null, '', '?' + q.toString());
    }
    if (opts.end) seek(S.N);
    S.dirty = true;
  }
  function seek(target) { S.playing = false; S.target = Math.min(S.N, target); S.dirty = true; }
  function setView(v) {
    S.view = v;
    for (const b of document.querySelectorAll('#view button')) b.classList.toggle('on', b.dataset.v === v);
    S.dirty = true;
  }
  function togglePlay() {
    if (!S.d) return;
    if (S.done) { reset(); S.playing = true; return; }
    S.target = null;
    S.playing = !S.playing;
    S.dirty = true;
  }

  function readParams() {
    S.params = { risk: +document.querySelector('#risk .on').dataset.v, cooldown: +$('#cool').value, novelty_gate: +$('#gate').value };
    $('#cool-v').textContent = S.params.cooldown;
    $('#gate-v').textContent = S.params.novelty_gate ? S.params.novelty_gate.toFixed(1) : 'off';
    const b = $('#cfg-badge');
    b.textContent = isDefault() ? 'submitted' : 'custom';
    b.classList.toggle('custom', !isDefault());
  }
  function settingsChanged() {
    readParams();
    if (!S.d) return;
    const wasPlaying = S.playing || S.i > 0;
    reset();
    if (wasPlaying) toast('Settings changed, so the stream restarted from the beginning.');
  }

  let toastTimer;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 3800);
  }

  async function upload(file) {
    if (!file) return;
    const d = parseCSV(await file.text());
    const good = Array.from(d.vs).filter(Number.isFinite).length;
    if (good < 400) { toast(`Sentinel needs at least 400 values (it learns from the first 300). This file has ${good}.`); return; }
    const meta = { name: 'upload/' + file.name, label: file.name.replace(/\.csv$/i, ''), cat: 'upload', n: d.vs.length, windows: [], ref: null, upload: true, data: d };
    S.manifest = [meta, ...S.manifest.filter(m => m.name !== meta.name)];
    S.byName.set(meta.name, meta);
    await select(meta);
    S.playing = true;
  }

  // ------------------------------------------------------------ events
  function bindSeg(sel, fn) {
    $(sel).addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      for (const x of $(sel).children) x.classList.toggle('on', x === b);
      fn(b.dataset.v);
    });
  }
  function bind() {
    $('#list').addEventListener('click', e => {
      const b = e.target.closest('.item');
      if (b) select(S.byName.get(b.dataset.name));
    });
    $('#q').addEventListener('input', renderList);
    $('#play').addEventListener('click', togglePlay);
    $('#end').addEventListener('click', () => S.d && seek(S.N));
    $('#reset').addEventListener('click', () => S.d && reset());
    bindSeg('#speed', v => { S.speed = +v; });
    bindSeg('#view', v => setView(v));
    bindSeg('#risk', settingsChanged);
    $('#cool').addEventListener('input', readParams);
    $('#cool').addEventListener('change', settingsChanged);
    $('#gate').addEventListener('input', readParams);
    $('#gate').addEventListener('change', settingsChanged);
    $('#cfg-reset').addEventListener('click', () => {
      for (const b of $('#risk').children) b.classList.toggle('on', b.dataset.v === '1e-5');
      $('#cool').value = 300;
      $('#gate').value = 0;
      settingsChanged();
    });
    $('#progress').addEventListener('click', e => {
      if (!S.d) return;
      const r = $('#progress').getBoundingClientRect();
      const target = Math.round((e.clientX - r.left) / r.width * S.N);
      if (target < S.i) reset();
      seek(target);
    });
    const drop = $('#drop');
    $('#file').addEventListener('change', e => upload(e.target.files[0]));
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); upload(e.dataTransfer.files[0]); });

    const charts = document.querySelector('.charts'), tip = $('#tip');
    const hover = (o, e) => {
      if (!S.d) return;
      const r = o.cv.getBoundingClientRect(), [x0, x1] = domain();
      const i = Math.round(x0 + (e.clientX - r.left - GUT) / (r.width - GUT - 12) * (x1 - x0));
      if (i < x0 || i >= Math.min(S.i, x1)) { S.hover = null; tip.hidden = true; S.dirty = true; return; }
      S.hover = i;
      const s = S.scores[i], an = s >= S.eng.threshold;
      tip.innerHTML = `<b>${esc(fmtTs(S.d.ts[i]))}</b><br>value <b>${compact(S.d.vs[i])}</b><br>rarity <b class="${an ? 'hot' : ''}">${i < S.eng.p.warmup ? 'learning' : oneIn(s)}</b>`;
      tip.hidden = false;
      const cr = charts.getBoundingClientRect();
      let x = e.clientX - cr.left + 16, y = e.clientY - cr.top + 16;
      if (x + 210 > cr.width) x -= 236;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
      S.dirty = true;
    };
    for (const o of [SIG, SCO]) {
      o.cv.addEventListener('mousemove', e => hover(o, e));
      o.cv.addEventListener('mouseleave', () => { S.hover = null; tip.hidden = true; S.dirty = true; });
    }

    $('#theme').addEventListener('click', toggleTheme);
    document.addEventListener('keydown', e => {
      if (e.target.closest('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === ' ') { e.preventDefault(); togglePlay(); }
      else if (k === 'e') S.d && seek(S.N);
      else if (k === 'r') S.d && reset();
      else if (k === 'v') setView(S.view === 'all' ? 'follow' : 'all');
      else if (k === 't') toggleTheme();
    });
  }
  function toggleTheme() {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    try { localStorage.setItem('sentinel-theme', root.dataset.theme); } catch (e) { /* storage blocked */ }
    readColors();
  }

  // ------------------------------------------------------------ start
  async function init() {
    try { const t = localStorage.getItem('sentinel-theme'); if (t) root.dataset.theme = t; } catch (e) { /* storage blocked */ }
    if (query.get('theme')) root.dataset.theme = query.get('theme');
    readColors();
    SIG = setupCanvas('#cv-sig');
    SCO = setupCanvas('#cv-score');
    bind();
    renderWatchers(true);
    if (document.fonts) document.fonts.ready.then(() => { S.dirty = true; });
    let man;
    try { man = await (await fetch('data/manifest.json')).json(); }
    catch (err) {
      $('#title').textContent = 'Open this page through a web server';
      $('#meta').textContent = 'Browsers block data files on file:// pages. Run: python -m http.server --directory web, then open http://localhost:8000/demo/';
      return;
    }
    S.manifest = man.series.map(s => ({ ...s, cat: s.name.split('/')[0], label: pretty(s.name) }));
    S.byName = new Map(S.manifest.map(m => [m.name, m]));
    if (query.get('view') === 'all') setView('all');
    requestAnimationFrame(frame);
    await select(S.byName.get(query.get('series')) || S.byName.get(SHOWCASE[0]), { end: query.get('end') === '1' });
    if (query.has('play')) S.playing = true;
  }
  init();
})();
