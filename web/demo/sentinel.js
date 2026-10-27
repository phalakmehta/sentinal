/* Sentinel detector in JavaScript: a line-for-line port of sentinel/engine.py, experts.py,
 * threshold.py and evidence.py.
 *
 * Arithmetic follows the Python original exactly: numpy's pairwise sums and linear quantiles,
 * Python 3.12's compensated sum(), round-half-even. scripts/check_web_engine.js replays all 58
 * series and compares every score and label with predictions/predictions.csv.
 *
 *   const sen = new Sentinel(defaultParams(), stepSeconds);
 *   const out = sen.update(timestampString, value);   // out.label: "NORMAL" | "ANOMALOUS"
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------ numpy / Python arithmetic
  function pairwise(a, lo, n) {            // numpy's pairwise summation (PW_BLOCKSIZE 128)
    if (n < 8) {
      let r = 0;
      for (let i = 0; i < n; i++) r += a[lo + i];
      return r;
    }
    if (n <= 128) {
      const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
      let i = 8;
      for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[lo + i];
      return res;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
  }
  function npSum(a) {                       // np.sum: pairwise within 8192-element buffers
    let acc = 0;
    for (let i = 0; i < a.length; i += 8192) acc += pairwise(a, i, Math.min(8192, a.length - i));
    return acc;
  }
  const npMean = a => npSum(a) / a.length;
  function sortedCopy(a) { return Float64Array.from(a).sort(); }
  function npMedian(a) {
    const s = sortedCopy(a), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function npQuantile(a, q) {               // method="linear", numpy's _lerp
    const s = sortedCopy(a), n = s.length;
    const v = (n - 1) * q;
    if (v >= n - 1) return s[n - 1];
    if (v < 0) return s[0];
    const lo = Math.floor(v), g = v - lo, A = s[lo], B = s[lo + 1], d = B - A;
    return g >= 0.5 ? B - d * (1 - g) : A + d * g;
  }
  function pySum(a) {                       // Python 3.12 sum(): Neumaier compensation
    if (!a.length) return 0;
    let f = a[0], c = 0;
    for (let i = 1; i < a.length; i++) {
      const x = a[i], t = f + x;
      c += Math.abs(f) >= Math.abs(x) ? (f - t) + x : (x - t) + f;
      f = t;
    }
    return c && Number.isFinite(c) ? f + c : f;
  }
  function pyRound(x) {                     // round half to even
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r;
  }
  function pyMedian(v) {                    // experts._median
    const s = v.slice().sort((p, q) => p - q), n = s.length;
    return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
  }
  function mad(v, med) {                    // experts._mad
    const dev = v.map(a => Math.abs(a - med));
    const m = pyMedian(dev);
    return m > 0 ? m : pySum(dev) / dev.length;
  }

  // ------------------------------------------------------------ threshold.py
  class TailModel {
    constructor(scores, initQ = 0.98, shrink = 0) {
      this.shrink = shrink;
      this.t = npQuantile(scores, initQ);
      const exc = [];
      for (const s of scores) if (s > this.t) exc.push(s - this.t);
      this.n = scores.length;
      this.nt = exc.length;
      this.s1 = npSum(exc);
      this.s2 = npSum(exc.map(e => e * e));
      this._fit();
    }
    _fit() {
      if (this.nt < 3) {
        this.gamma = 0;
        this.sigma = Math.max(this.s1 / Math.max(this.nt, 1), 1e-9);
        return;
      }
      const mean = this.s1 / this.nt;
      const vr = Math.max(this.s2 / this.nt - mean * mean, 1e-12);
      const r = mean * mean / vr;
      this.gamma = Math.min(Math.max(0.5 * (1 - r), 0), 0.8);
      this.sigma = Math.max(0.5 * mean * (r + 1), 1e-9);
      if (this.shrink > 0 && this.gamma > 0) {
        this.gamma *= this.nt / (this.nt + this.shrink);
        this.sigma = Math.max(mean * (1 - this.gamma), 1e-9);
      }
    }
    tailProb(s) {
      const base = (this.nt + 1) / (this.n + 1);
      if (s <= this.t) {
        if (this.t <= 0) return 1;
        const frac = this.t > 0 ? s / this.t : 1;
        return Math.min(1, base + (1 - base) * (1 - Math.max(frac, 0)));
      }
      const y = (s - this.t) / this.sigma;
      if (Math.abs(this.gamma) < 1e-6) return base * Math.exp(-y);
      const inner = 1 + this.gamma * y;
      if (inner <= 0) return 1e-300;
      return base * inner ** (-1 / this.gamma);
    }
    levelFor(p) {
      const base = (this.nt + 1) / (this.n + 1);
      if (p >= base) return this.t;
      const r = p / base;
      if (Math.abs(this.gamma) < 1e-6) return this.t - this.sigma * Math.log(r);
      return this.t + this.sigma / this.gamma * (r ** (-this.gamma) - 1);
    }
    update(s, cap = null) {
      this.n += 1;
      if (s > this.t) {
        if (cap !== null && s > cap) s = cap;
        const e = s - this.t;
        this.nt += 1;
        this.s1 += e;
        this.s2 += e * e;
        this._fit();
      }
    }
  }

  // ------------------------------------------------------------ experts.py
  class RobustEW {
    constructor(alpha, clip = 3, floor = 1e-9) { this.alpha = alpha; this.clip = clip; this.floor = floor; this.m = 0; this.s = 0; }
    init(m, s) { this.m = m; this.s = s; }
    get scale() { return this.s > this.floor ? this.s : this.floor; }
    z(x) { return (x - this.m) / this.scale; }
    update(x) {
      const lim = this.clip * this.scale, d = x - this.m, a = Math.abs(d);
      this.m += this.alpha * (d > lim ? lim : (d < -lim ? -lim : d));
      this.s += this.alpha * ((a < 2 * lim ? a : 2 * lim) - this.s);
    }
  }

  class LevelExpert {
    constructor(alpha, floor) { this.name = 'level'; this.ew = new RobustEW(alpha, 3, floor); }
    init(buf) { const med = pyMedian(buf); this.ew.init(med, mad(buf, med)); }
    score(x) { return this.ew.z(x); }
    update(x) { this.ew.update(x); }
    describe() { return { expected: this.ew.m, band: [this.ew.m - 3 * this.ew.scale, this.ew.m + 3 * this.ew.scale] }; }
  }

  class DiffExpert {
    constructor(alpha, floor) { this.name = 'jump'; this.ew = new RobustEW(alpha, 3, floor); this.prev = null; }
    init(buf) {
      const d = [];
      for (let i = 1; i < buf.length; i++) d.push(buf[i] - buf[i - 1]);
      const med = pyMedian(d);
      this.ew.init(med, mad(d, med));
      this.prev = buf[buf.length - 1];
    }
    score(x) { return this.ew.z(x - this.prev); }
    update(x) { this.ew.update(x - this.prev); this.prev = x; }
    describe(x) { return { step: x - this.prev, typical_step: this.ew.scale }; }
  }

  class SeasonalExpert {
    constructor(period, alpha, floor) { this.name = 'seasonal'; this.period = period; this.ew = new RobustEW(alpha, 3, floor); this.ring = []; this.pos = 0; }
    init(buf) {
      const p = this.period, r = [];
      for (let i = p; i < buf.length; i++) r.push(buf[i] - buf[i - p]);
      const med = pyMedian(r);
      this.ew.init(med, mad(r, med));
      this.ring = buf.slice(-p);
      this.pos = 0;
    }
    lagged() { return this.ring[this.pos]; }
    score(x) { return this.ew.z(x - this.lagged()); }
    update(x) { this.ew.update(x - this.lagged()); this.ring[this.pos] = x; this.pos = (this.pos + 1) % this.period; }
    describe() { return { same_time_last_period: this.lagged(), period_pts: this.period }; }
  }

  class ShiftExpert {
    constructor(alphaFast, alphaSlow, floor) { this.name = 'shift'; this.af = alphaFast; this.slow = new RobustEW(alphaSlow, 3, floor); this.fast = 0; }
    init(buf) { const med = pyMedian(buf); this.slow.init(med, mad(buf, med)); this.fast = med; }
    score(x) { const f = this.fast + this.af * (x - this.fast); return (f - this.slow.m) / this.slow.scale; }
    update(x) { this.fast += this.af * (x - this.fast); this.slow.update(x); }
    describe(x) { return { recent_mean: this.fast + this.af * (x - this.fast), baseline_mean: this.slow.m }; }
  }

  class VolatilityExpert {
    constructor(alphaFast, alphaSlow, floor) { this.name = 'volatility'; this.af = alphaFast; this.asl = alphaSlow; this.floor = floor; this.fast = this.slow = floor; this.prev = null; }
    init(buf) {
      const d = [];
      for (let i = 1; i < buf.length; i++) d.push(Math.abs(buf[i] - buf[i - 1]));
      const m = Math.max(pySum(d) / d.length, this.floor);
      this.fast = this.slow = m;
      this.prev = buf[buf.length - 1];
    }
    fastNext(x) { return this.fast + this.af * (Math.abs(x - this.prev) - this.fast); }
    score(x) { return Math.log(Math.max(this.fastNext(x), this.floor) / Math.max(this.slow, this.floor)); }
    update(x) {
      const a = Math.abs(x - this.prev);
      this.fast += this.af * (a - this.fast);
      const lim = 5 * this.slow;
      this.slow += this.asl * (Math.min(a, lim) - this.slow);
      this.prev = x;
    }
    describe(x) { return { recent_volatility: this.fastNext(x), baseline_volatility: this.slow }; }
  }

  class DiscordExpert {
    constructor(m, horizon) {
      this.name = 'novelty'; this.m = m; this.h = horizon;
      this.cap = 2 * (horizon + 2 * m);
      this.buf = new Float64Array(this.cap);
      this.n = 0;
      this.last = [0, -1];
    }
    init(buf) { for (const x of buf) this.update(x); }
    nn(x) {
      const m = this.m, n = this.n, buf = this.buf;
      if (n < 3 * m) return [0, -1];
      const lo = Math.max(0, n - this.h - m), q0 = n - m + 1;
      let best = Infinity, arg = -1;
      for (let s = lo; s < n - 2 * m + 2; s++) {
        let acc = 0;
        for (let k = 0; k < m; k++) {
          const d = buf[s + k] - (k === m - 1 ? x : buf[q0 + k]);
          acc += d * d;
          if (acc >= best) break;
        }
        if (acc < best) { best = acc; arg = s; }
      }
      return [Math.sqrt(best / m), arg];
    }
    score(x) { this.last = this.nn(x); return this.last[0]; }
    update(x) {
      if (this.n === this.cap) {
        const keep = this.h + 2 * this.m;
        this.buf.copyWithin(0, this.n - keep, this.n);
        this.n = keep;
      }
      this.buf[this.n++] = x;
    }
    describe() {
      const [d, j] = this.last;
      return { nn_distance: d, nn_age: j >= 0 ? this.n - this.m + 1 - j : null, window: this.m };
    }
  }

  // ------------------------------------------------------------ evidence.py
  function fmtG3(v) {                        // Python f"{v:.3g}"
    if (v === 0) return '0';
    const [mant, ex] = v.toExponential(2).split('e');
    const e = +ex;
    const strip = s => (s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s);
    if (e < -4 || e >= 3) return strip(mant) + 'e' + (e < 0 ? '-' : '+') + String(Math.abs(e)).padStart(2, '0');
    return strip(v.toFixed(Math.max(0, 2 - e)));
  }
  function f(v) {
    const a = Math.abs(v);
    if (a >= 1e5 || (a > 0 && a < 1e-2)) return fmtG3(v);
    return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function explain(expert, x, z, info) {
    const side = z > 0 ? 'above' : 'below';
    if (expert === 'level') {
      const [lo, hi] = info.band;
      return `Value ${f(x)} is ${Math.abs(z).toFixed(1)} scales ${side} the learned baseline ${f(info.expected)} (normal band ${f(lo)} to ${f(hi)}).`;
    }
    if (expert === 'jump') {
      const ratio = Math.abs(info.step) / Math.max(info.typical_step, 1e-12);
      return `Sudden ${info.step > 0 ? 'rise' : 'drop'} of ${f(Math.abs(info.step))} in one step, ${ratio.toFixed(1)}x the typical step size.`;
    }
    if (expert === 'seasonal') {
      return `Value ${f(x)} breaks the repeating pattern: the same point one period earlier (${info.period_pts} steps) was ${f(info.same_time_last_period)}.`;
    }
    if (expert === 'shift') {
      return `Sustained level shift: recent mean ${f(info.recent_mean)} vs baseline ${f(info.baseline_mean)} (${Math.abs(z).toFixed(1)} scales ${side}).`;
    }
    if (expert === 'volatility') {
      const ratio = info.recent_volatility / Math.max(info.baseline_volatility, 1e-12);
      if (ratio >= 1000) return 'Noise burst: step-to-step volatility jumped over 1,000x from a nearly flat baseline.';
      if (ratio >= 1) return `Noise burst: step-to-step volatility is ${ratio.toFixed(1)}x its baseline.`;
      return `Signal went unusually flat: volatility is ${ratio.toFixed(2)}x its baseline.`;
    }
    if (expert === 'novelty') {
      const when = info.nn_age !== null ? `${info.nn_age} steps ago` : 'in memory';
      return `The last ${info.window} points form a pattern not seen before: the closest past match (${when}) differs by ${f(info.nn_distance)} per point.`;
    }
    return `Score exceeded the adaptive threshold (expert ${expert}, z=${z.toFixed(1)}).`;
  }

  // ------------------------------------------------------------ engine.py
  function defaultParams() {
    return {
      warmup: 300, risk: 1e-5, hysteresis: 1.0, max_event_len: 30, cooldown: 300, init_q: 0.95,
      tail_shrink: 200.0, alpha_level: 0.02, alpha_diff: 0.02, alpha_seasonal: 0.02, alpha_fast: 0.1,
      alpha_slow: 0.005, novelty_gate: 0.0, cusum_k: 2.0, relearn: true, acf_min: 0.4, discord_m: 8,
      discord_h: 2000, experts: ['level', 'seasonal', 'shift', 'volatility', 'novelty'], history: 6000,
    };
  }

  function acf(x, lag) {
    const mu = npMean(x), c = x.map(v => v - mu);
    const den = npSum(c.map(v => v * v));
    if (!(den > 0)) return 0;
    const prod = new Array(c.length - lag);
    for (let i = 0; i < prod.length; i++) prod[i] = c[i + lag] * c[i];
    return npSum(prod) / den;
  }

  const round2 = v => pyRound(v * 100) / 100;

  class Sentinel {
    constructor(params, stepSeconds = 300) {
      this.p = Object.assign(defaultParams(), params || {});
      this.stepSeconds = stepSeconds;
      this.threshold = -Math.log10(this.p.risk);
      this.hist = [];
      this.t = 0;
      this.experts = [];
      this.tails = [];
      this.inEvent = false;
      this.eventLen = 0;
      this.acc = 0;
      this.novIdx = null;
      this.lastOnset = -1e9;
      this.events = [];
      this.relearns = [];
      this.lastZs = null;
      this.lastPs = null;
    }

    _floor(buf) {
      let q = Infinity;
      for (let i = 1; i < buf.length; i++) { const d = Math.abs(buf[i] - buf[i - 1]); if (d > 0 && d < q) q = d; }
      const quantum = q === Infinity ? 0 : q;
      return Math.max(0.5 * quantum, 1e-6 * npMedian(buf.map(Math.abs)), 1e-9);
    }
    _make(name, floor, period) {
      const p = this.p;
      if (name === 'level') return new LevelExpert(p.alpha_level, floor);
      if (name === 'jump') return new DiffExpert(p.alpha_diff, floor);
      if (name === 'seasonal') return new SeasonalExpert(period, p.alpha_seasonal, floor);
      if (name === 'shift') return new ShiftExpert(p.alpha_fast, p.alpha_slow, floor);
      if (name === 'volatility') return new VolatilityExpert(p.alpha_fast, p.alpha_slow, floor);
      if (name === 'novelty') return new DiscordExpert(p.discord_m, p.discord_h);
      throw new Error(name);
    }
    _calibrate(expert, buf) {
      buf = buf.slice();
      const k = Math.max(Math.floor(buf.length / 4), (expert.period || 0) + 10);
      expert.init(buf.slice(0, k));
      const scores = [];
      for (let i = k; i < buf.length; i++) { scores.push(Math.abs(expert.score(buf[i]))); expert.update(buf[i]); }
      return new TailModel(scores, this.p.init_q, this.p.tail_shrink);
    }
    _periods() {
      const day = pyRound(86400 / this.stepSeconds);
      return day >= 4 ? [day, 7 * day] : [];
    }
    _trySeasonal() {
      if (!this.p.experts.includes('seasonal') || this.experts.some(e => e.name === 'seasonal')) return;
      for (const period of this._periods()) {
        if (this.hist.length >= 3 * period && acf(this.hist, period) >= this.p.acf_min) {
          const e = this._make('seasonal', this.floor, period);
          this.tails.push(this._calibrate(e, this.hist));
          this.experts.push(e);
          return;
        }
      }
    }
    _setup() {
      this.floor = this._floor(this.hist);
      for (const name of this.p.experts) {
        if (name === 'seasonal') continue;
        const e = this._make(name, this.floor);
        this.tails.push(this._calibrate(e, this.hist));
        this.experts.push(e);
      }
      const names = this.experts.map(e => e.name);
      this.novIdx = names.includes('novelty') ? names.indexOf('novelty') : null;
      this._trySeasonal();
    }
    _relearn(ts) {
      const recent = this.hist.slice(-this.p.warmup);
      this.floor = this._floor(recent);
      this.experts.forEach((e, j) => {
        const buf = e.name === 'novelty' || e.name === 'seasonal' ? this.hist : recent;
        const ne = this._make(e.name, this.floor, e.period);
        this.tails[j] = this._calibrate(ne, buf);
        this.experts[j] = ne;
      });
      this.acc = 0;
      this.lastOnset = this.t;
      this.relearns.push({ t: this.t, timestamp: String(ts) });
    }
    _push(x) {
      this.hist.push(x);
      if (this.hist.length > this.p.history) this.hist.shift();
    }

    update(ts, x) {
      this.t += 1;
      if (x !== x) return { score: 0, threshold: this.threshold, label: 'NORMAL', expert: null, evidence: null };
      if (this.t <= this.p.warmup) {
        this._push(x);
        if (this.t === this.p.warmup) this._setup();
        return { score: 0, threshold: this.threshold, label: 'NORMAL', expert: null, evidence: null };
      }
      const zs = this.experts.map(e => e.score(x));
      const ps = this.tails.map((tl, i) => tl.tailProb(Math.abs(zs[i])));
      this.lastZs = zs;
      this.lastPs = ps;
      let k = 0;
      for (let i = 1; i < ps.length; i++) if (ps[i] < ps[k]) k = i;
      const pComb = Math.min(1, ps[k] * ps.length);
      let pointScore = -Math.log10(Math.max(pComb, 1e-300));
      if (this.p.novelty_gate > 0 && this.novIdx !== null) {
        const nov = -Math.log10(Math.max(ps[this.novIdx], 1e-300));
        if (nov < this.p.novelty_gate) pointScore = Math.min(pointScore, nov);
      }
      let score;
      if (this.p.cusum_k > 0) {
        this.acc = Math.max(0, this.acc + pointScore - this.p.cusum_k);
        score = Math.max(pointScore, this.acc);
      } else score = pointScore;

      let onset = false, relearn = false;
      if (this.inEvent) {
        this.eventLen += 1;
        if (score < this.threshold - this.p.hysteresis || this.eventLen >= this.p.max_event_len) this.inEvent = false;
      } else if (score >= this.threshold && this.t - this.lastOnset > this.p.cooldown) {
        if (this.p.relearn && this.t - this.lastOnset === this.p.cooldown + 1) relearn = true;
        else { this.inEvent = true; this.eventLen = 0; this.lastOnset = this.t; onset = true; this.acc = 0; }
      }

      const out = {
        score, threshold: this.threshold, label: this.inEvent ? 'ANOMALOUS' : 'NORMAL',
        expert: score >= this.threshold ? this.experts[k].name : null, evidence: null, relearn,
      };
      if (onset) {
        out.evidence = this._evidence(ts, x, zs, ps, k, score);
        this.events.push(out.evidence);
      }

      const pAlarm = this.p.risk / this.experts.length;
      this.experts.forEach((e, i) => {
        const tl = this.tails[i], a = Math.abs(zs[i]);
        tl.update(a, a > tl.t ? tl.levelFor(pAlarm) : null);
        e.update(x);
      });

      this._push(x);
      if (relearn) this._relearn(ts);
      if (this.experts.length < this.p.experts.length && this.t % 500 === 0) this._trySeasonal();
      return out;
    }

    _evidence(ts, x, zs, ps, k, score) {
      const detail = {};
      this.experts.forEach((e, i) => { detail[e.name] = { z: round2(zs[i]), p: +ps[i].toPrecision(3) }; });
      const top = this.experts[k], info = top.describe(x);
      return {
        timestamp: String(ts), value: x, score: round2(score), threshold: round2(this.threshold),
        expert: top.name, z: round2(zs[k]), experts: detail, info, explanation: explain(top.name, x, zs[k], info),
      };
    }
  }

  // median step in seconds, as run_series / predict.py compute it
  function stepSeconds(tsSeconds) {
    if (tsSeconds.length < 2) return 300;
    const d = [];
    for (let i = 1; i < tsSeconds.length; i++) d.push(tsSeconds[i] - tsSeconds[i - 1]);
    return npMedian(d);
  }

  const api = { Sentinel, TailModel, defaultParams, stepSeconds, explain, _np: { npSum, npQuantile, npMedian, pySum, pyRound } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SentinelEngine = api;
})(typeof self !== 'undefined' ? self : this);
