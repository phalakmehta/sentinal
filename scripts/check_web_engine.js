// Replays all 58 NAB series through the JavaScript port (web/demo/sentinel.js) and compares
// every point with predictions/predictions.csv (the submitted Python output).
//
//   node scripts/check_web_engine.js
//
// Pass: every label and every alarm identical, every score within the CSV's rounding (0.0005).
const fs = require('fs');
const path = require('path');
const { Sentinel, stepSeconds } = require('../web/demo/sentinel.js');

const ROOT = path.join(__dirname, '..');
const DATA = path.join(ROOT, 'web', 'demo', 'data');
const manifest = JSON.parse(fs.readFileSync(path.join(DATA, 'manifest.json'), 'utf8'));

function readSeries(name) {
  const lines = fs.readFileSync(path.join(DATA, 'series', name), 'utf8').split(/\r?\n/).filter(Boolean).slice(1);
  const ts = [], vs = [];
  for (const l of lines) {
    const c = l.indexOf(',');
    ts.push(l.slice(0, c));
    vs.push(parseFloat(l.slice(c + 1)));
  }
  return { ts, vs, secs: ts.map(t => Date.parse(t.replace(' ', 'T') + 'Z') / 1000) };
}

// predictions.csv: series,timestamp,value,anomaly_score,score,threshold,label,...
const ref = new Map();
for (const l of fs.readFileSync(path.join(ROOT, 'predictions', 'predictions.csv'), 'utf8').split(/\r?\n/).slice(1)) {
  if (!l) continue;
  const f = [];
  let i = 0;
  for (let k = 0; k < 7; k++) { const j = l.indexOf(',', i); f.push(l.slice(i, j)); i = j + 1; }
  if (!ref.has(f[0])) ref.set(f[0], { score: [], label: [] });
  ref.get(f[0]).score.push(+f[4]);
  ref.get(f[0]).label.push(f[6]);
}

let points = 0, maxDiff = 0, labelBad = 0, alarmBad = 0, alarms = 0, ms = 0;
for (const s of manifest.series) {
  const { ts, vs, secs } = readSeries(s.name);
  const r = ref.get(s.name);
  const sen = new Sentinel({}, stepSeconds(secs));
  const t0 = performance.now();
  const onsets = [];
  for (let i = 0; i < vs.length; i++) {
    const o = sen.update(ts[i], vs[i]);
    const d = Math.abs(o.score - r.score[i]);
    if (d > maxDiff) maxDiff = d;
    if (o.label !== r.label[i]) labelBad++;
    if (o.evidence) onsets.push(i);
  }
  ms += performance.now() - t0;
  points += vs.length;
  alarms += onsets.length;
  if (onsets.join() !== s.ref.alarms.join()) { alarmBad++; console.log('alarm mismatch', s.name, onsets, s.ref.alarms); }
}
const ok = labelBad === 0 && alarmBad === 0 && maxDiff <= 0.0005 + 1e-9;
console.log(`${manifest.series.length} series, ${points.toLocaleString('en-US')} points, ${alarms} alarms`);
console.log(`max |score diff| ${maxDiff.toExponential(2)} · label mismatches ${labelBad} · series with different alarms ${alarmBad}`);
console.log(`JS speed: ${(ms * 1000 / points).toFixed(1)} µs per point`);
console.log(ok ? 'PASS: identical to predictions.csv' : 'FAIL');
process.exit(ok ? 0 : 1);
