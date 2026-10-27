# The live web demo (`web/demo/`)

The Streamlit demo (`app.py`), rebuilt as a web page that can go on Vercel.
It is **really live**: the detector runs inside the browser, one value at a time. Nothing is
pre-recorded.

## How it works, in plain words

1. We rewrote the detector (`sentinel/*.py`, about 660 lines) in JavaScript: `web/demo/sentinel.js`.
   Same steps, same numbers, even the same way of adding numbers up (numpy's method), so the
   results match to the last digit.
2. The 58 benchmark signals ship with the site as their original CSV files (`web/demo/data/`).
3. The page feeds a signal into the detector point by point and draws what happens.

**Proof it is the same detector:** `node scripts/check_web_engine.js` runs all 58 signals
through the JavaScript version and compares every point with `predictions/predictions.csv`:

```
58 series, 365,558 points, 106 alarms
max |score diff| 5.00e-4 · label mismatches 0 · series with different alarms 0
PASS: identical to predictions.csv
```

(5e-4 is only the rounding in the CSV, which stores 3 decimals.) The page also checks this
itself: when a stream finishes with the submitted settings, it shows
**"Identical to our submitted predictions"**.

## What you can do on the page

| Part | What it shows |
|---|---|
| Signal list (left) | all 58 signals by category; "Start here" has the best examples; search box |
| Play / speed / Jump to end / Reset | stream the signal; click the progress bar to fast-forward |
| Counters | values streamed, alarms, anomalies caught, precision so far, µs per value |
| The signal | the line drawn live; grey bands = labelled anomalies; orange dot = alarm that was real; hollow dot = false alarm |
| How rare each value is | the score as "1 in N"; crossing the dashed line means an alarm |
| Five watchers | what each watcher sees for the latest value, live |
| Alarms and evidence (right) | every alarm with its reason, real or false, per-watcher rarity, and the raw JSON |
| Detector settings | risk, quiet period, novelty gate (changing them restarts the stream) |
| Your own data | drop any CSV with `timestamp,value` columns (at least 400 values) |

Keys: **Space** play/pause · **E** jump to end · **R** reset · **V** follow / whole series · **T** light/dark.

Share a signal with a link: `demo/?series=realTweets/Twitter_volume_CVS.csv` (add `&end=1`
to open it already finished, `&play` to start playing).

## Run it on your computer

Browsers block data files on `file://` pages, so serve the `web` folder:

```bash
python -m http.server 8080 --directory web
```

Then open http://localhost:8080/demo/ (the deck is at http://localhost:8080/).

## Put it online (Vercel)

Same project as the deck, no settings to change: `vercel.json` serves the `web` folder with no build step and
no environment variables. Open it as `/demo/` with the trailing slash (Vercel adds it automatically).
The deck is at `/`, the demo at `/demo/`. Size is about 10 MB, mostly the CSVs.

## When the data or detector changes

```bash
python scripts/web_demo_data.py       # copies the CSVs, writes data/manifest.json
node scripts/check_web_engine.js      # must print PASS after any change to sentinel.js
```

If you change the Python detector, make the same change in `web/demo/sentinel.js` and re-run
both commands.

The data is the Numenta Anomaly Benchmark (AGPL-3.0, license in `web/demo/data/NAB_LICENSE.txt`,
credited in the page footer).
