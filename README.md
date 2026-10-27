# Sentinel — adaptive streaming anomaly detection

**Utkrishta 2026 · Challenge 2 (Sentinel).** Sentinel learns each series' normal behaviour from the stream itself, scores every new point with five cheap *experts*, calibrates each expert's tail with **online extreme-value theory** (with small-sample shape shrinkage), re-learns its baseline when the normal itself changes, and raises sparse, explained alarms. Every flag carries an anomaly score, a NORMAL/ANOMALOUS label and plain-English evidence.

| | Event-F1 | Precision | Recall | False alarms | Latency (p50) |
|---|---|---|---|---|---|
| **Sentinel** | **0.594** | **0.632** | 0.560 | **39** | **61 µs / point** |
| Temporal clustering (k-means) | 0.316 | 0.224 | 0.534 | 249 | 6 µs |
| Ensemble of all 6 baselines (mean) | 0.280 | 0.178 | 0.647 | 419 | 221 µs |
| Autoencoder (16-8-4-8-16) | 0.272 | 0.176 | 0.603 | 390 | 140 µs |
| Local Outlier Factor | 0.260 | 0.165 | 0.621 | 457 | 13 µs |
| Isolation Forest | 0.249 | 0.177 | 0.422 | 261 | 54 µs |
| One-Class SVM | 0.235 | 0.142 | 0.681 | 594 | 8 µs |
| Rolling z-score | 0.226 | 0.148 | 0.474 | 351 | 0.5 µs |

All 58 NAB series, 116 labelled windows. The baselines cover every model family in the problem statement (`python scripts/run_baselines.py`): each learns from the first 300 values of every series and was given its *best* global threshold (oracle-tuned); the ensemble combines the six standardised scores (mean beat max, so mean is reported); Sentinel uses one global configuration, chosen under a **15-run experiment budget** (7 runs used, see [docs/02](docs/02_experiment_budget.md)). Sentinel is the **best of all 8 methods: 1.9× the runner-up** (temporal clustering) and **2.3× the best of the four standard methods**. The leave-one-category-out score over the budgeted runs is also 0.594, because the same run wins for every held-out category; since the hypotheses were designed after seeing all labels, expect the hidden-test score to be somewhat lower (in phase 1 the gap was about 0.02).

**Plain-language notes** on the data, the EDA, every experiment and the two improvements are in [`docs/`](docs/README.md).

## Submission at a glance

Team **if/elif/else**: Aarohi Gattewar, Phalak Mehta, Harshit Singh.

| Deliverable | Where |
|---|---|
| Source code | [`sentinel/`](sentinel/) (the detector), [`predict.py`](predict.py) (runs it on a folder of CSVs), [`scripts/`](scripts/) (EDA, experiments, figures) |
| README documentation | this file, plus plain-language notes in [`docs/`](docs/README.md) |
| Prediction results file | [`predictions/predictions.csv`](predictions/predictions.csv): one row per point, all 58 series (365,558 rows); alarms with evidence in [`predictions/events.json`](predictions/events.json); scores in [`predictions/metrics.json`](predictions/metrics.json) |
| Presentation | [`presentation/Sentinel.pdf`](presentation/Sentinel.pdf) and [`presentation/Sentinel.pptx`](presentation/Sentinel.pptx): 10 LaTeX Beamer slides (source `deck/sentinel.tex`), written to be read on their own; every chart has a "How to read" note. Backup Q&A slides: `presentation/Sentinel_backup.pdf` |
| Deployed project | [`web/`](web/) is a static site: the deck at `/` and the live demo at `/demo/`, where the detector runs in the browser ([docs/05](docs/05_web_demo.md)). URL: *to be added after deployment* |

---

## Quick start

```bash
pip install -r requirements.txt      # Python 3.12

python predict.py                    # run on data/nab -> predictions/
streamlit run app.py                 # Python demo at http://localhost:8501
python -m http.server 8080 --directory web   # web deck at http://localhost:8080, live demo at /demo/
```

Run on any other folder of CSVs (`timestamp,value` columns, any nesting):

```bash
python predict.py --data path/to/csvs --labels windows.json --out predictions
python predict.py --data path/to/csvs --no-labels            # no ground truth
```

`--labels` takes NAB's `combined_windows.json` format: `{"<relative csv path>": [[start, end], ...]}`.

## Outputs (`predictions/`)

| File | Contents |
|---|---|
| `predictions.csv` | One row per point: `series, timestamp, value, anomaly_score` (0–1, >0.5 means past the threshold), `score` (−log₁₀ p), `threshold`, `label` (NORMAL/ANOMALOUS), `expert`, `event_id`, `explanation` |
| `events.json` | One record per alarm: timestamp, value, score, threshold, the expert that fired, every expert's z-score and tail probability, and a sentence such as *"The last 8 points form a pattern not seen before: the closest past match (1451 steps ago) differs by 4.95 per point."* |
| `metrics.json` | Event-F1, precision, recall and detection delay, overall and per category |
| `runtime.json` | Per-point streaming latency (p50 / p99 / mean) and throughput |

## How it works

```
value x_t ──► 5 streaming experts ──► EVT tail model per expert ──► fusion ──► CUSUM ──► alarm logic ──► label + evidence
               (O(1) state each)       p_e = P(score ≥ s)            min p·E      evidence     cooldown,
                                                                                  accumulation  hysteresis
```

1. **Warm-up (300 points):** learn a robust baseline (median / MAD) for each expert. No alarms are raised. The warm-up is a fixed number of points, so it never needs to know the series length.
2. **Experts.** Each one scores the new point *before* learning from it, so the system is strictly causal:

   | Expert | Detects | Signal |
   |---|---|---|
   | `level` | point outliers | robust z-score vs. a clipped exponentially weighted (EW) baseline |
   | `seasonal` | broken daily/weekly patterns | residual vs. one period ago; the period is auto-detected by autocorrelation |
   | `shift` | sustained level shifts | fast EW mean vs. slow robust baseline |
   | `volatility` | noise bursts or flat-lining | log ratio of fast vs. slow step volatility |
   | `novelty` | never-before-seen shapes | **streaming left matrix profile**: distance from the last 8 points to their nearest neighbour in the past 2,000 (numba, early-abandoning) |

3. **Adaptive threshold (EVT / SPOT).** Each expert's scores above a high quantile are fit with a Generalized Pareto distribution, updated online by method of moments in O(1). With only ~12 peaks after warm-up the moment estimate of the shape γ is noisy and often spuriously heavy (median γ 0.27 on the level expert), which made obvious spikes unflaggable; heavy-tail shapes are therefore **shrunk toward the exponential tail** with weight `peaks / (peaks + 200)`, so a heavy tail is believed only once there is evidence for it. The alarm threshold is a **risk** (tail probability 1e-5), not a hand-picked constant, so it means the same thing on every series. Alarm scores are winsorised into the tail model: a recurring pattern gradually becomes normal, but one huge anomaly cannot mask the next.
4. **Fusion and accumulation.** The fused score is −log₁₀(min_e p_e × E), a Bonferroni correction across experts. A CUSUM on that score lets sustained moderate surprise build up to an alarm.
5. **Alarm logic.** A 300-point cooldown between alarm onsets, events capped at 30 points, and hysteresis on release. The result is one alarm per incident, which is what event-F1 rewards.
6. **Re-learning.** If the alarm condition is still on at the first point after a cooldown, it is the same incident still running, so the normal has changed. Instead of alarming again, every expert and tail is rebuilt from the last 300 points (novelty and seasonal keep their long memory). This removed 9 false alarms and lost no window; it fires 3 times in the whole corpus and takes about 0.2 s each time.

## Evaluation

`sentinel/metrics.py` implements the event-based protocol:

- a *predicted event* is a run of consecutive ANOMALOUS points;
- a window counts as detected if any predicted event overlaps it;
- precision = predicted events that hit a window / all predicted events;
- recall = detected windows / all windows;
- delay is measured from window start and from the labelled anomaly point.

The median first alarm comes **0 steps after the labelled anomaly point**. 66% of detections are within ±10 steps, and 18 fire before the label.

We do not use point adjustment (counting a whole window as found once any point in it fires). On our final labels it gives 0.823, but random alarms on 1% of points get 0.919, so it cannot tell a detector from noise. Under event-F1 the same random alarms get 0.163. `python scripts/metric_check.py` reproduces this from saved outputs (`results/metric_check.csv`, 0 runs).

## Reproducing everything

```bash
python scripts/eda.py                          # EDA: tables in results/eda/, figures eda_*.png (0 runs)
python scripts/experiment.py --status          # 15-run budget: run log and decisions
python scripts/experiment.py --run N --tag T --why "hypothesis" --set k=v   # one budgeted run -> runs/run_NN/
python scripts/analyze_run.py 6 --vs 1         # error analysis from saved outputs (0 runs)
python scripts/metric_check.py                 # event-F1 vs point-adjusted F1 vs random (0 runs)
python scripts/web_deck_data.py                # chart data for the web deck -> web/data.js (0 runs)
cd deck && latexmk -xelatex sentinel.tex && cd ..   # the submitted deck (figures: python scripts/deck_figures.py)
python scripts/export_pptx.py                  # deck PDFs -> presentation/Sentinel.pdf + .pptx, Sentinel_backup.*
python scripts/export_web_pptx.py              # optional: web deck -> presentation/Sentinel_web.pptx + .pdf
python scripts/web_demo_data.py                # 58 CSVs + manifest for the live demo -> web/demo/data/
node scripts/check_web_engine.js               # JavaScript detector == predictions.csv on all 365,558 points
python predict.py --no-labels                  # submission files, inference only
python scripts/finalize.py 6                   # check labels == run 6, write predictions/metrics.json
python scripts/profile_data.py                 # dataset profile (phase 1)
python scripts/benchmark.py --tag mytag        # one run, logged to experiments.csv
python scripts/tune.py --grid risk=1e-4,1e-5 cooldown=150,300   # grid + LOCO score
python scripts/run_baselines.py                # results/baselines.csv
python scripts/diagnose.py                     # per-series alarms / FP breakdown
python predict.py                              # final predictions + metrics
python scripts/make_figures.py                 # all slide figures -> results/figures/
```

Phase 1 (development) logged 124 runs to `experiments.csv`. Phase 2 ran under a 15-run budget: `runs/run_log.csv` holds each run's hypothesis (written before the run), result and keep/drop decision, and `runs/run_NN/` holds every output, so no run ever needs repeating. Parallel scripts use at most half the CPU cores with one BLAS thread each.

## Experiment history

| Step | Event-F1 | Change |
|---|---|---|
| v1 | 0.343 | 5 experts + EVT threshold |
| v2 | 0.473 | fixed perpetual alarms (tail never learning) and scale collapse on flat data |
| v3 | 0.500 | CUSUM evidence accumulation; shorter warm-up |
| v4 | 0.535 | novelty expert (streaming matrix profile) |
| v5 | 0.551 | risk 1e-5; ablation showed the `jump` expert only added false alarms, so it was dropped |
| *phase 2 (15-run budget, 7 used)* | | |
| run 2 | 0.561 | tail-shape shrinkage (n0 = 100): new extremes caught 31 → 35 |
| run 4 | 0.584 | + re-learning after a regime change: false alarms 47 → 38, no window lost |
| run 6 | **0.594** | shrinkage n0 = 200; bracketed by n0 = 100 (0.584) and 400 (0.571) |

Dropped in phase 2: shrinkage 400 without re-learning (0.544), warm-up 200 (0.556), shrinkage 400 with re-learning (0.571). Seven more ideas were rejected from saved outputs without spending a run ([docs/02](docs/02_experiment_budget.md)).

## Data

NAB corpus, downloaded from Kaggle (`boltzmannbrain/nab`), 58 series in `data/nab/series/`. The Kaggle archive has no labels, so the ground-truth windows (`combined_windows.json`) and anomaly points (`combined_labels.json`) come from the official Numenta NAB repository. The Kaggle CSVs were verified byte-identical to Numenta's (after line-ending normalisation). NAB is AGPL-3.0; see `data/nab/NAB_LICENSE.txt`.

## Limitations

- The first 300 points are learning-only, so the 3 windows that start inside the warm-up cannot be detected. A 200-point warm-up was tried (run 5) and added more early false alarms than detections.
- "Repeated extremes" (an equally large excursion already happened earlier without a label) are caught only 11 of 32 times. By design an extreme-value detector learns that such sizes are normal.
- Traffic (recall 0.29) and Ad Exchange (recall 0.36) remain the weakest categories: short series, and repeated labelled anomalies teach the detector that the anomaly is normal.
- Heavy-tailed bursty series (e.g. Twitter IBM) produce false alarms on large but unlabelled bursts.
- A re-learning step takes about 0.2 s (it fired 3 times in 365,558 points).
- Hyperparameters are global, never per series. The phase-2 hypotheses were designed after looking at all labelled data, so the public score is optimistic for the hidden test.

## Layout

```
predict.py      run the detector on a folder of CSVs -> predictions/
app.py          Streamlit demo (Python)
sentinel/       the detector: data.py experts.py threshold.py engine.py evidence.py metrics.py baselines.py
predictions/    SUBMITTED RESULTS: predictions.csv, events.json, metrics.json, runtime.json
presentation/   SUBMITTED DECK: Sentinel.pptx, Sentinel.pdf
web/            deployable site: index.html (deck), demo/ (live demo, detector ported to JavaScript)
docs/           plain-language notes: presenter guide, data & EDA, 15-run budget, improvements, web deck, web demo
scripts/        eda.py experiment.py analyze_run.py finalize.py make_figures.py metric_check.py
                web_deck_data.py export_web_pptx.py web_demo_data.py check_web_engine.js
                benchmark.py tune.py run_baselines.py diagnose.py profile_data.py   (phase 1)
runs/           run_log.csv + run_NN/ (outputs of every budgeted run)
results/        EDA tables, baselines, delay and metric checks, figures
data/nab/       the 58 NAB series and labels
deck/           LaTeX source of the submitted deck (sentinel.tex, preamble.tex, figures/), 5 backup Q&A slides; old/ = earlier technical version
experiments.csv phase-1 log (124 runs)
```
