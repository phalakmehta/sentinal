# Sentinel: project notes in plain words

These notes explain what we did and why, in simple language. The main
[README](../README.md) is the technical summary; these pages are the story behind it.

| Page | What it covers |
|---|---|
| [00 · Presenter guide](00_presenter_guide.md) | **Start here.** The whole project in plain words, what to say per slide and per figure, numbers to remember, judge Q&A |
| [01 · Data and EDA](01_data_and_eda.md) | What the data looks like, how we prepare it, what kinds of anomalies it has, and what that told us to try |
| [02 · The 15-run budget](02_experiment_budget.md) | The rules we followed, every run with its reason and result, and the runs we saved by thinking first |
| [03 · The two improvements](03_improvements_explained.md) | Tail-shape shrinkage and re-learning, explained with examples |
| [04 · The web deck](04_web_deck.md) | The final 10-slide presentation as a web page, how to present it, how it becomes `presentation/Sentinel.pptx` and `.pdf` |
| [05 · The live web demo](05_web_demo.md) | The demo that runs the detector in the browser, the proof that it matches our Python results, and how to deploy it |

## The story in one minute

1. We started from our v5 detector: **event-F1 0.551**.
2. We were allowed **15 experiment runs** at most. One run = run the detector on all 58 series and score it once.
3. Before spending any run, we studied the data (EDA). The key finding: **55 of the 116 labelled anomalies are "new extremes"**, the biggest value the series had ever shown, yet v5 caught only 56% of them.
4. Looking inside the detector showed why: its "how rare is this?" curve was often guessed as far too heavy, so even huge spikes looked ordinary.
5. Two changes fixed most of this:
   - **Tail-shape shrinkage:** do not believe a heavy curve until there is enough evidence for it.
   - **Re-learning:** when an alarm is still on after the 300-point quiet period, the "normal" has changed, so learn the new normal instead of alarming again.
6. Result: **event-F1 0.594** (precision 0.63, recall 0.56) using **7 of the 15 runs**. Speed is unchanged at about 61 µs per point.

| | Event-F1 | Precision | Recall | Windows found | False alarms |
|---|---|---|---|---|---|
| Start (v5, run 1) | 0.551 | 0.615 | 0.500 | 58 / 116 | 37 |
| **Final (run 6)** | **0.594** | **0.632** | **0.560** | **65 / 116** | 39 |

## How to rerun everything

```bash
python scripts/eda.py                    # EDA tables + figures (0 runs)
python scripts/experiment.py --status    # see the run log and budget
python scripts/analyze_run.py 6 --vs 1   # error analysis of a saved run (0 runs)
python scripts/metric_check.py           # why event-F1, not point-adjusted F1 (0 runs)
python predict.py --no-labels            # submission files, inference only (0 runs)
python scripts/finalize.py 6             # check they match run 6, write metrics.json
python scripts/make_figures.py           # all slide figures
```
