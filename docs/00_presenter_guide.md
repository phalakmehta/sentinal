# Presenter guide: understand it, then present it

Read this once from top to bottom. Everything you need to explain Sentinel to the judges
is on this page.

## 1. The whole project in 30 seconds

> Machines, websites and sensors produce a number every few minutes. Sometimes something
> goes wrong: a spike, a crash, a sudden change. Sentinel watches these numbers **live**,
> learns what "normal" looks like for each one, and raises an alarm only when something is
> **truly rare** for that signal, with a one-line explanation of why. On the official
> benchmark it finds **more than twice as many anomalies correctly** as the standard methods,
> and it handles **16,000 points per second** on one CPU core.

## 2. How Sentinel works, in plain words

Think of a security guard watching one number at a time.

1. **It never looks ahead.** It sees each new value once, decides, and moves on. That is what
   "streaming" means.
2. **It first learns what normal is.** For the first 300 values it only watches and learns; no
   alarms.
3. **Five watchers look for five kinds of weird:**

   | Watcher | Looks for | Everyday example |
   |---|---|---|
   | level | a sudden spike or dip | CPU jumps from 10% to 90% |
   | seasonal | the daily routine is broken | taxi demand is low on a normal Monday |
   | shift | a new level that stays | traffic speed drops and stays low |
   | volatility | the signal gets jumpy, or goes flat | a sensor freezes |
   | novelty | a shape never seen before | a pattern that has never happened |

4. **Each watcher has a "rarity meter".** It answers: *how often have I seen something this
   extreme on this particular signal?* An alarm needs something rarer than **1 in 100,000**.
   Because the rule is about rarity, not a fixed number, it works the same on CPU percentages,
   tweet counts or taxi rides, with nothing hand-tuned per signal. The maths behind the meter is
   **extreme-value theory (EVT)**: statistics for estimating how rare the most extreme values are.
5. **One alarm per incident.** After an alarm it waits 300 points before alarming again.
6. **Every alarm comes with a reason**, for example:
   *"Value 50.00 is 99.4 scales above the learned baseline 0.32."*

**What we improved in this round** (one sentence each):
- **Don't jump to conclusions:** with only a few extreme values seen, the rarity meter used to
  guess "huge values are normal" and missed obvious spikes. Now it assumes a normal amount of
  extremes until there is real evidence otherwise.
- **Accept a new normal:** if a problem is still going on when the waiting time ends, the signal
  has changed for good, so Sentinel re-learns normal instead of alarming again and again.

## 3. The three EDA figures, and what to say

**Figure 1 · `eda_examples.png`: what the data looks like**
> "This is the Numenta Anomaly Benchmark: 58 real and artificial signals, 365 thousand points,
> 116 labelled anomalies (grey bands). Servers, online ads, taxis, road sensors, Twitter. They
> look nothing alike, so one fixed threshold cannot work for all of them."

**Figure 2 · `eda_simple_rule_fails.png`: why simple rules fail**
> "Here is the textbook rule, 'alarm if a value is 3 standard deviations from average'. On this
> Twitter signal it fires 114 times; there are only 2 real anomalies. Across the whole dataset it
> raises 2,403 alarms for 116 real anomalies. Real data has extreme values all the time. Sentinel
> fires twice here, both correct, and 106 times on the whole dataset."

**Figure 3 · `eda_anomaly_types.png`: what kinds of anomalies exist**
> "We sorted every labelled anomaly into four types. Almost half, 55 of 116, are the biggest
> spike the signal has ever shown, so a rarity-based detector should catch them. 32 are big
> spikes where an even bigger one already happened earlier without a label; no method can tell
> those apart from the data alone. That split told us where to improve and where not to waste
> effort."

Then, on the results slide, **`error_by_type.png`** closes the loop:
> "After our improvements we catch 37 of the 55 biggest-ever spikes (up from 31) and 11 of 14
> level shifts."

## 4. Numbers to remember

| What | Number |
|---|---|
| Dataset | 58 signals · 365,558 points · 116 labelled anomaly windows |
| **Event-F1 (main score)** | **0.594** (we started this round at 0.551) |
| Precision / recall | 0.63 / 0.56 → 65 of 116 anomalies found, 39 false alarms |
| Other methods (7 tested) | best is temporal clustering, 0.316 → we are **1.9× better**; best standard method (LOF) 0.260 → **2.3×**; autoencoder 0.272; ensemble of all six 0.280. All had their threshold set using the answers |
| Speed | 61 µs per point = 16,000 points/second on one core; whole dataset in about 7 s |
| Timing of alarms | median **0 steps** after the labelled anomaly; 66% within 10 steps |
| Clean signals | 0 false alarms on the 5 signals that have no anomalies |
| Experiment budget | 7 of 15 runs used, 8 kept in reserve |

## 5. Slide by slide

| Slide | Show | Say |
|---|---|---|
| 1 Title | team, track | "Sentinel: a live anomaly detector that knows what is rare for each signal and explains every alarm." |
| 2 Problem | (text) | Downtime costs money; alarms that cry wolf get ignored; you need few, correct, explained alarms, live. |
| 3 Data + EDA | the 3 EDA figures | Section 3 above. Preprocessing: read in time order, no filling, no resampling, no scaling; the detector learns each signal's scale itself. |
| 4 Pipeline | diagram | value → 5 watchers → rarity meters → combine (take the most surprised) → alarm logic → label + explanation. |
| 5 Algorithm | `threshold_design.png`, `tail_shrinkage.png`, `relearn_example.png` | Rarity meter (EVT) gives one meaning of "rare" everywhere; plus the two upgrades. |
| 6 Experiments | `budget_runs.png` | 15-run budget, one hypothesis per run, kept only if the score rose; 3 kept, 3 dropped, 8 runs saved. |
| 7 Evaluation | `per_category.png`, `detection_delay.png` | Event-F1 = any alarm inside a grey band counts; precision, recall, timing; tested by holding out whole categories. Last line: we do **not** use point adjustment, because it lets random alarms score 0.92 (backup slide 5). |
| 8 Results | table + bars of all 8 methods | best of 8: 0.594 vs 0.316 (temporal clustering); 6 to 15× fewer false alarms. |
| 9 Demo | app screenshot, `detections.png` | live replay, alarms with explanations. |
| 10 Limits | `false_positives.png` | "seen before" spikes, short hourly series, unlabelled Twitter bursts; next steps. |

## 6. Questions judges may ask

**Why not deep learning, like an autoencoder?**
Each signal must be learned live from its own first few hundred points, with no training set.
Deep models need lots of data per signal, are slow to adapt, and are hard to explain. Ours is
We tested one anyway: an autoencoder trained the same way scores 0.272, and an ensemble of all six baselines 0.280.
Sentinel scores 0.594, explains every alarm, and runs in 61 µs per point.

**Papers report F1 above 0.9. Why is yours only 0.594?**
Many papers use "point adjustment": once any point of an anomaly window fires, they count every
point of that window as found. A median NAB window is 212 points long, so one lucky alarm becomes
212 correct points. We scored our own final labels that way, plus detectors that know nothing
(`python scripts/metric_check.py`, no run spent; show backup slide 5):

| Detector | Event-F1 (the challenge metric) | Point-adjusted F1 | Point-wise F1 |
|---|---|---|---|
| **Sentinel** | **0.594** | 0.823 | 0.045 |
| Random alarms on 1% of points | 0.163 | **0.919** | 0.018 |
| Random alarms on 0.28% of points (our own alarm rate) | 0.146 | 0.765 | 0.005 |
| Flag every point | 0.945 | 0.168 | 0.168 |

What to say: "With point adjustment, random noise beats us, so that number means nothing. The
challenge asks for event-F1, and there random scores 0.16 while we score 0.594."
If they ask about the last row: every F1 can be gamed. One endless alarm per series touches every
window. That is why we also report the alarm count (106 alarms, 39 of them false) and the delay,
and why our alarms are short.

**What does "1 in 100,000" mean?**
The alarm fires only when a value is that rare *for that signal*. It is the same rule on every
signal; we never set a threshold by hand.

**How do you know you didn't just overfit?**
We changed only global settings (the same for every signal), at most 15 experiments, and checked
every choice by holding out whole categories of data. We still expect the hidden-test score to be
a little lower than 0.594.

**Why is recall only 0.56?**
32 of the 116 anomalies are spikes where a bigger one already happened earlier, unlabelled. From
the data alone, nobody can tell those apart. On the anomalies that are detectable in principle we
do much better: 67% of biggest-ever spikes, 79% of level shifts.

**Where do the false alarms come from?**
Mostly large Twitter bursts that look anomalous but were not labelled (show `false_positives.png`).

**Is it really live, with no peeking at the future?**
Yes. Every point is processed once, and the answer at time t uses only data up to time t.

**What happens if the signal changes for good, like a new server load?**
Sentinel notices the alarm is still on after the waiting time, and re-learns the new normal
(show `relearn_example.png`: 7 false alarms down to 1).

**How does it explain an alarm?**
The watcher that fired writes one sentence with the value, the expected value and how far off it
is; the full numbers for all five watchers are saved too.

**What would you do with more time? Why not do it now?**
The next steps all need data this benchmark does not have. Operator feedback ("this was not an anomaly")
needs people marking alarms in real use; reusing the benchmark answers would be cheating. Watching related
signals together (CPU, disk and network of one machine) needs to know which signals belong together, and NAB
gives each one alone. Holiday effects need more than a year of data; no signal here covers a full year.
(Weekly rhythm is already handled: Sentinel tries a weekly cycle when it has enough history.)

**Why is 0.594 the best you could get?**
30 of the 51 missed anomalies are "seen before" spikes or subtle changes that a data-only detector cannot
separate from normal data. Every other idea we tried caught about 1 more anomaly for 6 to 14 more false
alarms, while one catch is only worth about 2.4 false alarms in event-F1. The tuning peaked at 0.594 with
both neighbours lower (0.584 and 0.571).

## 7. Be honest about these

- Do not promise a hidden-test score. Say "0.594 on the public benchmark".
- The 15-run budget covers this improvement round; our earlier development phase (v1 to v5) had
  124 logged runs.
- Name the limits yourself before the judges do: "seen before" spikes, short hourly signals, a
  0.2-second pause on the rare re-learning step (3 times in 365,558 points).
