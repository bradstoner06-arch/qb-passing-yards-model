# QB Passing Yards Model (v4)

A desktop tool that projects an NFL quarterback's **passing yards** and **touchdown passes** for an upcoming game, then turns each projection into calibrated over/under probabilities for a given line.

## How it works

**Baseline.** A blend of three views of the QB's recent form (last 8 usable games):
- an exponentially weighted moving average of passing yards (decay 0.85),
- the median, which resists one-off game-script outliers,
- attempts × yards per attempt, a volume-times-efficiency cross-check.

It's then adjusted for opponent and own-team pass EPA, sack rate, and aDOT, plus the Vegas total and spread.

**XGBoost.** A gradient-boosted model trained on every historical QB game. Its predictions are clamped to within 60 yards of the baseline, so one unstable leaf can't dominate. How much weight it gets depends on whether it *amplifies* or *corrects* an already-extreme baseline, a split measured in the backtest.

**Leak-free features.** Opponent defensive features come from a date-ordered index. Each training row sees only what that defense allowed in games played *before* the target game, so no future information gets into training.

**Low-volume games** (under 15 attempts: injury exits, benchings, rested starters) are down-weighted instead of dropped. Dropping them would bias projections high, because those props still settle.

**TD passes.** A separate Poisson model (XGBoost with `count:poisson`) produces an expected TD rate, and over/under probabilities come from the exact Poisson distribution.

**Probabilities.** Over/under probabilities for yards use a closed-form Normal or right-skewed shifted-gamma distribution, followed by logit calibration. The spread blends the QB's own recent variance with the model's out-of-sample residual error for that attempt-volume bucket.

**Backtest.** A walk-forward backtest fits the blend weights, Vegas slopes, bias terms, calibration strength, and distribution shape (scored by Brier score). The results are saved with a fingerprint of the model configuration. If the model changes, stale results are ignored rather than misapplied.

**Weather.** Wind, cold, rain, and snow adjust the mean and widen the spread for outdoor games.

## Backtest results

From a 5-fold expanding-window walk-forward backtest over 912 QB games (456 out-of-sample predictions, Nov 2025 – Nov 2026). Full output is in `backtest_results.json`.

| Metric | Result |
|---|---|
| Mean absolute error | 63.1 yds (vs. 64.7 for the rolling-average baseline alone) |
| Average bias | +2.0 yds (vs. +11.2 before the fitted bias/Vegas correction) |
| Calibration | Predicted over-probabilities within 2 percentage points of observed hit rates in every bin |
| Best XGBoost weight | 0.5 when correcting an extreme baseline toward the mean, 0 when amplifying it |

**Limitations:** No historical sportsbook lines are included, so over/under calibration is scored at synthetic lines placed around each projection, not real market lines. Results show forecast accuracy, not an edge against the market.

## Files

| File | Purpose |
|---|---|
| `qb_gui_V4.py` | Model and Tkinter desktop app |
| `prop_calibration.py` | Over/under distributions (Normal, shifted gamma) and calibration selection |
| `backtest_qb_model.py` | Walk-forward backtest; writes `backtest_results.json` |
| `backtest_results.json` | Latest backtest output, read by the app on startup |

## Data

Weekly QB stats and team EPA come from [nflverse](https://github.com/nflverse). Earlier seasons can also use Pro-Football-Reference game-log exports. When a season is covered by both sources, duplicate games are removed.

## Run it

```bash
pip install -r requirements.txt
python backtest_qb_model.py   # optional: re-run the backtest
python qb_gui_V4.py
```

## Disclaimer

This is a statistics project. Its output is a model opinion, not a guaranteed edge.
