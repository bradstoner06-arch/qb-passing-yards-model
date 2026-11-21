# QB Passing Yards Model

A desktop tool that projects an NFL quarterback's passing yards for an upcoming game, simulates the range of outcomes, and compares the resulting probabilities to a market line.

## What it does

- **Live data:** pulls current-season weekly player stats and team defensive EPA from [nflverse](https://github.com/nflverse) via `nfl_data_py`.
- **Two projection methods** (plus a blend):
  - *Adjustment model*: a transparent baseline from the QB's recent form, adjusted for opponent pass defense (EPA/play), Vegas total and spread, and home field.
  - *Global regression*: a standardized regression trained across every QB-week in the season, so each projection borrows strength from the whole league instead of one player's small sample.
- **Context adjustments:** wind, temperature, rain, snow, and indoor/dome games shift the mean and widen the spread.
- **Monte Carlo simulation:** 20,000 simulated games turn the projection into over/under probabilities for any line.
- **Pricing and sizing:** converts American odds to implied probability, reports the model's edge on each side, and sizes positions with fractional Kelly criterion.

## Project structure

| File | Purpose |
|---|---|
| `qb_gui.py` | Tkinter desktop app (inputs, threaded data loading, report) |
| `data_loader.py` | Loads nflverse data and builds pregame features for a QB |
| `model.py` | Projection, regression training, weather adjustments, simulation, odds and Kelly math |

## Run it

```bash
pip install -r requirements.txt
python qb_gui.py
```

1. Wait for the season data to load (status line at the top).
2. Optionally click **Train Global Model** to enable the regression and blend methods.
3. Enter the QB, opponent team code, Vegas total and spread, weather, and the line and odds.
4. Click **Run Projection**.

## Disclaimer

This is a statistics project. Its output is a model opinion, not a guaranteed edge. Markets already price in most public information.
