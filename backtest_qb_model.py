"""
Walk-forward backtest for the QB passing-yards model (qb_gui_V4).

Addresses three review findings that qb_gui_V4.py itself can't answer on its own:

  1. "There's currently no out-of-sample test proving XGB deserves 70%."
     -> trains XGB on an expanding window of chronologically-earlier games only,
        evaluates on later games it never saw, and sweeps the XGB/EWMA blend
        ratio to find the MAE/RMSE-minimizing weight instead of the hardcoded
        XGB_BLEND = 0.70 guess.

  2. "Your calibration isn't actually calibrated from results."
     -> sweeps CALIBRATION_ALPHA against out-of-sample Brier score instead of
        eyeballing a shrinkage factor.

  3. "The model's uncertainty is still pretty primitive."
     -> reports the actual out-of-sample residual std, bucketed by attempt
        volume, for the live app to blend with each QB's own recent-game std.

IMPORTANT LIMITATION — read before trusting the numbers:
No historical sportsbook lines exist in this repo, so there is no real market
line to test over/under accuracy or calibration against. This script uses the
EWMA-only baseline's own pregame forecast as a stand-in "line" — i.e. it asks
"when XGB/the blend disagrees with the simple baseline about over/under
relative to the baseline's own number, how often is it right, and are its
probabilities calibrated against that?" That is a legitimate, non-circular
test of whether XGB adds signal beyond the baseline, but it is NOT the same as
testing against a real closing line. Feed in actual historical prop lines
(keyed by qb/date) if you want the real thing.

Opponent defense features come from defense_allowed_asof() — a per-game,
date-lagged proxy built from QB box scores, looked up strictly before each
target game's own date (see build_defense_allowed_index in qb_gui_V4.py).
That's real weekly granularity, not a season aggregate, so both the
cross-season and within-season leaks the review originally flagged are
closed for those features. The two dominant-personnel features (EPA/Pass,
Pass Rate) were dropped entirely — their only source (SumerSports) turned
out to require a paid subscription, and it wasn't worth scraping a vendor's
paywalled product for two minor features. LOW_VOL_SAMPLE_WEIGHT
(0.35) is also still hand-picked rather than itself walk-forward-validated.

Usage:
    python backtest_qb_model.py

Writes backtest_results.json next to this script; qb_gui_V4.py reads it on
startup and uses best_xgb_blend / best_calibration_alpha / residual_std_by_bucket
in place of its hardcoded defaults if present.
"""
import json
import os
import time

import numpy as np

import prop_calibration as pc
import qb_gui_V4 as m

if not m.HAS_XGB or not m.HAS_PANDAS:
    raise SystemExit("xgboost and pandas are required to run the backtest.")

import xgboost as xgb

RESULTS_PATH = m.BACKTEST_RESULTS_PATH
VEGAS_LINES_PATH = os.path.join(m.BASE_DIR, "nflverse_vegas_lines.parquet")
K_FOLDS = 5
MIN_TRAIN_FRAC = 0.5          # first 50% of chronological rows seed the initial training window
BLEND_GRID = [i / 20 for i in range(21)]     # 0.00, 0.05, ..., 1.00
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
MIN_EDGE_YDS = 2.0             # ignore near-ties vs baseline when scoring directional accuracy


def load_vegas_lines():
    """Real (date, team) -> (total_line, team_spread) from fetch_vegas_lines.py,
    or {} if it hasn't been run. team_spread is already in standard American
    convention (negative = that team favored) from that team's own
    perspective. Returns {} (not an error) if missing — callers degrade to
    "no real line available for this row" rather than failing."""
    if not os.path.isfile(VEGAS_LINES_PATH):
        return {}
    import pandas as pd
    df = pd.read_parquet(VEGAS_LINES_PATH)
    # fetch_vegas_lines.py stores raw nflverse team codes (e.g. "KC") — resolve
    # to the same full-team-name key build_rows looks up with (m.resolve_team
    # handles both nflverse's and PFR's abbreviation sets already).
    return {(r["Date"], m.resolve_team(r["Team"])): (float(r["total_line"]), float(r["team_spread"]))
            for _, r in df.iterrows()}


def build_rows(data):
    """One row per historical QB game, built exactly the way
    build_xgb_training_data() does (leak-free, per-game-lagged opponent
    defense via defense_allowed_asof), plus the EWMA-only baseline forecast
    and the QB's own historical std at that point in time — so every row is
    fully reproducible pregame info."""
    defense_index    = data["defense_allowed_index"]
    defense_fallback = data["defense_allowed_fallback"]
    vegas_lines      = load_vegas_lines()

    rows = []
    for qb_name in m.list_qbs():
        try:
            games, _ = m.load_qb_games(qb_name)
        except Exception:
            continue
        if len(games) < m.ROLLING_GAMES + 1:
            continue

        for i in range(m.ROLLING_GAMES, len(games)):
            prior  = games[:i]
            target = games[i]
            if not str(target.get("Date", "")).strip():
                continue

            avgs = m.qb_rolling_avg(prior)
            pool = m.usable_games(prior)
            recent = pool[-m.ROLLING_GAMES:] if len(pool) >= m.ROLLING_GAMES else pool
            if len(recent) < 2:
                continue

            median_yds = float(np.median([g["Yds"] for g in recent]))
            attya      = avgs["Att"] * avgs["Y/A"]
            ewma_pred  = (m.EWMA_BLEND * avgs["Yds"] + m.MEDIAN_BLEND * median_yds
                          + m.ATTYA_BLEND * attya)

            if len(recent) >= 4:
                hist_std = float(np.std([g["Yds"] for g in recent], ddof=1))
            else:
                hist_std = m.DEFAULT_STD
            hist_std = m.clamp(hist_std, m.STD_MIN, m.STD_MAX)

            opp_full  = m.resolve_team(target["Opp"])
            def_proxy = m.defense_allowed_asof(defense_index, opp_full, target["Date"], defense_fallback)
            feat = m._qb_features(avgs, def_proxy, i)

            # Real total/spread for THIS QB's team: looked up via the
            # opponent's own (date, team) row and flipped, since we only know
            # the opponent's abbreviation from the game log, not the QB's own
            # team. total is shared; spread is the negation of the
            # opponent's (one side's favorite-by-X is the other's dog-by-X).
            opp_line = vegas_lines.get((str(target["Date"]), opp_full))
            total_line  = opp_line[0] if opp_line else None
            team_spread = -opp_line[1] if opp_line else None

            rows.append({
                "qb": qb_name,
                "date": str(target["Date"]),
                "att": avgs["Att"],
                "y": float(target["Yds"]),
                "low_vol": bool(target.get("low_vol", False)),
                "feat": feat,
                "ewma_pred": float(ewma_pred),
                "ewma_yds_only": float(avgs["Yds"]),
                "median_yds_only": median_yds,
                "attya_only": float(attya),
                "hist_std": hist_std,
                "total_line": total_line,
                "team_spread": team_spread,
            })

    rows.sort(key=lambda r: r["date"])
    return rows


def _new_model():
    """Must mirror train_xgb_model() in qb_gui_V4.py exactly — this backtest
    is only meaningful if it evaluates the same model production actually uses."""
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
        reg_alpha=0.1, reg_lambda=2.0, random_state=42, verbosity=0,
    )


def _row_total(r):
    return r["total_line"] if r.get("total_line") is not None else 44.0


def _row_spread(r):
    return r["team_spread"] if r.get("team_spread") is not None else 0.0


def fit_game_coefs(train_rows, xgb_train_preds=None):
    """Fits the game-context correction on TRAINING rows only (v7).

    OLS: (actual yds - raw pre-Vegas baseline) ~ 1 + (total - 44) + spread.
      - intercept  -> baseline_bias: the EWMA/median/Att×Y/A baseline's mean
        error at a 44 total / pick'em. Mostly comes from usable_games() leaving
        <15-att games out of the averages while outcomes still include them
        (injury exits, benchings, Week 18 rests — props on those still settle).
      - total/spread slopes replace apply_vegas()'s hand-set 1.7 / 1.0.
    Unweighted: low-volume outcomes are real bet outcomes, so they count fully
    here even though XGB down-weights them as training targets.

    XGB is trained with low-vol targets at LOW_VOL_SAMPLE_WEIGHT, so it already
    absorbs part of that drag and needs its own (smaller) intercept. xgb_bias =
    mean unweighted in-sample residual of the clamped XGB prediction after the
    same Vegas slopes (in-sample is fine for an intercept: squared-error boosting
    leaves the *weighted* mean residual near 0, so the unweighted mean isolates
    the down-weighting gap). Same slopes for both — XGB has no Vegas features.
    """
    X = np.array([[1.0, _row_total(r) - 44.0, _row_spread(r)] for r in train_rows])
    yres = np.array([r["y"] - r["ewma_pred"] for r in train_rows])
    coef, *_ = np.linalg.lstsq(X, yres, rcond=None)
    resid = yres - X @ coef
    se = np.sqrt(np.diag(np.var(resid, ddof=X.shape[1]) * np.linalg.inv(X.T @ X)))
    out = {
        "baseline_bias": float(coef[0]), "total_coef": float(coef[1]), "spread_coef": float(coef[2]),
        "baseline_bias_se": float(se[0]), "total_coef_se": float(se[1]), "spread_coef_se": float(se[2]),
        "n": len(train_rows), "xgb_bias": 0.0,
    }
    if xgb_train_preds is not None:
        vegas = coef[1] * X[:, 1] + coef[2] * X[:, 2]
        y = np.array([r["y"] for r in train_rows])
        xg = np.array([m.clamp_xgb_deviation(float(p), r["ewma_pred"])
                       for p, r in zip(xgb_train_preds, train_rows)])
        out["xgb_bias"] = float(np.mean(y - xg - vegas))
    return out


HARDCODED_COEFS = {"baseline_bias": 0.0, "xgb_bias": 0.0, "total_coef": 1.7, "spread_coef": 1.0}
NO_VEGAS_COEFS  = {"baseline_bias": 0.0, "xgb_bias": 0.0, "total_coef": 0.0, "spread_coef": 0.0}
MODES = ("none", "hardcoded", "fitted")


def _adjust(pred, r, coefs, kind):
    bias = coefs["xgb_bias"] if kind == "xgb" else coefs["baseline_bias"]
    return (pred + bias + coefs["total_coef"] * (_row_total(r) - 44.0)
            + coefs["spread_coef"] * _row_spread(r))


def walk_forward_all_modes(rows, k_folds=K_FOLDS, min_train_frac=MIN_TRAIN_FRAC):
    """Expanding-window walk-forward: fold k trains on every row strictly
    earlier (by date) than fold k's rows, never on rows from its own or a
    later fold. Each fold's XGB model is scored three ways from the same fit:
      none      — no Vegas/bias adjustment (what this backtest measured pre-v7)
      hardcoded — apply_vegas()'s old (total-44)*1.7 + spread*1.0 (pre-v7 production)
      fitted    — v7: fit_game_coefs() on that fold's training rows only
    Every result row keeps the raw pre-adjustment values as ewma_raw / xgb_raw
    (used for regime classification and as the stand-in line) and the adjusted
    ones as ewma_pred / xgb_pred (used for MAE/RMSE, blend and alpha sweeps)."""
    n = len(rows)
    start = int(n * min_train_frac)
    remaining = n - start
    fold_size = max(1, remaining // k_folds)

    results = {mode: [] for mode in MODES}
    fold_report = []
    for k in range(k_folds):
        test_start = start + k * fold_size
        test_end = n if k == k_folds - 1 else start + (k + 1) * fold_size
        if test_start >= n:
            break
        train_rows = rows[:test_start]
        test_rows  = rows[test_start:test_end]
        if len(train_rows) < m.MIN_XGB_SAMPLES or not test_rows:
            continue

        X_tr = np.array([r["feat"] for r in train_rows], dtype=np.float32)
        y_tr = np.array([r["y"] for r in train_rows], dtype=np.float32)
        w_tr = np.array([m.LOW_VOL_SAMPLE_WEIGHT if r["low_vol"] else 1.0
                          for r in train_rows], dtype=np.float32)

        model = _new_model()
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        fitted = fit_game_coefs(train_rows, model.predict(X_tr))
        coefs_by_mode = {"none": NO_VEGAS_COEFS, "hardcoded": HARDCODED_COEFS, "fitted": fitted}

        X_te = np.array([r["feat"] for r in test_rows], dtype=np.float32)
        xgb_preds = model.predict(X_te)

        for r, xp in zip(test_rows, xgb_preds):
            # Same MAX_XGB_DEVIATION_YDS guardrail production applies — evaluate
            # the model backtest actually reflects what a live prediction does.
            # Clamped against the raw baseline, pre-adjustment, as in the GUI.
            clamped = m.clamp_xgb_deviation(float(xp), r["ewma_pred"])
            for mode, c in coefs_by_mode.items():
                results[mode].append({
                    **r, "ewma_raw": r["ewma_pred"], "xgb_raw": clamped,
                    "ewma_pred": _adjust(r["ewma_pred"], r, c, "baseline"),
                    "xgb_pred": _adjust(clamped, r, c, "xgb"),
                    "xgb_pred_unclamped": float(xp),
                })
        fold_report.append({
            "fold": k, "train_rows": len(train_rows), "test_rows": len(test_rows),
            "train_end_date": train_rows[-1]["date"], "test_date_range": (test_rows[0]["date"], test_rows[-1]["date"]),
            "fitted_coefs": {kk: round(v, 3) for kk, v in fitted.items()},
        })

    return results, fold_report


def walk_forward(rows, k_folds=K_FOLDS, min_train_frac=MIN_TRAIN_FRAC, mode="fitted"):
    """Pooled out-of-sample predictions for one adjustment mode (default: the
    v7 production 'fitted' mode). Kept for qb_mc_experimental/backtest_vs_v4.py."""
    results, fold_report = walk_forward_all_modes(rows, k_folds, min_train_frac)
    return results[mode], fold_report


def sweep_blend(results):
    y    = np.array([r["y"] for r in results])
    ewma = np.array([r["ewma_pred"] for r in results])
    xgbp = np.array([r["xgb_pred"] for r in results])

    table = {}
    for b in BLEND_GRID:
        pred = b * xgbp + (1 - b) * ewma
        mae  = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        table[f"{b:.2f}"] = {"mae": mae, "rmse": rmse}

    best_blend = min(table, key=lambda k: table[k]["rmse"])
    return table, float(best_blend)


def _line(r):
    """Stand-in prop line: the RAW pre-Vegas, pre-bias baseline (v7). Fixed
    across adjustment modes so before/after are scored against the same number;
    in 'none' mode it equals ewma_pred, i.e. the pre-v7 definition."""
    return r.get("ewma_raw", r["ewma_pred"])


def directional_accuracy(results, best_blend):
    """Line = raw EWMA baseline forecast (_line). Edge = (blend/xgb pred) - line.
    Only scores rows where the model's edge vs. the line is non-trivial
    (>= MIN_EDGE_YDS) to avoid crediting coin-flip noise near a tie."""
    y    = np.array([r["y"] for r in results])
    ewma = np.array([r["ewma_pred"] for r in results])
    line = np.array([_line(r) for r in results])
    xgbp = np.array([r["xgb_pred"] for r in results])
    blend_pred = best_blend * xgbp + (1 - best_blend) * ewma

    def score(pred):
        edge = pred - line
        mask = np.abs(edge) >= MIN_EDGE_YDS
        if mask.sum() == 0:
            return None, 0
        correct = (edge[mask] > 0) == (y[mask] > line[mask])
        return float(np.mean(correct)), int(mask.sum())

    xgb_acc, xgb_n = score(xgbp)
    blend_acc, blend_n = score(blend_pred)
    return {
        "xgb_vs_baseline":   {"accuracy": xgb_acc, "n_scored": xgb_n},
        "blend_vs_baseline": {"accuracy": blend_acc, "n_scored": blend_n},
    }


def live_stds(results, resid_buckets):
    """The std the live app uses: half the QB's rolling std, half the OOS residual
    std of the attempt-volume bucket, clamped (weather widening not replayed)."""
    std_max = max(m.STD_MAX, min(150.0, max(resid_buckets.values()) * 1.25)) if resid_buckets else m.STD_MAX
    out = []
    for r in results:
        oos = resid_buckets.get(m._std_bucket(r["att"]))
        s = 0.5 * r["hist_std"] + 0.5 * oos if oos is not None else r["hist_std"]
        out.append(m.clamp(s, m.STD_MIN, std_max))
    return np.array(out)


def sweep_calibration(results, means, stds, previous=None):
    """v8: distribution (Normal vs shifted gamma) and alpha chosen jointly by
    pooled Brier at lines placed at a spread of offsets around the projection
    (mean + k*std, k in -1..+1; prop_calibration.sweep_offsets). The pre-v8
    method scored at the raw baseline (_line) — the projection's own neighbourhood,
    where P(over) ~ 0.5 for every row and alpha is barely identified.
    previous: (dist, alpha) of the config being replaced, scored on the same lines."""
    y = np.array([r["y"] for r in results], dtype=float)
    sw = pc.sweep_offsets(y, np.asarray(means, float), np.asarray(stds, float), baseline=previous)
    # Diagnostic only: the old stand-in curve (line = raw baseline).
    line = np.array([_line(r) for r in results])
    raw = np.array([m.prob_over(mu, sd, ln) for mu, sd, ln in zip(means, stds, line)])
    act = (y > line).astype(float)
    sw["brier_by_alpha_standin_line_DIAGNOSTIC_ONLY"] = {
        f"{a:.2f}": round(float(np.mean((pc.calibrate(raw, a) - act) ** 2)), 5) for a in pc.ALPHA_GRID}
    return sw


def residual_std_by_bucket(results, best_blend):
    xgbp = np.array([r["xgb_pred"] for r in results])
    ewma = np.array([r["ewma_pred"] for r in results])
    means = best_blend * xgbp + (1 - best_blend) * ewma
    y = np.array([r["y"] for r in results])
    atts = [r["att"] for r in results]

    buckets = {}
    for mean, actual, att in zip(means, y, atts):
        key = m._std_bucket(att)
        buckets.setdefault(key, []).append(actual - mean)

    out = {}
    for key, resid in buckets.items():
        if len(resid) >= 5:
            out[key] = round(float(np.std(resid, ddof=1)), 2)
    return out


AMPLIFY_DEV_THRESHOLD_YDS = 15.0  # how far ewma_pred must sit from league avg to call it an "extreme" baseline


def league_avg_yds(results):
    # Raw pre-Vegas baseline — the GUI compares raw_blend against this.
    return float(np.mean([r.get("ewma_raw", r["ewma_pred"]) for r in results]))


def classify_amplify_buckets(results, league_avg, threshold=AMPLIFY_DEV_THRESHOLD_YDS):
    """Splits OOS rows into three regimes by comparing the EWMA baseline's
    deviation from the league-average forecast to the direction XGB moves
    relative to that baseline:
      - "amplify": XGB pushes further the same way the baseline already
        deviates (a QB running low gets projected even lower, a QB running
        high gets projected even higher) — the "bad QB under / good QB over"
        pattern getting reinforced rather than checked.
      - "correct":  XGB pushes back toward league average.
      - "neutral":  baseline isn't far enough from league average to call
        either way.
    Rows keep their original dict (y/ewma_pred/xgb_pred/...) so callers can
    feed a bucket straight into sweep_blend().
    """
    buckets = {"amplify": [], "correct": [], "neutral": []}
    for r in results:
        # Classified on raw pre-adjustment values (ewma_raw / xgb_raw), exactly
        # as QBGuiApp.run() does with raw_blend / xgb_raw. Unchanged logic.
        base_raw = r.get("ewma_raw", r["ewma_pred"])
        dev = base_raw - league_avg
        adj = r.get("xgb_raw", r["xgb_pred"]) - base_raw
        if abs(dev) < threshold or abs(adj) <= 1e-6:
            bucket = "neutral"
        elif (dev > 0) == (adj > 0):
            bucket = "amplify"
        else:
            bucket = "correct"
        buckets[bucket].append(r)
    return buckets


def amplify_vs_correct_diagnostic(results, best_blend, league_avg):
    """Reports blend MAE vs. EWMA-only MAE against each row's REAL outcome
    (not the baseline-as-line proxy used elsewhere in this script, since
    actual yards are known here), split by the amplify/correct/neutral
    regimes above. If blend_mae < ewma_only_mae in "correct" but not in
    "amplify", that's direct evidence a single global XGB_BLEND makes the
    recurring bias worse on exactly the picks that produce it, not better.
    """
    buckets = classify_amplify_buckets(results, league_avg)
    out = {}
    for name, rs in buckets.items():
        if not rs:
            out[name] = {"n": 0}
            continue
        y    = np.array([r["y"] for r in rs])
        ewma = np.array([r["ewma_pred"] for r in rs])
        xgbp = np.array([r["xgb_pred"] for r in rs])
        blend_pred = best_blend * xgbp + (1 - best_blend) * ewma
        ewma_mae  = float(np.mean(np.abs(ewma - y)))
        blend_mae = float(np.mean(np.abs(blend_pred - y)))
        out[name] = {
            "n": len(rs),
            "ewma_only_mae": round(ewma_mae, 2),
            "blend_mae": round(blend_mae, 2),
            "blend_helps": blend_mae < ewma_mae,
        }
    return out


MIN_BUCKET_ROWS_FOR_OWN_BLEND = 40  # below this, fall back to the global best_blend rather than overfit a bucket


def sweep_blend_by_bucket(results, league_avg, global_best_blend):
    """Re-runs the blend sweep independently within each amplify/correct/
    neutral regime, so XGB can be weighted down (or to zero) specifically
    where it amplifies an already-extreme baseline, without throwing away
    the cases where it's earning its keep by correcting toward the mean.
    Buckets with too few OOS rows keep the global blend instead of fitting
    their own (noisy on ~80 rows otherwise)."""
    buckets = classify_amplify_buckets(results, league_avg)
    out = {}
    for name, rs in buckets.items():
        if len(rs) < MIN_BUCKET_ROWS_FOR_OWN_BLEND:
            out[name] = {"n": len(rs), "blend": global_best_blend, "fell_back_to_global": True}
            continue
        table, best = sweep_blend(rs)
        out[name] = {
            "n": len(rs),
            "blend": best,
            "fell_back_to_global": False,
            "mae_at_best_blend": table[f"{best:.2f}"]["mae"],
            "mae_at_global_blend": table[f"{global_best_blend:.2f}"]["mae"],
        }
    return out


def vegas_adjustment_backtest(rows, k_folds=K_FOLDS, min_train_frac=MIN_TRAIN_FRAC):
    """Was apply_vegas()'s hand-picked (total-44)*1.7 + spread*1.0 ever actually
    tested against real outcomes? No — it's applied live and in weekly_report,
    but build_rows previously had no real total/spread to test it against.
    Now that fetch_vegas_lines.py exists, this compares three OOS predictions
    for every row that has a real closing line:
      - no Vegas adjustment at all (pure pregame baseline)
      - the current hardcoded formula
      - a walk-forward-fit linear adjustment (OLS, refit each fold on
        strictly-earlier rows only, same expanding-window discipline as the
        XGB blend sweep)
    If "hardcoded" beats "none", the adjustment is earning its keep. If "fit"
    meaningfully beats "hardcoded", the coefficients are measurably off and
    should be updated to the fit values instead of the original guess.
    """
    lined = [r for r in rows if r.get("total_line") is not None]
    if len(lined) < 60:
        return {"n_with_real_lines": len(lined),
                "note": "too few rows with real lines (need nflverse_vegas_lines.parquet "
                        "— run fetch_vegas_lines.py) for a meaningful walk-forward fit"}

    n = len(lined)
    start = int(n * min_train_frac)
    remaining = n - start
    fold_size = max(1, remaining // k_folds)

    preds_none, preds_hardcoded, preds_fit, actuals = [], [], [], []
    fit_coefs = []
    for k in range(k_folds):
        test_start = start + k * fold_size
        test_end = n if k == k_folds - 1 else start + (k + 1) * fold_size
        if test_start >= n:
            break
        train = lined[:test_start]
        test = lined[test_start:test_end]
        if len(train) < 30 or not test:
            continue

        X = np.array([[1.0, r["total_line"] - 44.0, r["team_spread"]] for r in train])
        yres = np.array([r["y"] - r["ewma_pred"] for r in train])
        coef, *_ = np.linalg.lstsq(X, yres, rcond=None)
        fit_coefs.append(coef)

        for r in test:
            resid_fit = coef[0] + coef[1] * (r["total_line"] - 44.0) + coef[2] * r["team_spread"]
            preds_fit.append(r["ewma_pred"] + resid_fit)
            preds_hardcoded.append(_adjust(r["ewma_pred"], r, HARDCODED_COEFS, "baseline"))
            preds_none.append(r["ewma_pred"])
            actuals.append(r["y"])

    if not actuals:
        return {"n_with_real_lines": len(lined), "note": "no fold produced OOS rows"}

    actuals_arr = np.array(actuals)

    def mae(preds):
        return round(float(np.mean(np.abs(np.array(preds) - actuals_arr))), 2)

    avg_coef = np.mean(fit_coefs, axis=0) if fit_coefs else None
    return {
        "n_with_real_lines": len(lined),
        "n_scored": len(actuals),
        "mae_no_vegas_adjustment": mae(preds_none),
        "mae_hardcoded_formula": mae(preds_hardcoded),
        "mae_walk_forward_fit": mae(preds_fit),
        "avg_fit_intercept": round(float(avg_coef[0]), 2) if avg_coef is not None else None,
        "avg_fit_total_coef": round(float(avg_coef[1]), 3) if avg_coef is not None else None,
        "avg_fit_spread_coef": round(float(avg_coef[2]), 3) if avg_coef is not None else None,
        "current_hardcoded_total_coef": 1.7,
        "current_hardcoded_spread_coef": 1.0,
    }


def _regime_blend_preds(results, lg_avg, by_bucket):
    """Final mean exactly as production builds it: per-regime XGB weight."""
    buckets = classify_amplify_buckets(results, lg_avg)
    w_of = {}
    for name, rs in buckets.items():
        for r in rs:
            w_of[id(r)] = by_bucket[name]["blend"]
    return np.array([w_of[id(r)] * r["xgb_pred"] + (1 - w_of[id(r)]) * r["ewma_pred"] for r in results])


def mode_comparison(results_by_mode):
    """Before/after table: the same OOS rows and fold models, scored under each
    adjustment mode. 'hardcoded' is what production did pre-v7; 'fitted' is v7.
    Over/under metrics use the raw baseline as the stand-in line (_line) for
    every mode. over_rate_pred vs over_rate_actual shows the over-skew."""
    out = {}
    for mode, res in results_by_mode.items():
        y = np.array([r["y"] for r in res])
        line = np.array([_line(r) for r in res])
        table, best = sweep_blend(res)
        lg = league_avg_yds(res)
        by_bucket = sweep_blend_by_bucket(res, lg, best)
        prod = _regime_blend_preds(res, lg, by_bucket)
        cal = sweep_calibration(res, prod, live_stds(res, residual_std_by_bucket(res, best)))
        dacc = directional_accuracy(res, best)
        probs = np.array([m.prob_over(mu, r["hist_std"], _line(r)) for mu, r in zip(prod, res)])
        act = (y > line).astype(float)
        out[mode] = {
            "best_global_blend": best,
            "mae_global_blend": round(table[f"{best:.2f}"]["mae"], 2),
            "rmse_global_blend": round(table[f"{best:.2f}"]["rmse"], 2),
            "regime_blends": {k: v["blend"] for k, v in by_bucket.items()},
            "mae_regime_blend": round(float(np.mean(np.abs(prod - y))), 2),
            "rmse_regime_blend": round(float(np.sqrt(np.mean((prod - y) ** 2))), 2),
            "mean_error_regime_blend": round(float(np.mean(prod - y)), 2),
            "mean_error_baseline": round(float(np.mean([r["ewma_pred"] - r["y"] for r in res])), 2),
            "mean_error_xgb": round(float(np.mean([r["xgb_pred"] - r["y"] for r in res])), 2),
            "dir_acc_blend": dacc["blend_vs_baseline"]["accuracy"],
            "dir_acc_blend_n": dacc["blend_vs_baseline"]["n_scored"],
            "brier_raw": cal["normal_alpha_1"]["brier"],
            "best_alpha": cal["best_alpha"],
            "best_distribution": pc.dist_label(cal["best_distribution"]),
            "brier_best_alpha": cal["chosen"]["brier"],
            "brier_regime_blend_raw": round(float(np.mean((probs - act) ** 2)), 4),
            "over_rate_pred_regime_blend": round(float(np.mean(probs)), 3),
            "over_rate_actual": round(float(np.mean(act)), 3),
        }
    return out


def baseline_component_comparison(rows):
    """Checks whether EWMA_BLEND/MEDIAN_BLEND/ATTYA_BLEND (40/35/25) are
    double-counting correlated signal, and whether splitting yards into
    attempts x efficiency (the ATTYA term) earns its weight — both against
    real outcomes, using every leak-free pregame row (no fitting/training
    happens here, so there's no OOS-split requirement the way there is for
    XGB or the Vegas OLS fit above; each row's components were already built
    from strictly-prior games only)."""
    y    = np.array([r["y"] for r in rows])
    ewma = np.array([r["ewma_yds_only"] for r in rows])
    med  = np.array([r["median_yds_only"] for r in rows])
    atty = np.array([r["attya_only"] for r in rows])

    def mae(pred):
        return round(float(np.mean(np.abs(pred - y))), 2)

    out = {
        "ewma_only":          mae(ewma),
        "median_only":        mae(med),
        "attya_only":         mae(atty),
        "current_blend_40_35_25": mae(0.40 * ewma + 0.35 * med + 0.25 * atty),
        "corr_ewma_vs_median":  round(float(np.corrcoef(ewma, med)[0, 1]), 3),
        "corr_ewma_vs_attya":   round(float(np.corrcoef(ewma, atty)[0, 1]), 3),
        "corr_median_vs_attya": round(float(np.corrcoef(med, atty)[0, 1]), 3),
    }

    best_mae, best_w = None, None
    grid = [i / 20 for i in range(21)]
    for we in grid:
        for wm in grid:
            wa = 1.0 - we - wm
            if wa < 0:
                continue
            pred = we * ewma + wm * med + wa * atty
            cur = float(np.mean(np.abs(pred - y)))
            if best_mae is None or cur < best_mae:
                best_mae, best_w = cur, (we, wm, round(wa, 2))
    out["grid_search_best_weights_ewma_median_attya"] = best_w
    out["grid_search_best_mae"] = round(best_mae, 2)
    return out


def main():
    print("Loading team data...")
    data = m.load_all_team_data()

    print("Building historical rows (leak-free, pregame-only features)...")
    rows = build_rows(data)
    print(f"  {len(rows)} historical QB-game rows across {len(set(r['qb'] for r in rows))} QBs")
    if len(rows) < m.MIN_XGB_SAMPLES * 2:
        raise SystemExit(
            f"Only {len(rows)} rows available — need at least {m.MIN_XGB_SAMPLES * 2} "
            f"for a meaningful walk-forward split. Add more historical game logs and re-run."
        )

    print(f"Running {K_FOLDS}-fold expanding-window walk-forward (none / hardcoded / fitted adjustment)...")
    results_by_mode, fold_report = walk_forward_all_modes(rows)
    results = results_by_mode["fitted"]   # v7 production mode — everything below is tuned on it
    for f in fold_report:
        c = f["fitted_coefs"]
        print(f"  fold {f['fold']}: train={f['train_rows']} test={f['test_rows']} "
              f"train_end={f['train_end_date']} test_range={f['test_date_range']}  "
              f"fit: bias={c['baseline_bias']:+.1f} xgb_bias={c['xgb_bias']:+.1f} "
              f"total={c['total_coef']:+.2f} spread={c['spread_coef']:+.2f}")
    if not results:
        raise SystemExit("Walk-forward produced no out-of-sample rows — not enough history yet.")
    print(f"  {len(results)} pooled out-of-sample predictions")

    previous = None
    try:
        with open(RESULTS_PATH) as f:
            old = json.load(f)
        if old.get("best_calibration_alpha") is not None:
            previous = (old.get("best_distribution") or dict(pc.NORMAL), float(old["best_calibration_alpha"]))
    except (OSError, ValueError):
        pass

    print("Before/after: same OOS rows and fold models under each adjustment mode "
          "(line = raw pre-Vegas baseline)...")
    comparison = mode_comparison(results_by_mode)
    hdr = ["mode", "blend", "MAE", "RMSE", "MAE(reg)", "RMSE(reg)", "err(reg)", "dir acc", "Brier", "Brier(reg)", "P(O) pred/act"]
    print("  " + " | ".join(hdr))
    for mode, c in comparison.items():
        print(f"  {mode:9s} | {c['best_global_blend']:.2f} | {c['mae_global_blend']:.2f} | {c['rmse_global_blend']:.2f} | "
              f"{c['mae_regime_blend']:.2f} | {c['rmse_regime_blend']:.2f} | {c['mean_error_regime_blend']:+.1f} | "
              f"{(c['dir_acc_blend'] or 0)*100:.1f}% (n={c['dir_acc_blend_n']}) | {c['brier_raw']:.4f} | "
              f"{c['brier_regime_blend_raw']:.4f} | {c['over_rate_pred_regime_blend']:.3f}/{c['over_rate_actual']:.3f}")

    print("Fitting live game-context coefficients on ALL rows (what qb_gui_V4 will use)...")
    X_all = np.array([r["feat"] for r in rows], dtype=np.float32)
    w_all = np.array([m.LOW_VOL_SAMPLE_WEIGHT if r["low_vol"] else 1.0 for r in rows], dtype=np.float32)
    full_model = _new_model()
    full_model.fit(X_all, np.array([r["y"] for r in rows], dtype=np.float32), sample_weight=w_all)
    game_coefs = fit_game_coefs(rows, full_model.predict(X_all))
    print(f"  baseline_bias={game_coefs['baseline_bias']:+.2f} (se {game_coefs['baseline_bias_se']:.2f})  "
          f"xgb_bias={game_coefs['xgb_bias']:+.2f}  "
          f"total_coef={game_coefs['total_coef']:+.3f} (se {game_coefs['total_coef_se']:.3f})  "
          f"spread_coef={game_coefs['spread_coef']:+.3f} (se {game_coefs['spread_coef_se']:.3f})  n={game_coefs['n']}")

    print("Sweeping XGB/EWMA blend ratio on out-of-sample MAE/RMSE...")
    blend_table, best_blend = sweep_blend(results)
    print(f"  best blend (min RMSE): XGB_BLEND={best_blend:.2f} "
          f"(MAE={blend_table[f'{best_blend:.2f}']['mae']:.1f}, "
          f"RMSE={blend_table[f'{best_blend:.2f}']['rmse']:.1f})")
    print(f"  for reference — EWMA only (0.00): MAE={blend_table['0.00']['mae']:.1f}, RMSE={blend_table['0.00']['rmse']:.1f}")
    print(f"                  XGB only  (1.00): MAE={blend_table['1.00']['mae']:.1f}, RMSE={blend_table['1.00']['rmse']:.1f}")

    print("Directional accuracy vs. EWMA-baseline-as-line (non-circular, but not a real market line)...")
    dacc = directional_accuracy(results, best_blend)
    for k, v in dacc.items():
        if v["accuracy"] is not None:
            print(f"  {k}: {v['accuracy']*100:.1f}% ({v['n_scored']} rows with edge >= {MIN_EDGE_YDS} yds)")
        else:
            print(f"  {k}: no rows scored")

    print("Computing out-of-sample residual std by attempt-volume bucket...")
    resid_buckets = residual_std_by_bucket(results, best_blend)
    for k, v in resid_buckets.items():
        print(f"  {k}: std={v}")

    print("Checking whether XGB amplifies or corrects extreme EWMA baselines (vs. real outcomes)...")
    lg_avg = league_avg_yds(results)
    amp = amplify_vs_correct_diagnostic(results, best_blend, lg_avg)
    for k, v in amp.items():
        if v["n"] == 0:
            print(f"  {k}: no rows")
        else:
            verdict = "blend HELPS" if v["blend_helps"] else "blend HURTS"
            print(f"  {k} (n={v['n']}): EWMA-only MAE={v['ewma_only_mae']}, blend MAE={v['blend_mae']} -> {verdict}")

    print("Comparing EWMA/median/Att×Y-A baseline components (double-counting / weight check)...")
    base_check = baseline_component_comparison(rows)
    print(f"  EWMA-only MAE={base_check['ewma_only']}  median-only MAE={base_check['median_only']}  "
          f"Att×Y/A-only MAE={base_check['attya_only']}  current-blend MAE={base_check['current_blend_40_35_25']}")
    print(f"  pairwise correlation: ewma/median={base_check['corr_ewma_vs_median']}, "
          f"ewma/attya={base_check['corr_ewma_vs_attya']}, median/attya={base_check['corr_median_vs_attya']}")
    print(f"  grid-search best (ewma,median,attya) weights={base_check['grid_search_best_weights_ewma_median_attya']} "
          f"MAE={base_check['grid_search_best_mae']} (current 0.40/0.35/0.25 = {base_check['current_blend_40_35_25']})")

    print("Checking apply_vegas() against real closing lines (if fetch_vegas_lines.py has run)...")
    vegas_check = vegas_adjustment_backtest(rows)
    if vegas_check.get("note"):
        print(f"  {vegas_check['note']} ({vegas_check.get('n_with_real_lines', 0)} rows had a real line)")
    else:
        print(f"  n={vegas_check['n_scored']} OOS rows with a real total/spread line")
        print(f"  MAE with NO vegas adjustment: {vegas_check['mae_no_vegas_adjustment']}")
        print(f"  MAE with current hardcoded (total*1.7, spread*1.0): {vegas_check['mae_hardcoded_formula']}")
        print(f"  MAE with walk-forward-fit coefficients: {vegas_check['mae_walk_forward_fit']} "
              f"(avg fit: total_coef={vegas_check['avg_fit_total_coef']}, "
              f"spread_coef={vegas_check['avg_fit_spread_coef']}, intercept={vegas_check['avg_fit_intercept']})")

    print("Re-sweeping blend ratio independently within each regime...")
    blend_by_bucket = sweep_blend_by_bucket(results, lg_avg, best_blend)
    for k, v in blend_by_bucket.items():
        if v.get("fell_back_to_global"):
            print(f"  {k} (n={v['n']}): too few rows, using global blend={v['blend']:.2f}")
        else:
            print(f"  {k} (n={v['n']}): best blend={v['blend']:.2f} "
                  f"(MAE={v['mae_at_best_blend']:.1f} vs. MAE={v['mae_at_global_blend']:.1f} at global blend)")

    print("Sweeping over/under distribution + calibration alpha (Brier at offset lines, "
          "live regime-blend mean and std)...")
    cal = sweep_calibration(results, _regime_blend_preds(results, lg_avg, blend_by_bucket),
                            live_stds(results, resid_buckets), previous)
    pc.print_sweep("pass_yds", cal)

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": m.MODEL_VERSION,
        "model_fingerprint": m._model_config_fingerprint(),
        "n_rows": len(rows),
        "n_oos_predictions": len(results),
        "k_folds": K_FOLDS,
        "best_xgb_blend": best_blend,
        "best_calibration_alpha": float(cal["best_alpha"]),
        "best_distribution": cal["best_distribution"],
        "calibration_sweep": cal,
        "residual_std_by_bucket": resid_buckets,
        "league_avg_yds": lg_avg,
        "amplify_dev_threshold_yds": AMPLIFY_DEV_THRESHOLD_YDS,
        "amplify_vs_correct": amp,
        "best_xgb_blend_by_bucket": blend_by_bucket,
        "game_context_coefs": game_coefs,
        "adjustment_mode_comparison": comparison,
        "vegas_adjustment_check": vegas_check,
        "baseline_component_comparison": base_check,
        "blend_sweep_mae_rmse": blend_table,
        "directional_accuracy_vs_baseline": dacc,
        "fold_report": fold_report,
        "limitations": (
            "No historical sportsbook lines exist in this repo; directional accuracy is "
            "tested against the EWMA-only baseline's own forecast as a stand-in line; "
            "over/under calibration (distribution + alpha) is scored at synthetic lines "
            "mean + k*std, k in -1..+1 (prop_calibration), not real market lines. Opponent defense features use a "
            "per-game, date-lagged proxy built from QB box scores (real weekly "
            "granularity, leak-free). Personnel-package features were dropped entirely "
            "(no free, ToS-clean source). LOW_VOL_SAMPLE_WEIGHT is still a hand-picked "
            "0.35, not itself walk-forward-validated."
        ),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}  (model_version={m.MODEL_VERSION}, fingerprint={out['model_fingerprint']})")
    print("qb_gui_V4.py will verify that fingerprint matches its current config before "
          "trusting best_xgb_blend / best_calibration_alpha / residual_std_by_bucket — "
          "if you change feature logic or key constants without re-running this script, "
          "it will detect the mismatch and fall back to hardcoded defaults instead of "
          "silently applying stale results.")


if __name__ == "__main__":
    main()
