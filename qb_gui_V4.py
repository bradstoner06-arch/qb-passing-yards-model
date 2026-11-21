import os
import glob
import math
from bisect import bisect_left
import numpy as np

import prop_calibration as pc
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =========================
# Paths
# =========================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
QB_DIR_2025 = os.path.join(BASE_DIR, "QB_Stats_2025")
QB_DIR_2026 = os.path.join(BASE_DIR, "QB_Stats_2026")
# Written by fetch_nflverse_qb_stats.py — one file per season, real QB weekly
# stats pulled from nflverse instead of manually downloading game logs from
# PFR. Every nflverse_qb_weekly_*.parquet file found is merged (so pulling
# extra history, e.g. 2024, for XGB training is just running the fetch script
# again with --year 2024 — no code change needed). For 2026 specifically this
# is preferred over the legacy per-QB .xls files; those are still the fallback
# if this hasn't been generated yet. 2025 keeps using the existing PFR exports.
NFLVERSE_QB_CACHE_GLOB = os.path.join(BASE_DIR, "nflverse_qb_weekly_*.parquet")

DEF_STATS_2025 = os.path.join(BASE_DIR, "DEF_Stats_2025.xlsx")
DEF_STATS_2026 = os.path.join(BASE_DIR, "DEF_Stats_2026.xlsx")
OFF_STATS_2025 = os.path.join(BASE_DIR, "OFF_Stats_2025.xlsx")
OFF_STATS_2026 = os.path.join(BASE_DIR, "OFF_Stats_2026.xlsx")
# Written by fetch_nflverse_team_stats.py — real per-team EPA/Pass, Sack %,
# ADoT pulled from nflverse instead of manually downloading DEF/OFF_Stats_2026
# from SumerSports. Preferred over those legacy files when present.
NFLVERSE_TEAM_STATS_2026 = os.path.join(BASE_DIR, "nflverse_team_stats_2026.parquet")
# Personnel-package tendency data (formations) is no longer used — the only
# public free source (SumerSports) turned out to require a paid subscription,
# and nflverse doesn't publish personnel groupings. Dropped rather than scraped.

# =========================
# Model Configuration
# =========================
ROLLING_GAMES = 8
EWMA_DECAY    = 0.85
MIN_ATT       = 15     # below this a game is flagged "low volume" — downweighted, never deleted (see LOW_VOL_SAMPLE_WEIGHT)
LOW_VOL_SAMPLE_WEIGHT = 0.35   # XGB sample_weight for low-volume games instead of dropping them (avoids survivorship bias)

# WEIGHT_2026_TEAM is now computed dynamically per-call from how many 2026 games
# have actually been played league-wide — see _dynamic_weight_2026(). These are
# just the floor/ceiling of that ramp (early season -> mostly 2025, late season -> mostly 2026).
WEIGHT_2026_MIN = 0.20
WEIGHT_2026_MAX = 0.90
WEIGHT_2026_SATURATION_GAMES = 10.0   # games played at which the ramp reaches WEIGHT_2026_MAX

CALIBRATION_ALPHA = 0.7
# Pass-yds over/under distribution; backtest_qb_model.py picks Normal vs shifted
# gamma jointly with CALIBRATION_ALPHA at a spread of line offsets (v8).
YDS_DIST = dict(pc.NORMAL)

DEF_PASS_WEIGHT = 12.0
OFF_PASS_WEIGHT = 10.0
DEF_SACK_WEIGHT = 160.0
OFF_SACK_WEIGHT = 90.0
DEF_ADOT_WEIGHT = 5.0
OFF_ADOT_WEIGHT = 5.0

DEFAULT_STD  = 48.0
STD_MIN      = 15.0
# STD_MAX is defined below, in the "Backtest overrides" section — it starts at
# a generous hardcoded default and is raised further once backtest_qb_model.py
# has measured real out-of-sample residual std.

# Baseline blend weights (must sum to 1.0):
#   EWMA is recency-weighted but can be dragged by outlier low-attempt games
#   Median is outlier-resistant and anchors the projection to the QB's typical output
#   Att×Y/A captures volume × efficiency as a cross-check
EWMA_BLEND   = 0.40   # EWMA of rolling Yds
MEDIAN_BLEND = 0.35   # median of rolling window — resistant to 1-2 game-script outliers
ATTYA_BLEND  = 0.25   # Att × Y/A efficiency baseline

# XGBoost blend: how much weight to give XGB vs EWMA
XGB_BLEND       = 0.70
MIN_XGB_SAMPLES = 50

# Amplify-vs-correct regime split: instead of one flat XGB_BLEND, weight XGB
# differently depending on whether its prediction would amplify an already-
# extreme EWMA baseline (reinforcing the "bad QB under / good QB over"
# pattern every week) or correct it back toward league average. Measured
# directly by backtest_qb_model.py's sweep_blend_by_bucket: amplifying picks
# did WORSE with XGB blended in than with EWMA alone, while correcting picks
# did meaningfully better. Until a backtest has run, all three default to
# XGB_BLEND — i.e. behave exactly like the old flat blend.
LEAGUE_AVG_YDS            = None
AMPLIFY_DEV_THRESHOLD_YDS = 15.0
XGB_BLEND_AMPLIFY = XGB_BLEND
XGB_BLEND_CORRECT = XGB_BLEND
XGB_BLEND_NEUTRAL = XGB_BLEND

# Sanity guardrail: XGB's raw prediction can't diverge from the pure EWMA/
# median/Att×Y/A baseline by more than this many yards before blending. With
# only ~190 training rows and 13 features, XGB can carve a leaf from a
# handful of games sharing an uncommon feature combination and extrapolate
# far outside anything a real defense justifies — observed directly: for one
# QB, predicted yards ranged 107-278 across the league's 32 defenses, and the
# ranking wasn't even monotonic in the defensive inputs (a nominally weaker
# pass defense produced a LOWER prediction than a stronger one). This doesn't
# fix the overfitting at the source, but stops a single unstable leaf from
# dominating the blended projection.
MAX_XGB_DEVIATION_YDS = 60.0

# Game-context adjustment (apply_vegas): mean + bias + TOTAL×(total−44) + SPREAD×spread.
# These defaults are the pre-v7 hand-set values and are used only when
# backtest_results.json is missing or STALE. v7 replaces them with
# walk-forward-validated OLS fits (backtest_qb_model.fit_game_coefs):
#   - a 2024-26 study found the spread slope ~−0.2 ± 0.45, not +1.0 (favorites
#     were being docked yards), and
#   - the baseline over-projects by ~9 yds on average, almost all of it from
#     <15-att outcomes (injury exits, benchings, Week 18 rests) that
#     usable_games() keeps out of the averages. Those props still settle, so
#     every P(over) was skewed high.
# XGB gets its own (smaller) bias: it's trained on those games at 0.35 weight.
VEGAS_TOTAL_COEF  = 1.7
VEGAS_SPREAD_COEF = 1.0
BASELINE_BIAS_YDS = 0.0
XGB_BIAS_YDS      = 0.0

# PFR abbreviation → full team name
TEAM_ABB = {
    "ARI": "Arizona Cardinals",    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",     "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",   "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",       "DEN": "Denver Broncos",
    "DET": "Detroit Lions",        "GNB": "Green Bay Packers",
    "HOU": "Houston Texans",       "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KAN": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LVR": "Las Vegas Raiders",    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",    "NWE": "New England Patriots",
    "NOR": "New Orleans Saints",   "NYG": "New York Giants",
    "NYJ": "New York Jets",        "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",  "SFO": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",     "TAM": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",     "WAS": "Washington Commanders",
    "OAK": "Las Vegas Raiders",    "SDG": "Los Angeles Chargers",
    "STL": "Los Angeles Rams",
    # nflverse uses different codes than PFR for a handful of teams
    # (see fetch_nflverse_qb_stats.py) — both need to resolve correctly.
    "GB":  "Green Bay Packers",    "KC":  "Kansas City Chiefs",
    "NE":  "New England Patriots", "NO":  "New Orleans Saints",
    "SF":  "San Francisco 49ers",  "TB":  "Tampa Bay Buccaneers",
    "LV":  "Las Vegas Raiders",    "LA":  "Los Angeles Rams",
}

QB_FEATURE_NAMES = [
    "roll_yds", "roll_att", "roll_ya", "roll_cmp_pct",
    "def_yds_allowed_pg", "def_ya_allowed", "def_cmp_pct_allowed",
    "def_sack_rate_allowed", "def_td_allowed_pg", "def_int_allowed_pg",
    "n_games",
]

# Same shape/order as QB_FEATURE_NAMES, but the QB's own rolling stat in slot 0
# is TD rate instead of yards — everything else (opponent defense features,
# n_games) is identical and reused as-is via _qb_features(..., target="td").
TD_FEATURE_NAMES = [
    "roll_td", "roll_att", "roll_ya", "roll_cmp_pct",
    "def_yds_allowed_pg", "def_ya_allowed", "def_cmp_pct_allowed",
    "def_sack_rate_allowed", "def_td_allowed_pg", "def_int_allowed_pg",
    "n_games",
]

# =========================
# Backtest overrides (with staleness protection)
# =========================
# MODEL_VERSION must be bumped by hand whenever build_xgb_training_data's
# feature/label logic changes in a way that isn't captured by the config
# constants hashed below (e.g. switching the defensive features from
# season-aggregate Excel stats to the per-game lagged proxy). A stored
# backtest_results.json is only trusted if it was generated under the exact
# same fingerprint — otherwise its XGB_BLEND / CALIBRATION_ALPHA / residual
# std would be silently misapplied to a model that no longer resembles the
# one they were measured on.
# v7: apply_vegas slopes + baseline/XGB bias are fit in the backtest (game_context_coefs).
# v8: CALIBRATION_ALPHA and YDS_DIST chosen by Brier at a spread of line offsets around
#     the projection (prop_calibration), not at the projection itself (degenerate).
MODEL_VERSION = "v8-offset-calibration"

BACKTEST_RESULTS_PATH = os.path.join(BASE_DIR, "backtest_results.json")
RESIDUAL_STD_BY_BUCKET = {}
# Hardcoded ceiling used only until a backtest has run. It's deliberately
# generous (not calibrated) — the first backtest found real OOS residual std
# of 80-94 yds under the old feature set, so a "safe-looking" 65 was actively
# clipping evidence of real uncertainty. Once backtest_qb_model.py runs,
# STD_MAX is raised further if the measured residuals call for it.
STD_MAX = 90.0


def _model_config_fingerprint():
    import hashlib
    payload = repr((
        MODEL_VERSION, tuple(QB_FEATURE_NAMES), ROLLING_GAMES, EWMA_DECAY, MIN_ATT,
        LOW_VOL_SAMPLE_WEIGHT, EWMA_BLEND, MEDIAN_BLEND, ATTYA_BLEND, MAX_XGB_DEVIATION_YDS,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_backtest_overrides():
    """If backtest_qb_model.py has been run under the CURRENT model
    configuration, use its walk-forward, out-of-sample results in place of
    the hand-picked guesses above. If backtest_results.json exists but was
    generated under a different feature set/config (fingerprint mismatch —
    e.g. produced before this file's leakage/feature-set changes), it is
    ignored rather than silently misapplied, and the mismatch is reported."""
    global XGB_BLEND, CALIBRATION_ALPHA, RESIDUAL_STD_BY_BUCKET, STD_MAX, YDS_DIST
    global LEAGUE_AVG_YDS, AMPLIFY_DEV_THRESHOLD_YDS
    global XGB_BLEND_AMPLIFY, XGB_BLEND_CORRECT, XGB_BLEND_NEUTRAL
    global VEGAS_TOTAL_COEF, VEGAS_SPREAD_COEF, BASELINE_BIAS_YDS, XGB_BIAS_YDS

    if not os.path.isfile(BACKTEST_RESULTS_PATH):
        return "No backtest_results.json found — using hardcoded defaults. Run backtest_qb_model.py."
    try:
        import json
        with open(BACKTEST_RESULTS_PATH, "r") as f:
            d = json.load(f)
    except Exception as e:
        return f"backtest_results.json unreadable ({e}) — using hardcoded defaults."

    current_fp = _model_config_fingerprint()
    stored_fp  = d.get("model_fingerprint")
    if stored_fp != current_fp:
        return (f"backtest_results.json is STALE (generated under a different model "
                f"version/config: {d.get('model_version', '?')}/{stored_fp} vs current "
                f"{MODEL_VERSION}/{current_fp}) — ignoring it, using hardcoded defaults. "
                f"Re-run backtest_qb_model.py.")

    if d.get("best_xgb_blend") is not None:
        XGB_BLEND = float(d["best_xgb_blend"])
        XGB_BLEND_AMPLIFY = XGB_BLEND_CORRECT = XGB_BLEND_NEUTRAL = XGB_BLEND
    if d.get("best_calibration_alpha") is not None:
        CALIBRATION_ALPHA = float(d["best_calibration_alpha"])
    if d.get("best_distribution"):
        YDS_DIST = dict(d["best_distribution"])
    RESIDUAL_STD_BY_BUCKET = d.get("residual_std_by_bucket") or {}
    if RESIDUAL_STD_BY_BUCKET:
        STD_MAX = max(STD_MAX, min(150.0, max(RESIDUAL_STD_BY_BUCKET.values()) * 1.25))

    if d.get("league_avg_yds") is not None:
        LEAGUE_AVG_YDS = float(d["league_avg_yds"])
    if d.get("amplify_dev_threshold_yds") is not None:
        AMPLIFY_DEV_THRESHOLD_YDS = float(d["amplify_dev_threshold_yds"])
    bucket_blend = d.get("best_xgb_blend_by_bucket") or {}
    if bucket_blend.get("amplify", {}).get("blend") is not None:
        XGB_BLEND_AMPLIFY = float(bucket_blend["amplify"]["blend"])
    if bucket_blend.get("correct", {}).get("blend") is not None:
        XGB_BLEND_CORRECT = float(bucket_blend["correct"]["blend"])
    if bucket_blend.get("neutral", {}).get("blend") is not None:
        XGB_BLEND_NEUTRAL = float(bucket_blend["neutral"]["blend"])

    gc = d.get("game_context_coefs") or {}
    if all(gc.get(k) is not None for k in ("total_coef", "spread_coef", "baseline_bias", "xgb_bias")):
        VEGAS_TOTAL_COEF  = float(gc["total_coef"])
        VEGAS_SPREAD_COEF = float(gc["spread_coef"])
        BASELINE_BIAS_YDS = float(gc["baseline_bias"])
        XGB_BIAS_YDS      = float(gc["xgb_bias"])

    return (f"Loaded backtest results from {d.get('generated_at', '?')}: "
            f"XGB_BLEND={XGB_BLEND:.2f}, CALIBRATION_ALPHA={CALIBRATION_ALPHA:.2f} ({pc.dist_label(YDS_DIST)}), "
            f"STD_MAX={STD_MAX:.0f}, XGB_BLEND[amplify/correct/neutral]="
            f"{XGB_BLEND_AMPLIFY:.2f}/{XGB_BLEND_CORRECT:.2f}/{XGB_BLEND_NEUTRAL:.2f}, "
            f"Vegas total/spread={VEGAS_TOTAL_COEF:+.2f}/{VEGAS_SPREAD_COEF:+.2f}, "
            f"bias base/XGB={BASELINE_BIAS_YDS:+.1f}/{XGB_BIAS_YDS:+.1f} yds")


BACKTEST_STATUS = _load_backtest_overrides()

# =========================
# TD-passes backtest overrides — separate model, separate results file.
# Mirrors the yards section above exactly (same staleness-protection idea),
# just with its own fingerprint/JSON so the two models' calibration data
# never get conflated.
# =========================
TD_MODEL_VERSION = "td-v1"
TD_XGB_BLEND = 0.50
TD_CALIBRATION_ALPHA = 0.7
TD_BACKTEST_RESULTS_PATH = os.path.join(BASE_DIR, "backtest_results_td.json")


def _td_model_config_fingerprint():
    import hashlib
    payload = repr((
        TD_MODEL_VERSION, tuple(TD_FEATURE_NAMES), ROLLING_GAMES, EWMA_DECAY, MIN_ATT,
        LOW_VOL_SAMPLE_WEIGHT,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_td_backtest_overrides():
    """Same idea as _load_backtest_overrides(), for the TD-passes model. No
    STD_MAX equivalent — Poisson's variance is tied to its mean (lambda), so
    there's no separate spread parameter to calibrate."""
    global TD_XGB_BLEND, TD_CALIBRATION_ALPHA

    if not os.path.isfile(TD_BACKTEST_RESULTS_PATH):
        return "No backtest_results_td.json found — using hardcoded defaults. Run backtest_qb_model_td.py."
    try:
        import json
        with open(TD_BACKTEST_RESULTS_PATH, "r") as f:
            d = json.load(f)
    except Exception as e:
        return f"backtest_results_td.json unreadable ({e}) — using hardcoded defaults."

    current_fp = _td_model_config_fingerprint()
    stored_fp  = d.get("model_fingerprint")
    if stored_fp != current_fp:
        return (f"backtest_results_td.json is STALE (generated under a different model "
                f"version/config: {d.get('model_version', '?')}/{stored_fp} vs current "
                f"{TD_MODEL_VERSION}/{current_fp}) — ignoring it, using hardcoded defaults. "
                f"Re-run backtest_qb_model_td.py.")

    if d.get("best_xgb_blend") is not None:
        TD_XGB_BLEND = float(d["best_xgb_blend"])
    if d.get("best_calibration_alpha") is not None:
        TD_CALIBRATION_ALPHA = float(d["best_calibration_alpha"])
    return (f"Loaded TD backtest results from {d.get('generated_at', '?')}: "
            f"TD_XGB_BLEND={TD_XGB_BLEND:.2f}, TD_CALIBRATION_ALPHA={TD_CALIBRATION_ALPHA:.2f}")


TD_BACKTEST_STATUS = _load_td_backtest_overrides()

# =========================
# Helpers
# =========================
def to_float(x, default=None):
    try:
        s = str(x).strip().replace(",", "").replace("%", "").replace("*", "")
        if s in ("", "nan", "none", "-"):
            return default
        return float(s)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def calibrate_probability(p, alpha=None):
    """alpha=None -> the CURRENT CALIBRATION_ALPHA (looked up at call time, so a
    backtest override or a test can't be shadowed by a default bound at import)."""
    if alpha is None:
        alpha = CALIBRATION_ALPHA
    eps = 1e-6
    p = clamp(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)) * alpha
    return 1 / (1 + np.exp(-logit))

def resolve_team(abbr):
    return TEAM_ABB.get(str(abbr).strip().upper(), str(abbr).strip())

# Columns that must be stored as fractions (0.0-1.0), never raw percents (0-100).
# The whole model assumes this (DEF_SACK_WEIGHT=160 etc. are calibrated against
# fractions) — if a future data refresh exports "6.5" instead of "0.065" for one
# of these, every downstream adjustment silently blows up by ~100x. Guard it.
FRACTION_COLUMNS = {"Sack %", "Comp %", "Success %", "Scramble %", "Int %",
                     "Rate", "Pass Rate", "League Avg. Usage"}

def _sanity_fix_fractions(records, source_label=""):
    """records: dict[team] -> dict[col] -> value (or list of such dicts, for personnel).
    If a fraction-like column looks like it's stored as a raw percent (median > 1.5
    across teams), divide the whole column by 100 and warn — instead of silently
    producing adjustments ~100x too large."""
    flat = []
    for v in records.values():
        flat.extend(v if isinstance(v, list) else [v])
    for col in FRACTION_COLUMNS:
        vals = [to_float(row.get(col)) for row in flat if to_float(row.get(col)) is not None]
        if len(vals) < 3:
            continue
        if float(np.median(vals)) > 1.5:
            print(f"[WARN] {source_label}: column '{col}' looks like a raw percent "
                  f"(median={np.median(vals):.2f}), not a fraction — dividing by 100.")
            for row in flat:
                if col in row and row[col] is not None:
                    v = to_float(row[col])
                    if v is not None:
                        row[col] = v / 100.0
    return records

# =========================
# Weather
# =========================
def apply_temp_adjustment(mean, std, temp_f):
    t = float(temp_f)
    if t < 45:
        mean *= 1.0 - 0.10 * clamp((45.0 - t) / 45.0, 0.0, 1.0)
    if t < 32:
        std  *= 1.0 + 0.08 * clamp((32.0 - t) / 32.0, 0.0, 1.0)
    return mean, std

def apply_weather(mean, std, wind, temp, rain, snow, indoor):
    if not indoor:
        # Wind: ~0.5 yds/mph up to ~15 mph, then diminishing — roughly linear at 0.5
        # Research: 20mph costs ~10 yds, 30mph ~15 yds (not linear — cap the rate above 20)
        wind_adj = min(wind, 20) * 0.5 + max(wind - 20, 0) * 0.25
        mean -= wind_adj
        if rain:
            mean *= 0.97; std *= 1.04   # light-moderate rain: ~3% reduction
        if snow:
            mean *= 0.95; std *= 1.07   # snow: ~5% reduction (heavier effect captured by wind+cold combo)
        mean, std = apply_temp_adjustment(mean, std, temp)
    return mean, std

# =========================
# QB Game Log Reader
# =========================
def _read_qb_game_log(path, season=None):
    """Keeps every game with a real pass attempt — including sub-MIN_ATT games.
    Deleting low-attempt games before XGB ever sees them (the old behavior)
    doesn't just exclude them from the QB's rolling average, it also removes
    them as XGB *targets*, which biases the model toward normal-volume games
    and inflates expected yards for props. Low-volume games are flagged
    ("low_vol") and downweighted (LOW_VOL_SAMPLE_WEIGHT) instead of dropped."""
    dfs = pd.read_html(path, header=[0, 1])
    df  = dfs[0]
    flat = []
    seen = {}
    for top, bot in df.columns:
        name = str(bot).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        flat.append(name)
    df.columns = flat
    required = {"Rk", "Date", "Opp", "Att", "Yds"}
    missing = required - set(df.columns)
    if missing:
        # Usually means the wrong Pro-Football-Reference page was saved for
        # this file — e.g. the career year-by-year summary table instead of
        # the per-game "Game Log" page. Fail with a clear message instead of
        # a cryptic pandas indexing error several lines down.
        raise ValueError(
            f"{os.path.basename(path)} doesn't look like a game log — missing "
            f"column(s) {sorted(missing)}. Got columns: {list(df.columns)[:12]}... "
            f"Check that this file is the per-game Game Log export, not a season/career table."
        )
    df = df[pd.to_numeric(df["Rk"], errors="coerce").notna()].copy()
    games = []
    for _, r in df.iterrows():
        att = to_float(r.get("Att"), None)
        yds = to_float(r.get("Yds"), None)
        if att is None or att <= 0 or yds is None:
            continue
        games.append({
            "Date":    str(r.get("Date", "")).strip(),
            "Opp":     str(r.get("Opp", "")).strip().lstrip("@").strip(),
            "Att":     att,
            "Cmp":     to_float(r.get("Cmp"),  0.0),
            "Yds":     yds,
            "TD":      to_float(r.get("TD"),   0.0),
            "Int":     to_float(r.get("Int"),  0.0),
            "Cmp%":    to_float(r.get("Cmp%"), 0.0),
            "Y/A":     to_float(r.get("Y/A"),  0.0),
            "Sk":      to_float(r.get("Sk"),   0.0),
            "low_vol": att < MIN_ATT,
            "season":  season,
        })
    return games


_nflverse_cache = None   # lazy-loaded, process-wide: {qb_name: [game_dict, ...]}


def _load_nflverse_cache():
    """Reads every nflverse_qb_weekly_*.parquet file (written by
    fetch_nflverse_qb_stats.py, one per season fetched) once per process and
    groups them into the same game-dict shape _read_qb_game_log produces, so
    the rest of the pipeline can't tell the difference between a PFR-exported
    game and an nflverse one — or which season a given cache file covers.
    Returns {} if no cache files exist yet — callers fall back to the legacy
    per-QB .xls files in that case."""
    global _nflverse_cache
    if _nflverse_cache is not None:
        return _nflverse_cache
    _nflverse_cache = {}
    if not HAS_PANDAS:
        return _nflverse_cache
    paths = sorted(glob.glob(NFLVERSE_QB_CACHE_GLOB))
    if not paths:
        return _nflverse_cache
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return _nflverse_cache
    df = pd.concat(frames, ignore_index=True)
    for name, grp in df.groupby("player_display_name"):
        grp = grp.sort_values("Date")
        games = []
        for _, r in grp.iterrows():
            att = float(r["Att"])
            games.append({
                "Date":    str(r["Date"]),
                "Opp":     str(r["Opp"]),
                "Att":     att,
                "Cmp":     float(r["Cmp"]),
                "Yds":     float(r["Yds"]),
                "TD":      float(r["TD"]),
                "Int":     float(r["Int"]),
                "Cmp%":    float(r["Cmp%"]),
                "Y/A":     float(r["Y/A"]),
                "Sk":      float(r["Sk"]),
                "low_vol": att < MIN_ATT,
                "season":  int(r["season"]),
            })
        if games:
            _nflverse_cache[name] = games
    return _nflverse_cache


def list_qbs():
    names = set()
    for d in (QB_DIR_2025, QB_DIR_2026):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".xls") or f.endswith(".xlsx"):
                    names.add(f.rsplit(".", 1)[0])
    names |= set(_load_nflverse_cache().keys())
    return sorted(names)


def _find_qb_file(directory, qb_name):
    if not os.path.isdir(directory):
        return None
    for ext in (".xls", ".xlsx"):
        p = os.path.join(directory, qb_name + ext)
        if os.path.isfile(p):
            return p
    return None


def load_qb_games(qb_name):
    path_25 = _find_qb_file(QB_DIR_2025, qb_name)
    games_25 = []
    if path_25:
        try:
            games_25 = _read_qb_game_log(path_25, season=2025)
        except Exception as e:
            raise ValueError(f"Could not read 2025 game log for {qb_name}: {e}")

    nflverse_games = _load_nflverse_cache().get(qb_name, [])
    games_26_nflverse = [g for g in nflverse_games if g["season"] == 2026]
    # Any other cached season (e.g. 2024, fetched purely for extra XGB
    # training history — see fetch_nflverse_qb_stats.py --year 2024).
    games_extra = [g for g in nflverse_games if g["season"] != 2026]

    # 2026: prefer the nflverse cache (real weekly data, no manual downloads);
    # fall back to a legacy per-QB .xls file if this QB isn't in the cache
    # (e.g. fetch_nflverse_qb_stats.py hasn't been run yet).
    if games_26_nflverse:
        games_26 = games_26_nflverse
    else:
        path_26 = _find_qb_file(QB_DIR_2026, qb_name)
        games_26 = []
        if path_26:
            try:
                games_26 = _read_qb_game_log(path_26, season=2026)
            except Exception:
                pass

    # Dedup by Date: if a season is ever covered by BOTH a legacy PFR .xls
    # (games_25) AND an nflverse cache fetch (games_extra could pick up any
    # season via `fetch_nflverse_qb_stats.py --year <that season>`), the same
    # real game would otherwise be counted twice — once per source — silently
    # doubling its weight in the rolling average, std, and as an XGB training
    # row. A QB plays at most one game on a given date, so Date alone is a
    # safe dedup key. nflverse wins ties (it's the maintained, automated
    # source going forward); the PFR copy is dropped.
    by_date = {}
    for g in games_25:
        by_date[g["Date"]] = g
    for g in games_extra + games_26:
        by_date[g["Date"]] = g
    games = sorted(by_date.values(), key=lambda g: g["Date"])
    if not games:
        raise ValueError(f"No valid game log data found for: {qb_name}")
    seasons_present = sorted(set(g["season"] for g in games if g.get("season")))
    label = " + ".join(str(s) for s in seasons_present) if seasons_present else "unknown season"
    return games, label


def usable_games(games):
    """Low-volume (garbage-time / injury-shortened) games stay in the underlying
    data (see _read_qb_game_log) but are excluded from the averaging/display
    pool — they'd otherwise drag the baseline toward blowout/kneel-down
    performances. Falls back to the unfiltered list if a QB has nothing else."""
    return [g for g in games if not g.get("low_vol")] or games


def _std_bucket(att):
    """Attempt-volume bucket key, shared with backtest_qb_model.py's
    residual_std_by_bucket output so the live app's uncertainty estimate lines
    up with what was actually measured out-of-sample."""
    if att < 25:
        return "low_volume"
    if att < 35:
        return "mid_volume"
    return "high_volume"


def qb_rolling_avg(games):
    pool = usable_games(games)
    use = pool[-ROLLING_GAMES:] if len(pool) >= ROLLING_GAMES else pool
    n = len(use)
    w = np.array([EWMA_DECAY ** (n - 1 - i) for i in range(n)])
    w /= w.sum()
    keys = ("Att", "Cmp", "Yds", "TD", "Int", "Cmp%", "Y/A")
    return {k: float(np.dot(w, [g[k] for g in use])) for k in keys}

# =========================
# Team Data Loaders
# =========================
def _load_stats_excel(path):
    if not HAS_PANDAS:
        return {}
    try:
        df = pd.read_excel(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    result = {}
    rows = df.to_dict("records")
    i = 0
    while i < len(rows):
        tv = rows[i].get("Team", "")
        try:
            float(tv); is_data = True
        except (TypeError, ValueError):
            is_data = False
        if is_data and i + 1 < len(rows):
            name = str(rows[i + 1].get("Team", "")).strip()
            if name and name != "nan":
                result[name] = {k: v for k, v in rows[i].items() if k not in ("Team", "Season")}
                i += 2; continue
        i += 1
    return _sanity_fix_fractions(result, source_label=os.path.basename(path))


def _load_nflverse_team_stats(path, side):
    """Reads NFLVERSE_TEAM_STATS_2026 (written by fetch_nflverse_team_stats.py)
    and returns the same {team_full_name: {"EPA/Pass":.., "Sack %":.., "ADoT":..}}
    shape _load_stats_excel produces, filtered to one side (OFF or DEF), so
    load_all_team_data() can use either source interchangeably."""
    if not (HAS_PANDAS and os.path.isfile(path)):
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    df = df[df["side"] == side]
    result = {}
    for _, r in df.iterrows():
        name = resolve_team(r["Team"])
        result[name] = {"EPA/Pass": float(r["EPA/Pass"]), "Sack %": float(r["Sack %"]), "ADoT": float(r["ADoT"])}
    return result


def _blend_stats(d25, d26, w26=WEIGHT_2026_MAX):
    out = {}
    for team in set(d25) | set(d26):
        if team in d25 and team in d26:
            merged = {}
            for k in d25[team]:
                v25 = to_float(d25[team].get(k))
                v26 = to_float(d26[team].get(k))
                if v25 is not None and v26 is not None:
                    merged[k] = v26 * w26 + v25 * (1 - w26)
                elif v26 is not None:
                    merged[k] = v26
                else:
                    merged[k] = v25
            out[team] = merged
        elif team in d25:
            out[team] = d25[team]
        else:
            out[team] = d26[team]
    return out


def _league_avg(stats_dict, col):
    vals = [to_float(v.get(col)) for v in stats_dict.values() if to_float(v.get(col)) is not None]
    return float(np.mean(vals)) if vals else 0.0


# Sentinel "as of" date guaranteed to sort after every real game date (ISO
# strings), so defense_allowed_asof(..., PREDICT_ASOF_DATE) picks up every
# game on record — the correct thing to do for a live, still-in-the-future
# prediction, where "everything that's happened so far" IS the pregame state.
PREDICT_ASOF_DATE = "9999-12-31"

_DEFENSE_PROXY_KEYS = ("Att", "Yds", "Y/A", "Cmp%", "TD", "Int", "Sk")


def build_defense_allowed_index():
    """League-wide, per-defense, DATE-ORDERED cumulative passing stats allowed,
    built directly from every QB's game log (2025 and 2026 combined).

    This exists to close the leakage gap the season-aggregate Excel files
    can't: those files have no week-by-week split, so even after restricting
    training rows to the "fully resolved 2025 season" file, a Week 3 example
    still saw defensive numbers baked from Weeks 4-18. This index instead
    looks up, for a given (team, date), the average of ONLY the games that
    defense played strictly BEFORE that date — real week-by-week granularity,
    computed from data that already exists in QB_Stats_2025/2026.

    The same index and lookup function are used for historical training rows
    (asof = the target game's own date) AND live prediction (asof =
    PREDICT_ASOF_DATE, i.e. "everything known so far") — one code path, so
    train/predict can't drift out of sync the way the old EPA/Sack/ADoT
    features briefly did.

    Trade-off: this is raw box-score signal (yards/Y-per-A/Cmp%/sack-rate/
    TD/INT allowed per game), computed at the individual-game level from QB
    logs — distinct from NFLVERSE_TEAM_STATS_2026's real EPA, which is
    season-to-date only, not per-game-lagged.
    """
    per_team = {}
    for qb_name in list_qbs():
        try:
            games, _ = load_qb_games(qb_name)
        except Exception:
            continue
        for g in games:
            if not str(g.get("Date", "")).strip():
                continue
            team = resolve_team(g["Opp"])
            per_team.setdefault(team, []).append(g)

    index = {}
    for team, lst in per_team.items():
        lst.sort(key=lambda g: g["Date"])
        dates = [g["Date"] for g in lst]
        sums = {k: 0.0 for k in _DEFENSE_PROXY_KEYS}
        n = 0
        cums = []
        for g in lst:
            cums.append((dict(sums), n))   # snapshot BEFORE this game — strictly prior
            for k in _DEFENSE_PROXY_KEYS:
                sums[k] += to_float(g.get(k), 0.0) or 0.0
            n += 1
        cums.append((dict(sums), n))       # totals after every recorded game
        index[team] = {"dates": dates, "cums": cums}
    return index


def _defense_proxy_fallback(index):
    """League-average 'as of' proxy for a team with zero (or no) prior games —
    the average across every OTHER team's full-index totals, so an early-
    season or missing-data defense doesn't get zeros."""
    sums = {k: 0.0 for k in _DEFENSE_PROXY_KEYS}
    n = 0
    for entry in index.values():
        s, cnt = entry["cums"][-1]
        for k in _DEFENSE_PROXY_KEYS:
            sums[k] += s[k]
        n += cnt
    if n == 0:
        return {"yds_allowed": 220.0, "ya_allowed": 6.8, "cmp_pct_allowed": 64.0,
                "sack_rate_allowed": 0.065, "td_allowed": 1.4, "int_allowed": 0.8}
    return {
        "yds_allowed":        sums["Yds"] / n,
        "ya_allowed":         sums["Y/A"] / n,
        "cmp_pct_allowed":    sums["Cmp%"] / n,
        "sack_rate_allowed":  sums["Sk"] / max(sums["Att"] + sums["Sk"], 1.0),
        "td_allowed":         sums["TD"] / n,
        "int_allowed":        sums["Int"] / n,
    }


def defense_allowed_asof(index, team, date, fallback):
    """Average defense-allowed proxy strictly before `date` (or the full
    history if date=PREDICT_ASOF_DATE). Falls back to the league average if
    this team has no recorded prior games at that point."""
    entry = index.get(team)
    if not entry:
        return dict(fallback)
    i = bisect_left(entry["dates"], date)
    sums, n = entry["cums"][i]
    if n == 0:
        return dict(fallback)
    return {
        "yds_allowed":       sums["Yds"] / n,
        "ya_allowed":        sums["Y/A"] / n,
        "cmp_pct_allowed":   sums["Cmp%"] / n,
        "sack_rate_allowed": sums["Sk"] / max(sums["Att"] + sums["Sk"], 1.0),
        "td_allowed":        sums["TD"] / n,
        "int_allowed":       sums["Int"] / n,
    }


def _progress_2026():
    """Max number of 2026 games played by any QB, used as a league-wide proxy
    for how far into the 2026 season we are (see _dynamic_weight_2026)."""
    counts = [sum(1 for g in games if g["season"] == 2026)
              for games in _load_nflverse_cache().values()]
    counts = [c for c in counts if c > 0]
    if not counts and os.path.isdir(QB_DIR_2026):
        for f in os.listdir(QB_DIR_2026):
            if f.endswith(".xls") or f.endswith(".xlsx"):
                try:
                    counts.append(len(_read_qb_game_log(os.path.join(QB_DIR_2026, f), season=2026)))
                except Exception:
                    continue
    return max(counts) if counts else 0


def _dynamic_weight_2026(n_games=None):
    """WEIGHT_2026_TEAM used to be a fixed 0.65 applied to team stats no matter
    how many 2026 games existed — meaning a single Week 1 game could outweigh
    an entire 2025 season 65/35. Ramp it instead: mostly-2025 early, saturating
    toward WEIGHT_2026_MAX by WEIGHT_2026_SATURATION_GAMES games played."""
    if n_games is None:
        n_games = _progress_2026()
    frac = clamp(n_games / WEIGHT_2026_SATURATION_GAMES, 0.0, 1.0)
    return WEIGHT_2026_MIN + (WEIGHT_2026_MAX - WEIGHT_2026_MIN) * frac


def load_all_team_data():
    def_25 = _load_stats_excel(DEF_STATS_2025)
    off_25 = _load_stats_excel(OFF_STATS_2025)
    # 2026: prefer the nflverse cache (real weekly EPA, no manual downloads);
    # fall back to the legacy DEF/OFF_Stats_2026.xlsx if it hasn't been fetched yet.
    def_26 = _load_nflverse_team_stats(NFLVERSE_TEAM_STATS_2026, "DEF") or _load_stats_excel(DEF_STATS_2026)
    off_26 = _load_nflverse_team_stats(NFLVERSE_TEAM_STATS_2026, "OFF") or _load_stats_excel(OFF_STATS_2026)

    progress_2026 = _progress_2026()
    w26 = _dynamic_weight_2026(progress_2026)
    def_stats = _blend_stats(def_25, def_26, w26=w26)
    off_stats = _blend_stats(off_25, off_26, w26=w26)

    defense_allowed_index = build_defense_allowed_index()

    return {
        "def_stats": def_stats,
        "off_stats": off_stats,
        "weight_2026_used": w26,
        "progress_2026_games": progress_2026,
        # Per-game, date-lagged defense-allowed proxy — see build_defense_allowed_index.
        # This is what build_xgb_training_data and the live XGB prediction path
        # both use for the model's defensive features (leak-free at real weekly
        # granularity, not just season-to-season).
        "defense_allowed_index":    defense_allowed_index,
        "defense_allowed_fallback": _defense_proxy_fallback(defense_allowed_index),
        "lg_avg": {
            "epa_pass":     _league_avg(def_stats, "EPA/Pass"),
            "off_epa_pass": _league_avg(off_stats, "EPA/Pass"),
            "def_sack":     _league_avg(def_stats, "Sack %"),
            "off_sack":     _league_avg(off_stats, "Sack %"),
            "def_adot":     _league_avg(def_stats, "ADoT"),
            "off_adot":     _league_avg(off_stats, "ADoT"),
        },
    }


def team_names_sorted(data):
    return sorted(set(data["def_stats"]) | set(data["off_stats"]))

# =========================
# EWMA Adjustments (unchanged from V2)
# =========================
def apply_team_stats(mean, data, opp_full, qb_team_full):
    details = []
    adj = 0.0
    lg  = data["lg_avg"]

    def_stats = data["def_stats"].get(opp_full)
    if def_stats:
        epa = to_float(def_stats.get("EPA/Pass"))
        if epa is not None:
            # League-centered, like the OFF EPA/Pass adjustment below — if
            # league-average EPA/pass isn't exactly 0, an uncentered term
            # shifts every single projection in the same direction.
            d = (epa - lg["epa_pass"]) * DEF_PASS_WEIGHT; adj += d
            details.append(f"DEF EPA/Pass: {epa:+.3f} (lg {lg['epa_pass']:+.3f}) → {d:+.1f} yds")
        sack = to_float(def_stats.get("Sack %"))
        if sack is not None:
            d = -(sack - lg["def_sack"]) * DEF_SACK_WEIGHT; adj += d
            details.append(f"DEF Sack% {sack:.1%} (lg avg {lg['def_sack']:.1%}) → {d:+.1f} yds")
        adot = to_float(def_stats.get("ADoT"))
        if adot is not None:
            d = (adot - lg["def_adot"]) * DEF_ADOT_WEIGHT; adj += d
            details.append(f"DEF ADoT {adot:.2f} (lg avg {lg['def_adot']:.2f}) → {d:+.1f} yds")
    else:
        details.append(f"DEF Stats: no data for '{opp_full}'")

    if qb_team_full:
        off_stats = data["off_stats"].get(qb_team_full)
        if off_stats:
            epa = to_float(off_stats.get("EPA/Pass"))
            if epa is not None:
                d = (epa - lg["off_epa_pass"]) * OFF_PASS_WEIGHT; adj += d
                details.append(f"OFF EPA/Pass: {epa:+.3f} (lg {lg['off_epa_pass']:+.3f}) → {d:+.1f} yds")
            sack = to_float(off_stats.get("Sack %"))
            if sack is not None:
                d = -(sack - lg["off_sack"]) * OFF_SACK_WEIGHT; adj += d
                details.append(f"OFF Sack% {sack:.1%} (lg avg {lg['off_sack']:.1%}) → {d:+.1f} yds")
            adot = to_float(off_stats.get("ADoT"))
            if adot is not None:
                d = (adot - lg["off_adot"]) * OFF_ADOT_WEIGHT; adj += d
                details.append(f"OFF ADoT {adot:.2f} (lg avg {lg['off_adot']:.2f}) → {d:+.1f} yds")
        else:
            details.append(f"OFF Stats: no data for '{qb_team_full}'")

    return mean + adj, details, adj

# =========================
# Vegas
# =========================
def apply_vegas(mean, total, spread, kind="baseline"):
    """Game-context adjustment on a raw (pre-Vegas) projection. spread is the
    QB's own team spread, negative = favored. kind="xgb" uses XGB_BIAS_YDS
    instead of BASELINE_BIAS_YDS (see VEGAS_TOTAL_COEF et al.). Must mirror
    backtest_qb_model._adjust()."""
    bias = XGB_BIAS_YDS if kind == "xgb" else BASELINE_BIAS_YDS
    return mean + bias + (total - 44.0) * VEGAS_TOTAL_COEF + spread * VEGAS_SPREAD_COEF

# =========================
# XGBoost Feature Extraction
# =========================
def _qb_features(avgs, def_proxy, n_games, target="yds"):
    """target="yds" (default) uses the QB's own rolling yards as slot 0 and
    feeds the yards XGBRegressor; target="td" swaps in rolling TD rate for
    the TD-passes Poisson model — everything else (opponent defense features,
    n_games) is identical, see TD_FEATURE_NAMES."""
    primary = avgs["TD"] if target == "td" else avgs["Yds"]
    return [
        primary,
        avgs["Att"],
        avgs["Y/A"],
        avgs["Cmp%"],
        def_proxy["yds_allowed"],
        def_proxy["ya_allowed"],
        def_proxy["cmp_pct_allowed"],
        def_proxy["sack_rate_allowed"],
        def_proxy["td_allowed"],
        def_proxy["int_allowed"],
        n_games,
    ]

# =========================
# XGBoost Training
# =========================
def build_xgb_training_data(data):
    """Builds one training row per historical QB game using ONLY pregame/lagged
    opponent-defense data.

    The defensive features come from defense_allowed_asof(index, team, date),
    looked up at the TARGET GAME'S OWN DATE — so a September 2025 row only
    ever sees what that defense had allowed strictly before that date, at
    real weekly granularity. This replaced an earlier version that used the
    season-aggregate DEF_Stats Excel file (even after restricting it to the
    2025-only file to kill cross-season leakage, a Week 3 example still saw
    an average baked from Weeks 4-18 of that same season — a residual leak
    the season-aggregate file's lack of weekly splits couldn't avoid).

    Personnel-package features were dropped entirely (see MODEL_VERSION) — the
    only source for them turned out to require a paid subscription, and it
    wasn't worth scraping a vendor's paywalled product for two minor features.

    Low-volume games (see MIN_ATT) are kept as targets rather than deleted, to
    avoid survivorship bias toward normal-volume games — they're downweighted
    via sample_weight instead.
    """
    X, y, weights = [], [], []
    defense_index    = data["defense_allowed_index"]
    defense_fallback = data["defense_allowed_fallback"]

    for qb_name in list_qbs():
        try:
            games, _ = load_qb_games(qb_name)
        except Exception:
            continue
        if len(games) < ROLLING_GAMES + 1:
            continue

        for i in range(ROLLING_GAMES, len(games)):
            prior  = games[:i]
            target = games[i]
            avgs   = qb_rolling_avg(prior)

            opp_full  = resolve_team(target["Opp"])
            def_proxy = defense_allowed_asof(defense_index, opp_full, target["Date"], defense_fallback)

            X.append(_qb_features(avgs, def_proxy, i))
            y.append(target["Yds"])
            weights.append(LOW_VOL_SAMPLE_WEIGHT if target.get("low_vol") else 1.0)

    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(weights, dtype=np.float32))


def train_xgb_model(data):
    if not HAS_XGB or not HAS_PANDAS:
        return None, 0
    X, y, weights = build_xgb_training_data(data)
    n = len(y)
    if n < MIN_XGB_SAMPLES:
        return None, n
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,          # reduced from 4 — with ~190 rows, depth-4 trees can carve
        learning_rate=0.05,   # leaves matching only a handful of games, then extrapolate
        subsample=0.8,        # wildly for any opponent that lands in one (see
        colsample_bytree=0.8, # MAX_XGB_DEVIATION_YDS docstring for an observed example)
        min_child_weight=8,   # require each leaf to represent a meaningful group of games
        reg_alpha=0.1,
        reg_lambda=2.0,       # increased from 1.0 to further discourage over-specific splits
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, sample_weight=weights)
    return model, n


def clamp_xgb_deviation(xgb_pred, baseline_pred, max_dev=MAX_XGB_DEVIATION_YDS):
    return clamp(xgb_pred, baseline_pred - max_dev, baseline_pred + max_dev)

# =========================
# XGBoost Training — TD passes
# =========================
def build_xgb_training_data_td(data):
    """Same walk (leak-free, pregame-lagged defense) as build_xgb_training_data,
    targeting TD count instead of yards. Kept as a separate function rather than
    a target= branch inside the yards one, so a change here can never silently
    alter the yards model's training data or its MODEL_VERSION fingerprint."""
    X, y, weights = [], [], []
    defense_index    = data["defense_allowed_index"]
    defense_fallback = data["defense_allowed_fallback"]

    for qb_name in list_qbs():
        try:
            games, _ = load_qb_games(qb_name)
        except Exception:
            continue
        if len(games) < ROLLING_GAMES + 1:
            continue

        for i in range(ROLLING_GAMES, len(games)):
            prior  = games[:i]
            target = games[i]
            avgs   = qb_rolling_avg(prior)

            opp_full  = resolve_team(target["Opp"])
            def_proxy = defense_allowed_asof(defense_index, opp_full, target["Date"], defense_fallback)

            X.append(_qb_features(avgs, def_proxy, i, target="td"))
            y.append(target["TD"])
            weights.append(LOW_VOL_SAMPLE_WEIGHT if target.get("low_vol") else 1.0)

    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(weights, dtype=np.float32))


def train_xgb_td_model(data):
    if not HAS_XGB or not HAS_PANDAS:
        return None, 0
    X, y, weights = build_xgb_training_data_td(data)
    n = len(y)
    if n < MIN_XGB_SAMPLES:
        return None, n
    model = xgb.XGBRegressor(
        objective="count:poisson",  # TD counts are small non-negative integers
        max_delta_step=0.7,         # (0-4 in practice) — Poisson fits that shape much
        n_estimators=300,           # better than squared-error regression, which happily
        max_depth=3,                # predicts nonsensical negative/fractional values.
        learning_rate=0.05,         # max_delta_step=0.7 is XGBoost's own recommended
        subsample=0.8,              # stabilizer for count:poisson on sparse count data.
        colsample_bytree=0.8,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, sample_weight=weights)
    return model, n

# =========================
# Over/Under probability
# =========================
def prob_over(mean, std, line):
    """Exact P(X > line) for X ~ Normal(mean, std), via the closed-form normal
    CDF (erfc) instead of Monte Carlo sampling. The old version drew 10,000
    normal samples and counted how many exceeded the line — identical inputs
    could then produce slightly different reported percentages run to run,
    and 10k samples adds ~0.3-0.5pp of sampling noise on typical probabilities
    for no accuracy benefit, since the normal CDF has a closed form. Clipping
    the distribution at 0 (QB yards can't be negative) doesn't change this
    calculation as long as line > 0, which is always true for a real prop
    line — clipping only redistributes mass that was already below the line."""
    if std <= 0:
        return 1.0 if mean > line else 0.0
    z = (line - mean) / (std * math.sqrt(2.0))
    return 0.5 * math.erfc(z)


def prob_over_yds(mean, std, line):
    """Pass-yds P(over) under the backtest-chosen YDS_DIST (Normal unless the
    backtest found a skewed shape calibrates better)."""
    return pc.prob_over(mean, std, line, YDS_DIST)


def poisson_prob_over(lam, line):
    """Exact P(X > line) for X ~ Poisson(lam) — the right distribution for a
    small non-negative count like TD passes (usually 0-4), unlike Normal/std
    which assumes a continuous, symmetric, negative-capable quantity. Poisson
    has no separate spread parameter: variance IS the mean, so a QB projected
    for 1.5 TDs is inherently less uncertain in absolute terms than one
    projected for 3.5 — that's correct, not a simplification.
    P(X > line) = 1 - P(X <= floor(line)), computed by summing the PMF
    directly (TD counts are small, so this converges in a handful of terms —
    no need for a gamma-function library)."""
    if lam <= 0:
        return 0.0
    k_max = int(math.floor(line))
    if k_max < 0:
        return 1.0
    cdf = 0.0
    term = math.exp(-lam)   # P(X=0)
    cdf += term
    k = 0
    while k < k_max:
        k += 1
        term *= lam / k     # P(X=k) = P(X=k-1) * lam/k
        cdf += term
    return clamp(1.0 - cdf, 0.0, 1.0)

# =========================
# GUI
# =========================
class QBGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QB Passing Yard Model v4  [+ XGBoost]")
        self.geometry("900x980")

        style = ttk.Style(self)
        style.theme_use("clam")

        self.xgb_model    = None
        self.xgb_td_model = None
        self._status_var = tk.StringVar(value="Loading team data...")
        self._build_ui()
        self.after(100, self._init_data)

    def _init_data(self):
        self._set_status("Loading team data...")
        self.update_idletasks()
        try:
            self.data   = load_all_team_data()
            self._teams = team_names_sorted(self.data)
            self._qbs   = list_qbs()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        self.qb_box["values"]      = self._qbs
        self.qb_team_box["values"] = self._teams
        self.opp_box["values"]     = self._teams
        if self._qbs:
            self.qb_box.set(self._qbs[0])
        self.qb_team_box.set("Detroit Lions")
        self.opp_box.set("Green Bay Packers")

        if not HAS_XGB:
            self._set_status("XGBoost not installed — using EWMA only (pip install xgboost)")
            return

        self._set_status(f"Training XGBoost on {len(self._qbs)} QB game logs...")
        self.update_idletasks()

        try:
            self.xgb_model, n = train_xgb_model(self.data)
            if self.xgb_model is not None:
                self._set_status(f"XGBoost trained on {n} game samples. Ready.")
            else:
                self._set_status(f"Only {n} training samples — need {MIN_XGB_SAMPLES}+. Using EWMA.")
        except Exception as e:
            self.xgb_model = None
            self._set_status(f"XGBoost training failed: {e}. Using EWMA.")

        try:
            self.xgb_td_model, n_td = train_xgb_td_model(self.data)
        except Exception:
            self.xgb_td_model = None

    def _build_ui(self):
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame, text="QB Passing Yard Model v4  [+ XGBoost]",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(0, 4))

        ttk.Label(frame, textvariable=self._status_var,
                  font=("Segoe UI", 9, "italic"), foreground="#555").pack(pady=(0, 2))

        backtest_color = "#a15c00" if ("STALE" in BACKTEST_STATUS or "No backtest" in BACKTEST_STATUS) else "#2a6b2a"
        ttk.Label(frame, text=BACKTEST_STATUS, font=("Segoe UI", 8, "italic"),
                  foreground=backtest_color, wraplength=850).pack(pady=(0, 2))

        td_backtest_color = "#a15c00" if ("STALE" in TD_BACKTEST_STATUS or "No backtest" in TD_BACKTEST_STATUS) else "#2a6b2a"
        ttk.Label(frame, text=TD_BACKTEST_STATUS, font=("Segoe UI", 8, "italic"),
                  foreground=td_backtest_color, wraplength=850).pack(pady=(0, 8))

        inputs = ttk.LabelFrame(frame, text="Inputs", padding=12)
        inputs.pack(fill="x")

        def row(label, widget, r):
            ttk.Label(inputs, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="w", pady=3)

        self.qb_box = ttk.Combobox(inputs, values=[], width=28)
        row("Quarterback", self.qb_box, 0)

        self.qb_team_box = ttk.Combobox(inputs, values=[], width=28)
        row("QB's Team", self.qb_team_box, 1)

        self.opp_box = ttk.Combobox(inputs, values=[], width=28)
        row("Opponent", self.opp_box, 2)

        fields = [
            ("Vegas Total",  "44.5"),
            ("Spread",       "-3.0"),
            ("Yds Line",     "235.5"),
            ("TD Line",      "1.5"),
            ("Wind (mph)",   "0"),
            ("Temp (F)",     "70"),
            ("Rain (0/1)",   "0"),
            ("Snow (0/1)",   "0"),
            ("Indoor (0/1)", "1"),
        ]
        self.entries = {}
        for i, (label, default) in enumerate(fields, start=3):
            e = ttk.Entry(inputs, width=10)
            e.insert(0, default)
            row(label, e, i)
            self.entries[label] = e

        ttk.Button(frame, text="Run Projection", command=self.run).pack(pady=12)

        self.output = tk.Text(frame, height=36, font=("Consolas", 10))
        self.output.pack(fill="both", expand=True)

    def _set_status(self, msg):
        self._status_var.set(msg)

    def run(self):
        try:
            qb      = self.qb_box.get().strip()
            qb_team = self.qb_team_box.get().strip()
            opp     = self.opp_box.get().strip()
            total   = float(self.entries["Vegas Total"].get())
            spread  = float(self.entries["Spread"].get())
            line    = float(self.entries["Yds Line"].get())
            td_line = float(self.entries["TD Line"].get())
            wind    = float(self.entries["Wind (mph)"].get())
            temp    = float(self.entries["Temp (F)"].get())
            rain    = int(self.entries["Rain (0/1)"].get())
            snow    = int(self.entries["Snow (0/1)"].get())
            indoor  = int(self.entries["Indoor (0/1)"].get())

            games, data_label = load_qb_games(qb)
            avgs   = qb_rolling_avg(games)
            pool   = usable_games(games)   # excludes low-vol games from the display/std window, same as the average
            n_used = min(ROLLING_GAMES, len(pool))
            recent = pool[-n_used:]
            dates  = [g["Date"] for g in recent]

            # ── Baseline: blend EWMA + median + Att×Y/A ───────────────────
            rolling_yds = avgs["Yds"]
            baseline    = avgs["Att"] * avgs["Y/A"]
            median_yds  = float(np.median([g["Yds"] for g in recent]))  # outlier-resistant anchor
            raw_blend   = EWMA_BLEND * rolling_yds + MEDIAN_BLEND * median_yds + ATTYA_BLEND * baseline
            ewma_mean   = apply_vegas(raw_blend, total, spread)
            ewma_mean, stats_details, stats_adj = apply_team_stats(
                ewma_mean, self.data, opp, qb_team or None
            )

            # ── XGBoost prediction ─────────────────────────────────────────
            xgb_raw = xgb_adjusted = xgb_unclamped = None
            xgb_clamped = False
            model_source = "EWMA only"

            if self.xgb_model is not None:
                # Same lagged-proxy lookup as training, just "as of now" —
                # every game on record counts, since that's genuinely all
                # pregame information for a future game (see PREDICT_ASOF_DATE).
                def_proxy = defense_allowed_asof(
                    self.data["defense_allowed_index"], opp,
                    PREDICT_ASOF_DATE, self.data["defense_allowed_fallback"]
                )
                feat = _qb_features(avgs, def_proxy, len(games))
                xgb_unclamped = float(self.xgb_model.predict(np.array([feat], dtype=np.float32))[0])
                # Guardrail against a single unstable leaf dominating the blend —
                # see MAX_XGB_DEVIATION_YDS. Compared against raw_blend since both
                # are pre-Vegas/EPA-adjustment at this point.
                xgb_raw = clamp_xgb_deviation(xgb_unclamped, raw_blend)
                xgb_clamped = abs(xgb_raw - xgb_unclamped) > 0.01
                # Vegas + weather applied on top (not in training data)
                xgb_adjusted = apply_vegas(xgb_raw, total, spread, kind="xgb")

                # Which regime is this pick in? Compared pre-Vegas (raw_blend
                # vs xgb_raw), matching exactly what backtest_qb_model.py
                # measured (ewma_pred vs xgb_pred) — see XGB_BLEND_AMPLIFY et al.
                blend_regime = "neutral"
                blend_weight = XGB_BLEND_NEUTRAL
                if LEAGUE_AVG_YDS is not None:
                    dev = raw_blend - LEAGUE_AVG_YDS
                    adj = xgb_raw - raw_blend
                    if abs(dev) >= AMPLIFY_DEV_THRESHOLD_YDS and abs(adj) > 1e-6:
                        if (dev > 0) == (adj > 0):
                            blend_regime, blend_weight = "amplify", XGB_BLEND_AMPLIFY
                        else:
                            blend_regime, blend_weight = "correct", XGB_BLEND_CORRECT

                final_mean = blend_weight * xgb_adjusted + (1 - blend_weight) * ewma_mean
                model_source = (f"XGBoost ({blend_weight:.0%}, {blend_regime}) + "
                                f"EWMA ({1-blend_weight:.0%})")
            else:
                final_mean = ewma_mean

            # ── TD passes: separate Poisson-based model (see poisson_prob_over).
            # No Vegas/weather adjustment here — those formulas (yds/mph wind
            # etc.) are calibrated for yards, not meaningful for a TD count. ──
            td_xgb_raw = None
            td_model_source = "EWMA only"
            td_ewma = avgs["TD"]
            if self.xgb_td_model is not None:
                td_def_proxy = defense_allowed_asof(
                    self.data["defense_allowed_index"], opp,
                    PREDICT_ASOF_DATE, self.data["defense_allowed_fallback"]
                )
                td_feat = _qb_features(avgs, td_def_proxy, len(games), target="td")
                td_xgb_raw = float(self.xgb_td_model.predict(np.array([td_feat], dtype=np.float32))[0])
                td_lambda = TD_XGB_BLEND * td_xgb_raw + (1 - TD_XGB_BLEND) * td_ewma
                td_model_source = f"XGBoost ({TD_XGB_BLEND:.0%}) + EWMA ({1-TD_XGB_BLEND:.0%})"
            else:
                td_lambda = td_ewma
            td_lambda = max(td_lambda, 0.01)  # Poisson lambda must be positive

            td_raw_over = poisson_prob_over(td_lambda, td_line)
            td_cal_over = float(calibrate_probability(td_raw_over, alpha=TD_CALIBRATION_ALPHA))

            # ── Std dev: blend the QB's own recent-game std with the model's
            # out-of-sample residual std for this attempt-volume bucket (once
            # backtest_qb_model.py has been run). Until then RESIDUAL_STD_BY_BUCKET
            # is empty and this behaves exactly like the old QB-only std. ──────
            qb_hist_std = float(np.std([g["Yds"] for g in recent], ddof=1)) if len(recent) >= 4 else None
            bucket  = _std_bucket(avgs["Att"])
            oos_std = RESIDUAL_STD_BY_BUCKET.get(bucket)
            if qb_hist_std is not None and oos_std is not None:
                std = 0.5 * qb_hist_std + 0.5 * oos_std
                std_source = f"blend: QB hist {qb_hist_std:.1f} + model OOS resid [{bucket}] {oos_std:.1f}"
            elif oos_std is not None:
                std = oos_std
                std_source = f"model OOS residual std [{bucket}] (insufficient QB history)"
            elif qb_hist_std is not None:
                std = qb_hist_std
                std_source = f"QB rolling std ({len(recent)} games) — run backtest_qb_model.py for OOS residual std"
            else:
                std = DEFAULT_STD
                std_source = "default"
            # First backtest run showed real out-of-sample residual std (80-94 yds)
            # well above STD_MAX=65 — flag it instead of silently clipping away
            # evidence the model's confidence band may be too tight for props.
            std_clip_warning = std > STD_MAX
            std = clamp(std, STD_MIN, STD_MAX)

            final_mean, std = apply_weather(final_mean, std, wind, temp, rain, snow, bool(indoor))
            std_clip_warning = std_clip_warning or std > STD_MAX
            std = clamp(std, STD_MIN, STD_MAX)

            raw_over = prob_over_yds(final_mean, std, line)
            cal_over = float(calibrate_probability(raw_over))

            # ── Output ─────────────────────────────────────────────────────
            self.output.delete("1.0", tk.END)
            out = self.output

            def w(s): out.insert(tk.END, s)
            def sep(): w(f"{'─'*65}\n")

            w(f"QB: {qb}  |  Team: {qb_team or 'N/A'}  |  Opponent: {opp}\n")
            w(f"Data: {data_label}  |  Total games: {len(games)}\n")
            w(f"2026 team-data weight: {self.data.get('weight_2026_used', 0):.0%} "
              f"(dynamic, {self.data.get('progress_2026_games', 0)} 2026 games played league-wide)\n")
            w(f"Last {n_used} games: {', '.join(dates)}\n")
            sep()
            w(f"EWMA Rolling Averages (last {n_used} games, decay={EWMA_DECAY}):\n")
            w(f"  Att={avgs['Att']:.1f}  Cmp={avgs['Cmp']:.1f}  Cmp%={avgs['Cmp%']:.1f}  Y/A={avgs['Y/A']:.2f}\n")
            w(f"  TD={avgs['TD']:.2f}  Int={avgs['Int']:.2f}  Yds={avgs['Yds']:.1f}\n")
            sep()
            w(f"EPA/Sack/ADoT Adjustments  ({stats_adj:+.1f} yds):\n")
            for d in stats_details:
                w(f"  • {d}\n")
            sep()
            w(f"EWMA Yds (decay={EWMA_DECAY}): {rolling_yds:.1f}  (weight {EWMA_BLEND:.0%})\n")
            w(f"Median last {n_used} games:       {median_yds:.1f}  (weight {MEDIAN_BLEND:.0%})\n")
            w(f"Att×Y/A baseline:             {baseline:.1f}  (weight {ATTYA_BLEND:.0%})\n")
            w(f"Blend (pre-Vegas/EPA-adj): {raw_blend:.1f} yds\n")
            w(f"Game context: bias {BASELINE_BIAS_YDS:+.1f} (XGB {XGB_BIAS_YDS:+.1f}) "
              f"{VEGAS_TOTAL_COEF:+.2f}×(total−44) {VEGAS_SPREAD_COEF:+.2f}×spread "
              f"= {apply_vegas(0.0, total, spread):+.1f} yds\n")
            w(f"Blend (post-Vegas/EPA-adj): {ewma_mean:.1f} yds\n")
            if xgb_raw is not None:
                if xgb_clamped:
                    w(f"XGBoost raw (model): {xgb_unclamped:.1f} yds\n")
                    w(f"  [!] Clamped to {xgb_raw:.1f} yds — diverged from the EWMA baseline "
                      f"by more than {MAX_XGB_DEVIATION_YDS:.0f} yds (see MAX_XGB_DEVIATION_YDS). "
                      f"With this little training data, a single unstable leaf can produce "
                      f"an extreme, unsupported prediction for this matchup.\n")
                else:
                    w(f"XGBoost raw:      {xgb_raw:.1f} yds\n")
                w(f"XGBoost (+vegas): {xgb_adjusted:.1f} yds\n")
            w(f"\nModel: {model_source}\n")
            sep()
            w(f"Final Projection:   {final_mean:.1f} yards\n")
            w(f"Std Dev:            {std:.1f} yds ({std_source})\n")
            if std_clip_warning:
                w(f"  [!] Uncalibrated estimate exceeded STD_MAX={STD_MAX:.1f} and was clipped — "
                  f"confidence interval may be too tight. See backtest_qb_model.py residual std.\n")
            w("\n")
            w(f"Line:               {line}\n")
            w(f"Raw Over %:         {raw_over * 100:.2f}%\n")
            w(f"Calibrated Over %:  {cal_over * 100:.2f}%\n")
            w(f"Calibrated Under %: {(1 - cal_over) * 100:.2f}%\n")

            sep()
            w("TOUCHDOWN PASSES (separate Poisson model)\n")
            w(f"EWMA rolling TD rate: {td_ewma:.2f}\n")
            if td_xgb_raw is not None:
                w(f"XGBoost (Poisson) raw: {td_xgb_raw:.2f}\n")
            w(f"Model: {td_model_source}\n")
            w(f"Projected TD rate (lambda): {td_lambda:.2f}\n")
            w(f"\nTD Line:            {td_line}\n")
            w(f"Raw Over %:         {td_raw_over * 100:.2f}%\n")
            w(f"Calibrated Over %:  {td_cal_over * 100:.2f}%\n")
            w(f"Calibrated Under %: {(1 - td_cal_over) * 100:.2f}%\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    QBGuiApp().mainloop()
