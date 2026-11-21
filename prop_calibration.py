"""
Over/under distribution + calibration selection shared by the RB and QB yardage models.

Why this exists (2026-11-14)
----------------------------
Both backtests used to pick the calibration alpha by scoring P(over) at a stand-in
line equal to the model's own baseline forecast. At that line P(over) is ~0.5 for
every row, so Brier is flat across alphas and the "best" alpha is noise — RB
rush_yds alpha=0.2 came from an exact tie (every alpha scored 0.2500) resolved by
taking the first grid value. Real prop lines sit some distance from the model's
projection, and that is exactly where alpha and the distribution's shape matter.

So calibration is now scored at a SPREAD OF LINES around the projection:
line = floor(mean + k * std) + 0.5 for k in OFFSETS_SD (half-yard, like real props,
so integer outcomes never push), dropping lines below MIN_LINE (books don't post them).

Distribution: rushing/receiving yards are right-skewed (median < mean), which a
symmetric Normal can't represent — at off-projection lines it overstated P(over)
by ~8-9pp in the middle of the range. The candidate family here is a SHIFTED GAMMA
(X + shift ~ Gamma with the model's mean and std), the same shape
rb_model_V4.leading_rusher_probs already uses. The backtest picks Normal vs. gamma
(and the shift) and alpha jointly, by Brier across the offset lines.

Selection never resolves a flat curve silently: select_best() raises
FlatCurveError when the curve can't distinguish its candidates, and records every
candidate within TIE_TOL of the minimum. The full curve is written to the JSON.
"""
import math

import numpy as np

try:
    from scipy.special import gammaincc as _gammaincc
    HAS_SCIPY = True
except ImportError:          # pragma: no cover - scipy is installed in this project
    HAS_SCIPY = False

OFFSETS_SD = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
MIN_LINE = 0.5
ALPHA_GRID = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
GAMMA_SHIFTS = (0.0, 5.0, 10.0, 20.0, 40.0)
FLAT_TOL = 1e-4     # max-min Brier below this -> the curve can't pick anything
TIE_TOL = 1e-4      # candidates this close to the minimum are reported as ties

NORMAL = {"family": "normal"}


class FlatCurveError(ValueError):
    """The scoring curve is flat — any pick from it would be arbitrary."""


# =========================
# Distributions
# =========================
def normal_prob_over(mean, std, line):
    if std <= 0:
        return 1.0 if mean > line else 0.0
    return 0.5 * math.erfc((line - mean) / (std * math.sqrt(2.0)))


def gamma_prob_over(mean, std, line, shift=0.0):
    """P(X > line) where X + shift ~ Gamma(mean + shift, std). Right-skewed with
    median < mean; shift lets a few slightly negative outcomes exist and makes the
    shape less skewed as it grows (shift -> inf approaches the Normal)."""
    if std <= 0:
        return 1.0 if mean > line else 0.0
    ms = max(mean + shift, 1.0)
    x = line + shift
    if x <= 0:
        return 1.0
    shape = (ms / std) ** 2
    scale = std ** 2 / ms
    if HAS_SCIPY:
        return float(_gammaincc(shape, x / scale))
    # Wilson-Hilferty fallback
    c = 1.0 / (9.0 * shape)
    z = ((x / (shape * scale)) ** (1.0 / 3.0) - (1 - c)) / math.sqrt(c)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def prob_over(mean, std, line, dist=None):
    dist = dist or NORMAL
    if dist.get("family") == "gamma":
        return gamma_prob_over(mean, std, line, float(dist.get("shift", 0.0)))
    return normal_prob_over(mean, std, line)


def calibrate(p, alpha):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-alpha * np.log(p / (1 - p))))


def dist_label(dist):
    if dist.get("family") == "gamma":
        return f"gamma(shift={float(dist.get('shift', 0.0)):g})"
    return "normal"


def candidate_dists(shifts=GAMMA_SHIFTS):
    return [dict(NORMAL)] + [{"family": "gamma", "shift": float(s)} for s in shifts]


# =========================
# Selection
# =========================
def select_best(curve, prefer=None, flat_tol=FLAT_TOL, tie_tol=TIE_TOL):
    """curve: {key: score}, lower is better. Returns (best_key, ties).

    Raises FlatCurveError if max - min < flat_tol: the scores can't tell the
    candidates apart (e.g. every line equals the forecast, so P(over) ~ 0.5 and
    every alpha scores 0.25). Candidates within tie_tol of the minimum are ties;
    among them the one with the smallest prefer(key) wins (default: the minimum),
    and all of them are returned so the caller reports them."""
    if not curve:
        raise FlatCurveError("empty curve")
    vals = np.array(list(curve.values()), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise FlatCurveError("non-finite scores in curve")
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < flat_tol:
        raise FlatCurveError(f"curve is flat (range {hi - lo:.2e} < {flat_tol:.0e}): "
                             f"{ {k: round(v, 5) for k, v in curve.items()} }")
    ties = [k for k, v in curve.items() if v - lo <= tie_tol]
    best = min(ties, key=prefer) if (prefer and len(ties) > 1) else min(curve, key=curve.get)
    return best, ties


# =========================
# Offset-line scoring
# =========================
def offset_lines(mean, std, offsets=OFFSETS_SD, min_line=MIN_LINE):
    """Returns (row_index, offset_k, line) arrays for every realistic line."""
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    idx, ks, lines = [], [], []
    for k in offsets:
        ln = np.floor(mean + k * std) + 0.5
        ok = ln >= min_line
        idx.append(np.nonzero(ok)[0]); ks.append(np.full(ok.sum(), k)); lines.append(ln[ok])
    return np.concatenate(idx), np.concatenate(ks), np.concatenate(lines)


def raw_probs(mean, std, idx, lines, dist):
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    if dist.get("family") == "gamma" and HAS_SCIPY:
        s = float(dist.get("shift", 0.0))
        ms = np.maximum(mean[idx] + s, 1.0)
        sd = std[idx]
        x = lines + s
        p = _gammaincc((ms / sd) ** 2, np.maximum(x, 0.0) / (sd ** 2 / ms))
        return np.where(x <= 0, 1.0, p)
    return np.array([prob_over(mean[i], std[i], ln, dist) for i, ln in zip(idx, lines)])


def calibration_bins(p, act, edges=(0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.01), min_n=20):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (p >= lo) & (p < hi)
        if mk.sum() >= min_n:
            out.append({"range": [lo, min(hi, 1.0)], "n": int(mk.sum()),
                        "pred": round(float(p[mk].mean()), 3), "actual": round(float(act[mk].mean()), 3)})
    return out


def _brier_by_offset(p, act, ks):
    return {f"{k:+.2f}": round(float(np.mean((p[ks == k] - act[ks == k]) ** 2)), 4)
            for k in sorted(set(ks.tolist())) if (ks == k).sum()}


def sweep_offsets(y, mean, std, dists=None, alpha_grid=ALPHA_GRID, baseline=None):
    """Joint (distribution, alpha) selection by pooled Brier over offset lines.

    baseline: optional (dist, alpha) — the configuration being replaced — scored
    on the same lines so before/after are comparable.
    Returns a JSON-ready dict with the full curve, the pick, ties, Brier by offset
    and calibration bins for the pick (and for the baseline/raw Normal)."""
    y = np.asarray(y, float)
    dists = dists or candidate_dists()
    idx, ks, lines = offset_lines(mean, std)
    act = (y[idx] > lines).astype(float)

    raw_cache, curve = {}, {}
    for d in dists:
        lab = dist_label(d)
        raw_cache[lab] = (d, raw_probs(mean, std, idx, lines, d))
        for a in alpha_grid:
            curve[f"{lab}|{a:.2f}"] = float(np.mean((calibrate(raw_cache[lab][1], a) - act) ** 2))

    def prefer(key):   # among ties: alpha closest to 1 (least distortion), then Normal, then larger shift
        lab, a = key.split("|")
        d = raw_cache[lab][0]
        return (abs(float(a) - 1.0), d["family"] != "normal", -float(d.get("shift", 0.0)))

    best_key, ties = select_best(curve, prefer=prefer)
    lab, a = best_key.split("|")
    best_dist, best_alpha = raw_cache[lab][0], float(a)
    p_best = calibrate(raw_cache[lab][1], best_alpha)

    def summary(p):
        return {"brier": round(float(np.mean((p - act) ** 2)), 4),
                "brier_near_line": round(float(np.mean((p[np.abs(ks) <= 0.5] - act[np.abs(ks) <= 0.5]) ** 2)), 4),
                "brier_by_offset": _brier_by_offset(p, act, ks),
                "calibration_bins": calibration_bins(p, act),
                "over_rate_pred": round(float(p.mean()), 3)}

    out = {
        "method": "pooled Brier over lines floor(mean + k*std)+0.5, k in OFFSETS_SD, line >= MIN_LINE",
        "offsets_sd": list(OFFSETS_SD), "n_rows": int(len(y)), "n_scored": int(len(idx)),
        "over_rate_actual": round(float(act.mean()), 3),
        "brier_curve": {k: round(v, 5) for k, v in curve.items()},
        "best_distribution": best_dist, "best_alpha": best_alpha,
        "ties": ties, "chosen": summary(p_best),
    }
    nl = "normal"
    for a_ in (1.0,):
        out["normal_alpha_1"] = summary(calibrate(raw_cache[nl][1], a_))
    # best alpha for the Normal alone, so the distribution change and the alpha
    # change can be read separately
    norm_curve = {k: v for k, v in curve.items() if k.startswith("normal|")}
    nk = min(norm_curve, key=norm_curve.get)
    out["normal_best"] = {"alpha": float(nk.split("|")[1]), **summary(calibrate(raw_cache[nl][1], float(nk.split("|")[1])))}
    if baseline is not None:
        bd, ba = baseline
        out["previous_config"] = {"distribution": bd, "alpha": ba,
                                  **summary(calibrate(raw_probs(mean, std, idx, lines, bd), ba))}
    return out


def print_sweep(name, sw, indent="  "):
    """Human-readable dump: the full curve (one row per distribution), the pick,
    ties, and before/after by offset and calibration bins."""
    print(f"{indent}calibration sweep [{name}]: {sw['n_scored']} line-rows from {sw['n_rows']} OOS rows, "
          f"actual over rate {sw['over_rate_actual']:.1%}")
    labels = []
    for k in sw["brier_curve"]:
        lab = k.split("|")[0]
        if lab not in labels:
            labels.append(lab)
    alphas = sorted({k.split("|")[1] for k in sw["brier_curve"]}, key=float)
    print(f"{indent}  {'Brier':16s}" + "".join(f"a={a:>5s} " for a in alphas))
    for lab in labels:
        print(f"{indent}  {lab:16s}" + "".join(f"{sw['brier_curve'][f'{lab}|{a}']:.4f}  " for a in alphas))
    print(f"{indent}  -> pick {dist_label(sw['best_distribution'])} alpha={sw['best_alpha']:.2f}"
          + (f"  (ties within {TIE_TOL:g}: {', '.join(sw['ties'])})" if len(sw["ties"]) > 1 else ""))
    rows = [("previous", sw.get("previous_config")), ("normal a=1", sw["normal_alpha_1"]),
            (f"normal a={sw['normal_best']['alpha']:.1f}",
             None if sw["normal_best"]["alpha"] == 1.0 else sw["normal_best"]), ("chosen", sw["chosen"])]
    offs = list(sw["chosen"]["brier_by_offset"])
    print(f"{indent}  {'Brier by offset (k*sd)':24s}" + "".join(f"{o:>7s}" for o in offs) + "   pooled  |k|<=.5")
    for nm, s in rows:
        if s is None:
            continue
        print(f"{indent}  {nm:24s}" + "".join(f"{s['brier_by_offset'].get(o, float('nan')):7.4f}" for o in offs)
              + f"   {s['brier']:.4f}  {s['brier_near_line']:.4f}")
    for nm, s in rows:
        if s is None:
            continue
        print(f"{indent}  bins {nm:20s}" + " | ".join(
            f"{b['range'][0]:.2f}-{b['range'][1]:.2f} n={b['n']} {b['pred']:.1%}/{b['actual']:.1%}"
            for b in s["calibration_bins"]))
