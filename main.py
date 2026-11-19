"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each PS5 trophy's Exophase earn rate at a case's target as five
quantiles, from the house's recorded reads of the trophy page.

Model (own design, standard library only, deterministic, offline).

The page is not a smooth series: Exophase refreshes a game's counts on its
own schedule, hours apart, and between refreshes every read is identical.
So the history is treated as a sequence of distinct states (one per refresh),
and the forecast is built in COUNTS, not in rates:

  * N(t): players tracked.  Fitted as a power law in hours since release
    (log N against log(1 + t)) over the last 24 hours of distinct
    states only -- the launch-night catch-up burst would dominate a
    fit over the whole history and explode the extrapolation -- then
    extrapolated to the target, with the slope and the total growth
    capped.
  * E_i(t): players holding trophy i.  Across each refresh, the jump in
    earners has two causes: newly tracked players who already hold the
    trophy, and tracked players earning it since the last refresh:

        dE_i = a_i * dN  +  lam_i * (N_prev - E_i_prev) * dt

    A free two-parameter fit of (a_i, lam_i) is not identified: dN and
    the non-holder-hours (N - E) * dt move almost in proportion across
    refreshes, so least squares splits the jump between the two causes
    arbitrarily.  So a_i is fixed by a prior -- the launch-night
    refresh put a new cohort's attainment at about 7% of the old
    cohort's rate for the same trophy, a_i = 0.07 * r_i / 100 -- and
    only lam_i, the hourly rate at which tracked non-holders earn the
    trophy, is fitted per trophy (one-parameter least squares, shrunk
    toward its tier's pooled fit).  Earners are then projected to the
    target by integrating dE/dt = a_i * N'(t) + lam_i * (N - E) with N
    growing exponentially from its last value to the fitted target.
    This separates dilution (new players arriving with few trophies
    pulls every rate down) from progression (players earning trophies
    over days pushes rates up), which a plain rate-trend fit cannot do.

  * When the history holds no refresh at all (the practice cases: every
    read identical), there is nothing to fit, so the forecast falls back
    to a prior over the next refresh, calibrated on the launch-night
    refresh the competition spec itself describes (638 -> 3,868 players
    tracked, 943 -> 4,276 trophies earned; the new cohort held the
    non-100% trophies at about 7% of the old cohort's rate):
    a Poisson refresh hazard (mean interval 3.5 h) gives the weight of
    "no refresh before the target" against "at least one refresh", and
    the refresh branch dilutes each rate by the new cohort's attainment
    and the player-count jump.  The five quantiles are the quantiles of
    that two-branch mixture, computed on a logit grid, so a short
    horizon keeps its median near the last rate while its lower
    quantiles already price in the refresh.

  * Spread: quantiles are normal on the logit scale around the median
    rate, with a sigma that grows with the horizon and with the fit's
    residual noise, so the 90% band is honest about how far the rate
    can move.  A trophy last seen at 100% keeps near-100% quantiles:
    every tracked player holds it, and the launch-night refresh left it
    at 100%.

I/O: reads every *.in file in $MAKEFUN_INPUT_DIR (else /input) and writes
one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output):
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
No randomness, no wall clock, no network: two runs give identical bytes.
"""

import json
import math
import os
from datetime import datetime, timezone

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- priors for the no-refresh fallback (see module docstring) ---
REFRESH_MEAN_HOURS = 3.5    # mean interval between Exophase refreshes
REFRESH_MULT = 4.0          # players-tracked multiplier at one refresh
KAPPA = 0.07                # new cohort's attainment, as a fraction of
                            # the current cohort's rate for the same trophy
SIGMA_NO_REFRESH = 0.18     # logit spread when the page stays frozen
SIGMA_REFRESH = 0.5         # logit spread of the refresh branch
# --- data-driven path ---
SIGMA_BASE = 0.18           # base logit sigma (tuned on backtests:
                            # 0.30 over-covers at ~0.98, 0.18 keeps
                            # ~0.97 coverage while cutting pinball ~25%)
SIGMA_PER_ROOT_DAY = 0.09   # extra sigma per sqrt(horizon in days)
N_SHRINK = 1.0              # shrink the fitted log-growth of the tracked
                            # player count toward zero; tested at 0.7 and
                            # 0.85 on the backtest suite and both hurt
                            # (the fit is usually right; shrinking it
                            # pushes every forecast rate too high), so
                            # it stays 1.0 -- no shrinkage
SHRINK_K = 2.0              # transitions' worth of weight given to the
                            # tier-pooled fit when shrinking per-trophy fits


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(
        timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(L):
    if L >= 40.0:
        return 99.99
    if L <= -40.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-L))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantiles(branches, levels=LEVELS):
    """Quantiles of a mixture of normals on the logit scale.

    branches: list of (weight, logit_mean, logit_sigma).  The CDF is
    evaluated on a fixed grid and inverted by scanning; fully
    deterministic.
    """
    lo, hi, step = -9.21, 9.21, 0.01
    n_grid = int(round((hi - lo) / step)) + 1
    grid = [lo + i * step for i in range(n_grid)]
    cdfs = []
    for L in grid:
        c = 0.0
        for w, mu, sd in branches:
            c += w * norm_cdf((L - mu) / sd)
        cdfs.append(c)
    out = []
    for tau in levels:
        q = grid[-1]
        for L, c in zip(grid, cdfs):
            if c >= tau:
                q = L
                break
        out.append(inv_logit_pct(q))
    return out


def normal_quantiles(median_pct, sigma, levels=LEVELS):
    mu = logit_pct(median_pct)
    return [inv_logit_pct(mu + z * sigma) for z in Z]


def distinct_states(case):
    """Collapse the history to one entry per distinct page state.

    Returns [(hours_since_release, players_tracked, rates_tuple)] with
    the first time each distinct state was seen.
    """
    states = []
    seen = set()
    for h in case["history"]:
        rates = tuple(h["rates"])
        key = (h["players_tracked"], rates)
        if key in seen:
            continue
        seen.add(key)
        t = (parse_time(h["at"]) - RELEASE).total_seconds() / 3600.0
        states.append((max(t, 0.0), h["players_tracked"], rates))
    return states


def last_known_rates(case):
    """Last non-null rate per trophy (rates may be null in a read)."""
    n = len(case["trophies"])
    out = [None] * n
    for h in case["history"]:
        for i, r in enumerate(h["rates"]):
            if r is not None:
                out[i] = r
    return out


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, resid_sd, sxx, x_mean)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    if n > 2:
        rsd = math.sqrt(
            sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        rsd = 0.0
    return a, b, rsd, sxx, mx


def _near_universal_q(r_last):
    """Quantiles for a trophy (nearly) every tracked player holds.
    Anchored on its last rate with a tight band just under 100%: the
    logit scale punishes even a 0.1-point miss up here, and the one
    such trophy stayed at exactly 100% through a 6x dilution on launch
    night, so the band bets on it staying (near-)universal while its
    lower quantiles still price in a small dip."""
    base = min(max(r_last, 0.0), 99.99)
    qs = [base - 0.2, base - 0.05, base - 0.01, 99.99, 99.99]
    return _monotone([round(q, 4) for q in qs])


def transition_fits(states, n_trophies, tiers, last_rates):
    """Per-trophy progression rate lam_i, with the new-cohort attainment
    fixed by the prior a_i = KAPPA * r_i / 100.

    A free two-parameter fit of (a_i, lam_i) is not identified: across
    refreshes, dN and the non-holder-hours (N - E) * dt move almost in
    proportion (both scale with the player base and the gap between
    refreshes), so least squares splits the earners' jump between the
    two causes arbitrarily.  Fixing a_i by the prior -- the launch-night
    refresh put the new cohort's attainment at ~7% of the old cohort's
    rate -- leaves lam_i as a one-parameter fit, which is identified:
    lam_i = sum X * (dE - a_i * dN) / sum X^2 over the refreshes, with
    X = (N_prev - E_prev) * dt.  lam_i is then shrunk toward its tier's
    pooled one-parameter fit.  Returns (lams, tier_lams, rel_resid)."""
    n_trans = len(states) - 1
    t_last = states[-1][0]
    tier_num = {}
    tier_den = {}
    per_num = [0.0] * n_trophies
    per_den = [0.0] * n_trophies
    for t in set(tiers):
        tier_num[t] = 0.0
        tier_den[t] = 0.0
    for s in range(1, len(states)):
        t_prev, n_prev, r_prev = states[s - 1]
        t_cur, n_cur, r_cur = states[s]
        dn = n_cur - n_prev
        dt = max(t_cur - t_prev, 0.0)
        if dn <= 0 or dt <= 0:
            continue
        # Recency weight: the earning rate drifts as the player base
        # ages (early adopters earn fast, later arrivals slower), so
        # recent refreshes speak louder about the target's regime.
        # Half-life 12 hours.
        w_s = 0.5 ** ((t_last - t_cur) / 12.0)
        for i in range(n_trophies):
            rp = r_prev[i]
            rc = r_cur[i]
            if rp is None or rc is None:
                continue
            a_i = KAPPA * rp / 100.0
            e_prev = rp / 100.0 * n_prev
            de = rc / 100.0 * n_cur - e_prev
            x = (n_prev - e_prev) * dt
            per_num[i] += w_s * x * (de - a_i * dn)
            per_den[i] += w_s * x * x
            tier_num[tiers[i]] += w_s * x * (de - a_i * dn)
            tier_den[tiers[i]] += w_s * x * x
    tier_lams = {}
    for t in tier_num:
        lam = (tier_num[t] / tier_den[t] if tier_den[t] > 0 else 0.0)
        tier_lams[t] = min(max(lam, 0.0), 0.1)
    lams = []
    rel_res = []
    for i in range(n_trophies):
        raw = (per_num[i] / per_den[i] if per_den[i] > 0 else None)
        tlam = tier_lams[tiers[i]]
        if raw is None:
            lams.append(tlam)
            continue
        raw = min(max(raw, 0.0), 0.1)
        w = n_trans / (n_trans + SHRINK_K) if n_trans > 0 else 0.0
        lams.append(w * raw + (1.0 - w) * tlam)
        a_i = KAPPA * last_rates[i] / 100.0
        for s in range(1, len(states)):
            t_prev, n_prev, r_prev = states[s - 1]
            t_cur, n_cur, r_cur = states[s]
            dn = n_cur - n_prev
            dt = max(t_cur - t_prev, 0.0)
            if dn <= 0 or dt <= 0:
                continue
            rp, rc = r_prev[i], r_cur[i]
            if rp is None or rc is None:
                continue
            e_prev = rp / 100.0 * n_prev
            de = rc / 100.0 * n_cur - e_prev
            pred = a_i * dn + lams[i] * (n_prev - e_prev) * dt
            rel_res.append(abs(de - pred) / max(e_prev, 1.0))
    rel_resid = 0.0
    if rel_res:
        rel_res.sort()
        rel_resid = rel_res[len(rel_res) // 2]
    return lams, tier_lams, rel_resid


def fit_n_target(states, t_target):
    """Players tracked at the target from a power law in hours since
    release, fitted on the last 24 hours of distinct states only.

    The launch-night catch-up burst (a 6x jump in one refresh) would
    dominate a fit over the whole history and explode the extrapolation,
    so only the recent regime is used; the slope is capped and so is the
    total growth, because a tracked-player count cannot keep multiplying.
    Returns (n_target, sig_log_n)."""
    n_last = states[-1][1]
    t_last = states[-1][0]
    recent = [s for s in states if s[0] >= t_last - 24.0]
    use = recent if len(recent) >= 3 else states
    xs = [math.log1p(t) for t, _, _ in use]
    ys = [math.log(n) for _, n, _ in use]
    _, b_n, rsd_n, sxx_n, mx_n = fit_line(xs, ys)
    # Cap the slope, pivoting on the data's centroid: capping b without
    # refitting the intercept would swing the line through the origin
    # and could push the forecast below the last value.
    b_n = min(max(b_n, 0.0), 2.0)
    my_n = sum(ys) / len(ys)
    x_t = math.log1p(t_target)
    n_target = math.exp(my_n + N_SHRINK * b_n * (x_t - mx_n))
    n_target = min(max(n_target, n_last), n_last * 20.0)
    if sxx_n > 0 and len(xs) > 2:
        sig_log_n = rsd_n * math.sqrt(
            1.0 + 1.0 / len(xs) + (x_t - mx_n) ** 2 / sxx_n)
    else:
        sig_log_n = 0.30
    sig_log_n = min(max(sig_log_n, 0.05), 0.6)
    return n_target, sig_log_n


def project_earners(e_last, n_last, n_target, a_i, lam_i, horizon_h):
    """Earners at the target by integrating
    dE/dt = a_i * N'(t) + lam_i * (N(t) - E(t)) with N growing
    exponentially from n_last to n_target over the horizon.
    Deterministic fixed-step integration (100 steps)."""
    if horizon_h <= 0:
        return e_last
    steps = 100
    dt = horizon_h / steps
    log_growth = math.log(n_target / n_last) / horizon_h if n_target > 0 \
        else 0.0
    e = e_last
    n = float(n_last)
    for _ in range(steps):
        n_next = n * math.exp(log_growth * dt)
        n_prime = (n_next - n) / dt
        e += (a_i * n_prime + lam_i * max(n - e, 0.0)) * dt
        n = n_next
    return max(e, 0.0)


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    tiers = [t["tier"] for t in trophies]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    t_target = max((target - RELEASE).total_seconds() / 3600.0, 0.0)

    last_rates = last_known_rates(case)
    # defensive fallback if a trophy was never seen: tier-typical rate
    tier_typical = {"platinum": 0.3, "gold": 0.5, "silver": 0.6,
                    "bronze": 0.8}
    for i in range(n_tr):
        if last_rates[i] is None:
            last_rates[i] = tier_typical.get(tiers[i], 0.8)

    states = distinct_states(case)
    n_last = states[-1][1] if states else 638

    rows = []
    if len(states) >= 2:
        rows = _forecast_driven(case, trophies, tiers, states,
                                last_rates, n_last, horizon_h,
                                horizon_d, t_target)
    else:
        rows = _forecast_flat(case, trophies, tiers, last_rates,
                              horizon_h, horizon_d)
    return {"trophies": rows}


def _forecast_driven(case, trophies, tiers, states, last_rates, n_last,
                     horizon_h, horizon_d, t_target):
    # --- players tracked at the target (recent-regime power law) ---
    n_target, sig_log_n = fit_n_target(states, t_target)

    lams, _, rel_resid = transition_fits(states, len(trophies), tiers,
                                         last_rates)
    sigma_base = math.sqrt(
        (SIGMA_BASE + SIGMA_PER_ROOT_DAY * math.sqrt(horizon_d)) ** 2
        + (0.4 * sig_log_n) ** 2
        + min(rel_resid * 2.0, 0.6) ** 2)

    rows = []
    for i, t in enumerate(trophies):
        r_last = last_rates[i]
        if r_last >= 99.0:
            rows.append({"id": t["id"], "q": _near_universal_q(r_last)})
            continue
        e_last = r_last / 100.0 * n_last
        a_i = KAPPA * r_last / 100.0
        e_t = project_earners(e_last, n_last, n_target, a_i, lams[i],
                              horizon_h)
        r_med = clamp_pct(100.0 * e_t / n_target)
        qs = normal_quantiles(r_med, sigma_base)
        qs = [round(q, 4) for q in qs]
        rows.append({"id": t["id"], "q": _monotone(qs)})
    return rows


def _forecast_flat(case, trophies, tiers, last_rates, horizon_h,
                   horizon_d):
    # No refresh observed: mixture over "page stays frozen" vs "a refresh
    # lands before the target".  The refresh branch applies the prior
    # dilution once per expected refresh, capped at three refreshes.
    w0 = math.exp(-horizon_h / REFRESH_MEAN_HOURS)
    k = max(1, min(int(round(horizon_h / REFRESH_MEAN_HOURS)), 3))
    per_refresh = (1.0 + (REFRESH_MULT - 1.0) * KAPPA) / REFRESH_MULT
    sig_refresh = SIGMA_REFRESH + 0.25 * math.sqrt(horizon_d)
    rows = []
    for i, t in enumerate(trophies):
        r_last = last_rates[i]
        if r_last >= 99.0:
            rows.append({"id": t["id"], "q": _near_universal_q(r_last)})
            continue
        r_ref = clamp_pct(r_last * (per_refresh ** k))
        branches = [(w0, logit_pct(r_last), SIGMA_NO_REFRESH),
                    (1.0 - w0, logit_pct(r_ref), sig_refresh)]
        qs = mixture_quantiles(branches)
        qs = [round(q, 4) for q in qs]
        rows.append({"id": t["id"], "q": _monotone(qs)})
    return rows


def _monotone(qs):
    out = []
    cur = 0.01
    for q in qs:
        cur = max(cur, min(q, 99.99))
        out.append(round(cur, 4))
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
