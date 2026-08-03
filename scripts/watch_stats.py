"""Watch a running simulation's statistics checkpoint and re-log convergence.

Each time the solver rewrites ``turbulence_stats_state.npz`` this regenerates
the figure set and prints ONE line per reference saying how far each key
quantity sits from the published value, and -- more usefully -- whether that
distance grew or shrank since the previous batch. A snapshot tells you almost
nothing; the trend tells you whether more averaging is still buying anything.

The quantities are chosen to separate the two things that can be wrong:

  peak u'_rms, peak -<u'w'>   sensitive to u_tau, so they move if the friction
                              velocity is off even when the field is fine
  R_uw = -<u'w'>/(u'_rms w'_rms)   dimensionless: u_tau cancels EXACTLY, so a
                              gap here is the numerics, not the normalisation
  total-stress deviation      max| -<u'w'>+ + nu dU+/dz+ - (1 - z/delta) |,
                              a reference-free consequence of the momentum
                              equation; it converges to 0 or the run is wrong

Usage:
    python scripts/watch_stats.py results_re180_closed \
        --config examples/re180_closed/config.yaml \
        --reference mkm180 vreman180_fd2 --output figures_local/re180_closed
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_statistics import (compute_dUdz, load_reference,   # noqa: E402
                             _add_plane_mean_variance)


def _metrics(state_path, nu, open_channel):
    """Reduce a statistics checkpoint to the handful of numbers worth watching."""
    d = np.load(state_path)
    n = int(d["n_samples"])
    if n < 1:
        return None
    z = d["z_c"]
    Lz = float(d["dz_f"].sum())
    delta = Lz if open_channel else Lz / 2.0

    U = d["U_sum"] / n
    uu = d["uu_sum"] / n
    vv = d["vv_sum"] / n
    ww = d["ww_sum"] / n
    uw = d["uw_sum"] / n
    # Same correction the figures apply, so the logged numbers and the plots
    # cannot drift apart.
    uu, vv, ww, uw = _add_plane_mean_variance(d, n, U, uu, vv, ww, uw)

    # Closed channel: FOLD about the centreline. The two halves are statistically
    # identical, so averaging them doubles the sample count for free -- and it
    # does so for precisely the quantity that converges slowest here, the
    # large-scale streamwise energy in the log layer. Must happen BEFORE dU/dz is
    # taken, or the folded shear stress would be checked against an unfolded
    # viscous term and the total-stress residual would be meaningless.
    if not open_channel:
        mirror = lambda a: np.interp((2.0 * delta - z)[::-1], z, a)[::-1]
        U = 0.5 * (U + mirror(U))          # even about the centreline
        uu = 0.5 * (uu + mirror(uu))       # even
        vv = 0.5 * (vv + mirror(vv))       # even
        ww = 0.5 * (ww + mirror(ww))       # even
        uw = 0.5 * (uw - mirror(uw))       # ODD: <u'w'> changes sign

    dUdz, dUdz_wall, _ = compute_dUdz(U, z, Lz, open_channel=open_channel)
    u_tau = float(np.sqrt(nu * abs(dUdz_wall)))

    m = z <= delta if not open_channel else np.ones_like(z, dtype=bool)
    z_plus = (z * u_tau / nu)[m]
    u_rms = (np.sqrt(np.maximum(uu, 0)) / u_tau)[m]
    w_rms = (np.sqrt(np.maximum(ww, 0)) / u_tau)[m]
    uw_plus = -(uw / u_tau ** 2)[m]
    R_uw = uw_plus / np.maximum(u_rms * w_rms, 1e-30)

    total = uw_plus + (nu * dUdz / u_tau ** 2)[m]
    exact = 1.0 - z[m] / delta

    return dict(n=n, n_samples=n, u_tau=u_tau, Re_tau=u_tau * delta / nu,
                z_plus=z_plus, u_rms=u_rms, w_rms=w_rms, uw=uw_plus, R=R_uw,
                stress_dev=float(np.abs(total - exact).max()))


def _compare(cur, ref_name):
    """Percentage gaps against one published dataset."""
    r = load_reference(ref_name)
    zr = r["z_plus"]
    ur = np.sqrt(np.maximum(r["uu_plus"], 0))
    wr = np.sqrt(np.maximum(r["ww_plus"], 0))
    uwr = -r["uw_plus"]
    Rr = uwr / np.maximum(ur * wr, 1e-30)

    # R_uw is compared over the log-layer band, where it is flattest and where
    # both datasets are well resolved; the peaks are compared as peaks.
    band = (cur["z_plus"] > 20) & (cur["z_plus"] < 150)
    R_ref = np.interp(cur["z_plus"][band], zr, Rr).mean()
    # Band AND peak: they disagree materially (the deficit is worse in the log
    # layer than at the peak), and that gap is itself the diagnostic -- the log
    # layer is large-scale dominated, so a band-heavy deficit points at
    # large-scale convergence rather than at grid resolution.
    u_band = np.interp(cur["z_plus"][band], zr, ur).mean()
    return dict(
        d_urms_band=100 * (cur["u_rms"][band].mean() - u_band) / u_band,
        d_urms=100 * (cur["u_rms"].max() - ur.max()) / ur.max(),
        d_wrms=100 * (cur["w_rms"].max() - wr.max()) / wr.max(),
        d_uw=100 * (cur["uw"].max() - uwr.max()) / uwr.max(),
        d_R=100 * (cur["R"][band].mean() - R_ref) / R_ref,
    )


def _arrow(prev, cur, key, tol=0.05):
    if prev is None or key not in prev:
        return ""
    a, b = abs(prev[key]), abs(cur[key])
    if abs(b - a) < tol:
        return "  ="
    return " ->" if b < a else " <-"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", help="results folder holding the checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--reference", nargs="+", default=["mkm180"])
    p.add_argument("--output", required=True, help="figure output prefix directory")
    p.add_argument("--state", default="turbulence_stats_state.npz")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--history", default=None, help="JSON trend log (default: <output>/trend.json)")
    p.add_argument("--once", action="store_true", help="log a single batch and exit")
    a = p.parse_args(argv)

    import yaml
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    nu = 1.0 / float(cfg["flow"]["Re"])
    open_channel = cfg.get("boundary_conditions", {}).get("top_wall", {}).get("type") == "neumann"

    state = os.path.join(a.results, a.state)
    os.makedirs(a.output, exist_ok=True)
    hist_path = a.history or os.path.join(a.output, "trend.json")
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else []

    tmp = os.path.join(a.output, "_snapshot.npz")
    last_mtime = None
    while True:
        if os.path.exists(state):
            mtime = os.path.getmtime(state)
            if mtime != last_mtime:
                last_mtime = mtime
                time.sleep(3)                       # let the write settle
                try:
                    shutil.copy(state, tmp)
                    cur = _metrics(tmp, nu, open_channel)
                except Exception as exc:            # partial write: try again next tick
                    print(f"  (checkpoint unreadable: {exc})", flush=True)
                    cur = None

                if cur is not None:
                    for ref in a.reference:
                        subprocess.run(
                            [sys.executable, "plot_statistics.py", tmp, "--checkpoint",
                             "--config", a.config, "--reference", ref, "--format", "png",
                             "--output", os.path.join(a.output, ref)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    print(f"n={cur['n_samples']:4d}  Re_tau={cur['Re_tau']:7.2f}  "
                          f"u_tau={cur['u_tau']:.6f}  total-stress dev={cur['stress_dev']:.4f}",
                          flush=True)
                    entry = dict(n=cur["n_samples"], Re_tau=cur["Re_tau"],
                                 stress_dev=cur["stress_dev"])
                    prev = history[-1] if history else None
                    for ref in a.reference:
                        c = _compare(cur, ref)
                        entry[ref] = c
                        pr = prev.get(ref) if prev else None
                        print(f"    vs {ref:14s} u'rms pk {c['d_urms']:+6.2f}%{_arrow(pr, c, 'd_urms')}"
                              f" band {c['d_urms_band']:+6.2f}%{_arrow(pr, c, 'd_urms_band')}"
                              f" | w'rms {c['d_wrms']:+6.2f}%{_arrow(pr, c, 'd_wrms')}"
                              f" | -uw {c['d_uw']:+6.2f}%{_arrow(pr, c, 'd_uw')}"
                              f" | R_uw {c['d_R']:+6.2f}%{_arrow(pr, c, 'd_R')}",
                              flush=True)
                    history.append(entry)
                    with open(hist_path, "w") as fh:
                        json.dump(history, fh, indent=1)

                if a.once:
                    return 0
        elif a.once:
            print("no statistics checkpoint yet", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
