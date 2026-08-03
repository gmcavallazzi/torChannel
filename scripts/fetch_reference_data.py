"""Download published channel-flow DNS profiles and convert them to torChannel's
conventions, writing compact CSVs into ``torchannel/data/reference/``.

Sources (both freely distributed by the Oden Institute, UT Austin):

  MKM   Moser, Kim & Mansour (1999), "DNS of turbulent channel flow up to
        Re_tau = 590", Physics of Fluids 11(4), 943-945.
  LM    Lee & Moser (2015), "Direct numerical simulation of turbulent channel
        flow up to Re_tau = 5200", J. Fluid Mech. 774, 395-415.

AXIS CONVENTION -- the reason this script exists rather than a raw download.
Both references use y as the WALL-NORMAL direction and z as spanwise:

    reference:   x streamwise (u),  y wall-normal (v),  z spanwise (w)
    torChannel:  x streamwise (u),  y spanwise    (v),  z wall-normal (w)

so the Reynolds stresses must be remapped, not just renamed:

    R_uu -> uu       R_vv (wall-normal) -> ww
    R_uv -> uw       R_ww (spanwise)    -> vv

Overlaying the reference columns in file order would silently swap v'v' and
w'w'. The output CSVs use torChannel's naming, already remapped.

The published profiles are CLOSED-channel data spanning y/delta = 0..1 (the
lower half, by symmetry). They are therefore directly comparable to an open
channel in the near-wall region; expect a real, physical difference toward the
centreline, where a symmetry/free-slip boundary suppresses the large-scale
motions that cross the centreline of a closed channel.

Usage:
    python scripts/fetch_reference_data.py [--outdir torchannel/data/reference]
"""

import argparse
import io
import os
import sys
import urllib.request

BASE_MKM = "https://turbulence.oden.utexas.edu/data/MKM"
BASE_LM = "https://turbulence.oden.utexas.edu/channel2015/data"

# name -> (means_url, stress_url, kind, citation)
DATASETS = {
    "mkm180": (
        f"{BASE_MKM}/chan180/profiles/chan180.means",
        f"{BASE_MKM}/chan180/profiles/chan180.reystress",
        "mkm",
        "Moser, Kim & Mansour (1999), Phys. Fluids 11(4), 943-945",
    ),
    "mkm590": (
        f"{BASE_MKM}/chan590/profiles/chan590.means",
        f"{BASE_MKM}/chan590/profiles/chan590.reystress",
        "mkm",
        "Moser, Kim & Mansour (1999), Phys. Fluids 11(4), 943-945",
    ),
    "lm550": (
        f"{BASE_LM}/LM_Channel_0550_mean_prof.dat",
        f"{BASE_LM}/LM_Channel_0550_vel_fluc_prof.dat",
        "lm",
        "Lee & Moser (2015), J. Fluid Mech. 774, 395-415",
    ),
}


def _fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def _numeric_rows(text):
    """Rows of floats, skipping '#'/'%' comments and any unparseable line."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#%":
            continue
        try:
            rows.append([float(t) for t in s.split()])
        except ValueError:
            continue
    return rows


def _re_tau(text, default=None):
    """The dataset's ACTUAL Re_tau, which is never exactly the nominal one."""
    for line in text.splitlines():
        s = line.strip()
        if not (s.startswith("#") or s.startswith("%")):
            continue
        if "Re_tau" in s:
            for tok in s.replace("=", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if v > 1.0:  # skip the "5200" in the LM paper title line
                    return v
    return default


def convert(name, outdir):
    means_url, stress_url, kind, citation = DATASETS[name]
    means_txt, stress_txt = _fetch(means_url), _fetch(stress_url)
    re_tau = _re_tau(means_txt)

    m, s = _numeric_rows(means_txt), _numeric_rows(stress_txt)
    if not m or not s:
        raise RuntimeError(f"{name}: no numeric rows parsed")
    if len(m) != len(s):
        raise RuntimeError(f"{name}: means/stress row mismatch ({len(m)} vs {len(s)})")

    # Both layouts: col0 = y/delta, col1 = y+.
    #   MKM means:  y, y+, Umean, dUmean/dy, Wmean, dWmean/dy, Pmean
    #   LM  means:  y/delta, y+, U, dU/dy, W, P
    #   MKM/LM stresses: y, y+, R_uu, R_vv, R_ww, R_uv, R_uw, R_vw[, k]
    # MKM is normalised by u_tau already, so col2 is U+ in both cases.
    out = []
    for mr, sr in zip(m, s):
        z_delta, z_plus, U_plus = mr[0], mr[1], mr[2]
        r_uu, r_vv, r_ww, r_uv = sr[2], sr[3], sr[4], sr[5]
        # Remap reference (y wall-normal) -> torChannel (z wall-normal).
        out.append((z_delta, z_plus, U_plus, r_uu, r_ww, r_vv, r_uv))
        #                                    uu     vv     ww    uw
        #   reference R_ww (spanwise)    -> torChannel vv
        #   reference R_vv (wall-normal) -> torChannel ww

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.csv")
    buf = io.StringIO()
    buf.write(f"# {citation}\n")
    buf.write(f"# Downloaded from {means_url}\n")
    buf.write(f"#                 {stress_url}\n")
    buf.write(f"# Actual Re_tau = {re_tau}\n")
    buf.write(
        "# Closed channel, lower half (z/delta = 0..1). Reynolds stresses are\n"
        "# REMAPPED to torChannel axes: reference R_vv (wall-normal) -> ww,\n"
        "# reference R_ww (spanwise) -> vv, reference R_uv -> uw.\n"
        "# Normalised by u_tau (stresses by u_tau^2).\n"
    )
    buf.write("z_delta,z_plus,U_plus,uu_plus,vv_plus,ww_plus,uw_plus\n")
    for row in out:
        buf.write(",".join(f"{v:.6e}" for v in row) + "\n")
    with open(path, "w") as fh:
        fh.write(buf.getvalue())
    return path, re_tau, len(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outdir", default="torchannel/data/reference")
    p.add_argument("--only", choices=sorted(DATASETS), help="convert one dataset")
    a = p.parse_args(argv)

    names = [a.only] if a.only else sorted(DATASETS)
    for n in names:
        try:
            path, re_tau, npts = convert(n, a.outdir)
        except Exception as exc:  # network or format problem: report, keep going
            print(f"{n:8s} FAILED: {exc}", file=sys.stderr)
            continue
        print(f"{n:8s} Re_tau = {re_tau:8.2f}  {npts:4d} points  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
