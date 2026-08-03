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
import argparse
import io
import os
import sys
import tarfile
import tempfile
import urllib.request

import numpy as np

BASE_MKM = "https://turbulence.oden.utexas.edu/data/MKM"
BASE_LM = "https://turbulence.oden.utexas.edu/channel2015/data"
BASE_VRE = "http://www.vremanresearch.nl"

# One archive holding all three MKM cases plus budgets, spectra, correlations
# and PDFs. --tarball pulls this once instead of one HTTP request per profile,
# which also makes a fully offline rebuild possible.
#
# PROVENANCE, because the headers invite a specific misreading: every file in it
# reads "Authors: Moser, Kim & Mansour ... 1999", and the line "Numerical Method:
# Kim, Moin & Moser, 1987" cites the METHOD, not the dataset. The chan180 case is
# MKM's own re-computation on 128x129x128 over 4pi x 2 x 4pi/3 at Re_tau=178.12,
# not KMM's 1987 run (192x129x160 over 4pi x 2 x 2pi at Re_tau=180): the spectra
# files give it away via nx = 128 and dk_z = 1.5 -> L_z = 4pi/3. KMM's own
# statistics are not publicly distributed. The lineage is recorded in the CSV
# headers so nobody has to re-derive this.
URL_TARBALL = f"{BASE_MKM}/chandata.tar.gz"
MKM_METHOD = "Kim, Moin & Moser (1987), J. Fluid Mech. 177, 133-166"

# Vreman & Kuerten's Re_tau=180 comparison databases. Same box as MKM chan180
# (4pi x 2 x 4pi/3) but far better resolved, and available in BOTH spectral and
# second-order finite-difference form -- which makes them the right reference for
# separating "under-resolved" from "FD differs from spectral".
VREMAN = {
    "vreman180_s2":  ("Chan180_S2",  "Vreman & Kuerten, spectral 384x193x192"),
    "vreman180_fd2": ("Chan180_FD2", "Vreman & Kuerten, finite-difference 512x256x256"),
}

MKM_CITE = "Moser, Kim & Mansour (1999), Phys. Fluids 11(4), 943-945"

# name -> (means_url, stress_url, kind, citation)
DATASETS = {
    "mkm180": (
        f"{BASE_MKM}/chan180/profiles/chan180.means",
        f"{BASE_MKM}/chan180/profiles/chan180.reystress",
        "mkm", MKM_CITE,
    ),
    "mkm395": (
        f"{BASE_MKM}/chan395/profiles/chan395.means",
        f"{BASE_MKM}/chan395/profiles/chan395.reystress",
        "mkm", MKM_CITE,
    ),
    "mkm590": (
        f"{BASE_MKM}/chan590/profiles/chan590.means",
        f"{BASE_MKM}/chan590/profiles/chan590.reystress",
        "mkm", MKM_CITE,
    ),
    "lm550": (
        f"{BASE_LM}/LM_Channel_0550_mean_prof.dat",
        f"{BASE_LM}/LM_Channel_0550_vel_fluc_prof.dat",
        "lm",
        "Lee & Moser (2015), J. Fluid Mech. 774, 395-415",
    ),
}


def _fetch(url):
    """Read a source, which may be a URL or a path inside an extracted tarball."""
    if not url.startswith(("http://", "https://")):
        with open(url, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def unpack_tarball(dest=None):
    """Download and extract chandata.tar.gz; return the directory holding chanNNN/.

    Note the server does not answer on port 80 -- use https.
    """
    dest = dest or tempfile.mkdtemp(prefix="mkm-")
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, "chandata.tar.gz")
    if not os.path.exists(archive):
        with urllib.request.urlopen(URL_TARBALL, timeout=300) as r, open(archive, "wb") as fh:
            fh.write(r.read())
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers()
                   if not (m.name.startswith("/") or ".." in m.name.split("/"))]
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:                          # filter= is Python 3.12+
            tf.extractall(dest, members=members)
    return dest


def _retarget_to_tarball(name, root):
    """Local paths for an MKM case inside the extracted archive, or None."""
    case = name.replace("mkm", "chan")            # mkm180 -> chan180
    base = os.path.join(root, case, "profiles", case)
    local = (f"{base}.means", f"{base}.reystress")
    return local if all(os.path.exists(p) for p in local) else None


def _method(text):
    """The 'Numerical Method:' line, so the CSV records the code's lineage."""
    for line in text.splitlines():
        s = line.strip().lstrip("#%").strip()
        if s.lower().startswith("numerical method"):
            return s.split(":", 1)[1].strip()
    return None


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


def _vreman_rows(text):
    rows = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t[0] in "%#":
            continue
        try:
            rows.append([float(x) for x in t.split()])
        except ValueError:
            continue
    return np.asarray(rows)


def convert_vreman(name, outdir):
    """Vreman & Kuerten Re_tau=180 -> torChannel axes.

    Their y is wall-normal, so the remap is the same as for MKM: their v is our
    w, their w is our v, and their <u'v'> is our <u'w'>. Confirmed against the
    near-wall asymptotics (their rms(v) peaks at 0.84, rms(w) at 1.09 -- the
    wall-normal component must be the smaller one).

    Data are normalised on u_tau and H with nu = 1/180, so the tabulated <u> IS
    U+ and the rms values ARE already divided by u_tau; the header's computed
    u_tau (0.9999...) is divided out for exactness.
    """
    tag, cite = VREMAN[name]
    txt = {c: _fetch(f"{BASE_VRE}/{tag}_basic_{c}.txt") for c in "uvw"}
    ut = 1.0
    for line in txt["u"].splitlines():
        if "Computed u_tau" in line:
            ut = float(line.split(":")[1])
            break
    U, V, W = (_vreman_rows(txt[c]) for c in "uvw")
    zp = U[:, 0]
    out = []
    for i in range(len(zp)):
        out.append((zp[i] / 180.0,          # z/delta
                    zp[i],                  # z+
                    U[i, 1] / ut,           # U+
                    (U[i, 2] / ut) ** 2,    # uu+   (their u  -> our u)
                    (W[i, 2] / ut) ** 2,    # vv+   (their w  -> our v, spanwise)
                    (V[i, 2] / ut) ** 2,    # ww+   (their v  -> our w, wall-normal)
                    V[i, 5] / ut ** 2))     # uw+   (their <u'v'> -> our <u'w'>)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.csv")
    with open(path, "w") as fh:
        fh.write(f"# {cite}\n")
        fh.write(f"# Vreman & Kuerten, 'Comparison of DNS databases of turbulent\n")
        fh.write(f"#   channel flow at Re_tau=180', Phys. Fluids 26, 015102 (2014)\n")
        fh.write(f"# Downloaded from {BASE_VRE}/{tag}_basic_[uvw].txt\n")
        fh.write(f"# Domain 4pi x 2 x 4pi/3, computed u_tau = {ut}\n")
        fh.write("# Actual Re_tau = 180.0\n")
        fh.write("# Closed channel, lower half (z/delta = 0..1). Reynolds stresses are\n"
                 "# REMAPPED to torChannel axes: their v (wall-normal) -> ww,\n"
                 "# their w (spanwise) -> vv, their <u'v'> -> uw.\n")
        fh.write("z_delta,z_plus,U_plus,uu_plus,vv_plus,ww_plus,uw_plus\n")
        for r in out:
            fh.write(",".join(f"{v:.6e}" for v in r) + "\n")
    return path, 180.0, len(out)


def convert(name, outdir, tarball_root=None):
    means_url, stress_url, kind, citation = DATASETS[name]
    # Read locally when the archive is available, but keep the canonical URLs in
    # the header: an extraction directory is not provenance anyone can follow.
    src = (tarball_root and kind == "mkm" and _retarget_to_tarball(name, tarball_root)) \
        or (means_url, stress_url)
    means_txt, stress_txt = _fetch(src[0]), _fetch(src[1])
    re_tau = _re_tau(means_txt)
    method = _method(means_txt)

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
    if method:
        # Recorded because this line is the source of a recurring misreading: it
        # is the NUMERICAL METHOD, not the authorship of these statistics.
        buf.write(f"# Numerical method: {method}\n")
    buf.write(f"# Downloaded from {means_url}\n")
    buf.write(f"#                 {stress_url}\n")
    if src[0] != means_url:
        buf.write(f"#   (read from the bundle {URL_TARBALL})\n")
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
    p.add_argument("--only", choices=sorted(DATASETS) + sorted(VREMAN),
                   help="convert one dataset")
    p.add_argument("--tarball", nargs="?", const=True, default=None, metavar="DIR",
                   help="take the MKM cases from chandata.tar.gz (one download for "
                        "all three) instead of per-profile URLs; pass a directory "
                        "to reuse an already-extracted copy")
    a = p.parse_args(argv)

    root = None
    if a.tarball is not None:
        root = unpack_tarball(None if a.tarball is True else a.tarball)
        print(f"MKM archive extracted to {root}")

    all_names = sorted(DATASETS) + sorted(VREMAN)
    names = [a.only] if a.only else all_names
    for n in names:
        try:
            path, re_tau, npts = (
                convert_vreman(n, a.outdir) if n in VREMAN
                else convert(n, a.outdir, tarball_root=root))
        except Exception as exc:  # network or format problem: report, keep going
            print(f"{n:8s} FAILED: {exc}", file=sys.stderr)
            continue
        print(f"{n:8s} Re_tau = {re_tau:8.2f}  {npts:4d} points  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
