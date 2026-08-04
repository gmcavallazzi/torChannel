#!/bin/bash
# Watch a CaNS run's velstats output, re-average it into a torChannel-format npz
# and regenerate the comparison figures each time new snapshots appear.
#
# CaNS writes one velstats_fld_*.out per output step, so "new data" is simply a
# larger file count. Figures land in figures_local/re180_cans/ and are drawn by
# the same plot_statistics.py path as the torChannel run, so the two are
# directly comparable -- everything except the 2D spectra, which CaNS does not
# write.
#
# Usage: scripts/watch_cans.sh [cans_data_dir] [out_dir] [every_n_new_files]
set -u
DATA=${1:-/home/giorgio/CaNS_DRL/run_re180_tc/data}
OUT=${2:-figures_local/re180_cans}
EVERY=${3:-10}
NPZ=$OUT/cans_stats.npz

cd /home/giorgio/torChannel
source /etc/profile.d/modules.sh 2>/dev/null; module load texlive 2>/dev/null
mkdir -p "$OUT"

LAST=0
while true; do
    N=$(ls "$DATA"/velstats_fld_*.out 2>/dev/null | wc -l)
    if [ "$N" -ge $((LAST + EVERY)) ]; then
        LAST=$N
        # skip the first 3 files: the restart transient as CaNS re-derives its
        # own pressure and RK history from the transferred field
        python scripts/cans_stats_to_npz.py "$DATA" "$NPZ" --nu 3.5807e-4 \
            --Lx 12.566370614359172 --Ly 6.283185307179586 --skip 3 >/dev/null 2>&1 || continue
        for r in mkm180 vreman180_fd2; do
            python plot_statistics.py "$NPZ" --config examples/re180_closed/config.yaml \
                --reference "$r" --format png --output "$OUT/$r" >/dev/null 2>&1
        done
        python - "$NPZ" <<'PY'
import sys, numpy as np
sys.path.insert(0, '.')
from plot_statistics import compute_dUdz, load_reference
d = np.load(sys.argv[1]); nu = float(d['nu'])
z, Lz = d['z_c'], float(d['Lz']); delta = Lz/2
U, uu, ww, uw = d['U_mean'], d['uu_mean'], d['ww_mean'], d['uw_mean']
mir = lambda a: np.interp((2*delta - z)[::-1], z, a)[::-1]
U, uu, ww, uw = 0.5*(U+mir(U)), 0.5*(uu+mir(uu)), 0.5*(ww+mir(ww)), 0.5*(uw-mir(uw))
dUdz, dw, _ = compute_dUdz(U, z, Lz, open_channel=False); ut = np.sqrt(nu*abs(dw))
m = z <= delta; zp = (z*ut/nu)[m]
ur = (np.sqrt(np.maximum(uu,0))/ut)[m]; wr = (np.sqrt(np.maximum(ww,0))/ut)[m]
uwp = -(uw/ut**2)[m]
tot = uwp + (nu*dUdz/ut**2)[m]
band = (zp>20)&(zp<150)
r = load_reference('mkm180')
URf = np.interp(zp, r['z_plus'], np.sqrt(np.maximum(r['uu_plus'],0)))
print(f"CaNS n={int(d['n_samples']):4d}  Re_tau={ut*delta/nu:7.2f}  "
      f"totstress dev={np.abs(tot-(1-z[m]/delta)).max():.4f}  "
      f"u'rms pk {100*(ur.max()/np.sqrt(np.maximum(r['uu_plus'],0)).max()-1):+6.2f}%  "
      f"band {100*(ur[band].mean()/URf[band].mean()-1):+6.2f}%  "
      f"-uw {100*(uwp.max()/(-r['uw_plus']).max()-1):+6.2f}%", flush=True)
PY
    fi
    sleep 60
done
