#!/usr/bin/env python
"""Benchmark compute.precision: wall-clock, memory and divergence per mode.

Produces the numbers that back the performance claims in the README. Run it on
an IDLE GPU -- a concurrent job makes the timings meaningless.

    PYTORCH_JIT=0 TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 \
        python tests/bench_precision.py --config examples/re180_open/config.yaml

Reports steps/s, peak memory and the divergence floor for float64, mixed and
float32. Expect ~2x, not 64x: these kernels are memory-bandwidth bound, so
halving the bytes is what you actually collect. The fp64:fp32 arithmetic ratio
(1/64 on consumer NVIDIA parts) barely enters.
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_default_dtype(torch.float64)


def bench_one(base_config, precision, n_steps, n_warmup, out_dir):
    from torchannel.solver import ChannelFlow
    from torchannel.utils import compute_divergence

    cfg = yaml.safe_load(open(base_config))
    cfg.setdefault('compute', {})['precision'] = precision
    cfg['statistics'] = {'n_stats': 0}
    cfg['output'] = {'results_folder': out_dir, 'n_out': 10**9, 'n_save': 10**9}
    cfg['time']['t_max'] = 1e9
    cfg['time']['n_steps'] = 10**9
    # Fixed dt: adaptive dt would make the three runs take different step counts
    # and quietly change what is being compared.
    cfg['time']['dt_update_interval'] = 0
    path = os.path.join(out_dir, f'_bench_{precision}.yaml')
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(path, 'w'))

    flow = ChannelFlow(config_file=path)
    dt = flow.dt

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(n_warmup):          # torch.compile / CUDA-graph capture
        flow.step_imex(dt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        flow.step_imex(dt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    div = compute_divergence(flow.u, flow.v, flow.w, flow.nx, flow.ny, flow.nz,
                             flow.dx, flow.dy, flow.dz_f_c)
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else float('nan'))

    return {
        'precision': precision,
        's_per_step': elapsed / n_steps,
        'steps_per_s': n_steps / elapsed,
        'peak_gib': peak,
        'max_div': float(div.abs().max()),
        'cells': flow.nx * flow.ny * flow.nz,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', default='examples/re180_open/config.yaml')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--out', default='/tmp/torchannel_bench')
    p.add_argument('--precisions', nargs='+',
                   default=['float64', 'mixed', 'float32'])
    a = p.parse_args()

    results = []
    for prec in a.precisions:
        r = bench_one(a.config, prec, a.steps, a.warmup, a.out)
        results.append(r)
        print(f"\n>>> {prec}: {r['s_per_step'] * 1e3:.1f} ms/step, "
              f"peak {r['peak_gib']:.2f} GiB, max|div| {r['max_div']:.2e}\n",
              flush=True)

    base = next((r for r in results if r['precision'] == 'float64'), results[0])
    print("\n" + "=" * 78)
    print(f"Grid: {base['cells'] / 1e6:.1f}M cells   "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print("=" * 78)
    print(f"{'precision':<10} {'ms/step':>10} {'steps/s':>10} {'speedup':>9} "
          f"{'peak GiB':>10} {'max|div|':>11}")
    for r in results:
        print(f"{r['precision']:<10} {r['s_per_step'] * 1e3:>10.1f} "
              f"{r['steps_per_s']:>10.2f} "
              f"{base['s_per_step'] / r['s_per_step']:>8.2f}x "
              f"{r['peak_gib']:>10.2f} {r['max_div']:>11.2e}")
    print("=" * 78)
    print("Reminder: reduced precision is memory-bandwidth limited here, so ~2x is")
    print("the honest expectation. The halved memory (bigger grids on the same")
    print("card) is often the more useful half.")


if __name__ == '__main__':
    main()
