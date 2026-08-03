# Opposition control at Re_τ = 180

A worked example of driving torChannel as a **control environment** rather than
as a batch job. The solver is stepped from a Python loop; each action reads the
wall shear field and writes blowing/suction back to the wall, in-process.

The control law is Choi, Moin & Kim (1994):

> v_wall(x, z) = −v(x, y_d, z)

— oppose the wall-normal velocity sensed at a detection plane y_d. At
y_d⁺ ≈ 10–15 this gives **~20–25 % drag reduction** at Re_τ = 180, which makes
the example self-validating: if the API is wired up correctly, you get a
published number back.

## Run

Start from an **equilibrated** turbulent field — otherwise you measure the
transition transient, not the control:

```bash
# 1. get a turbulent field (see ../re180_open/)
sbatch slurm/re180_open_gb10.sh

# 2. run the controller
python examples/opposition_control/run.py --field results_re180_open/fields.npz
```

Useful flags: `--detection-z-plus` (sweep it — drag reduction degrades and
eventually reverses as y_d⁺ grows past ~25), `--action-interval`, `--n-actions`,
`--max-action`.

## The API in ten lines

```python
from torchannel.control import ChannelFlowEnv, OppositionControl

env = ChannelFlowEnv("examples/re180_open/config.yaml", action_interval=0.05)
obs = env.reset()                       # obs = wall shear stress field
policy = OppositionControl(env, detection_z_plus=15.0)

for _ in range(400):
    obs, reward, done, info = env.step(policy(obs))
    print(info["drag_reduction"], info["u_tau"])
```

`env.to_gym()` wraps this as a `gymnasium.Env` for RL libraries (requires
`gymnasium`; it is not a core dependency).

## Things worth knowing

- **Actions are mean-subtracted.** With periodic x/y and no flux through the
  top, a non-zero net wall flux violates global mass conservation, and the
  all-Neumann pressure Poisson problem has no solution — its compatibility
  condition is exactly zero net flux. `set_wall_velocity` enforces this rather
  than letting the projection diverge.
- **Actuation does not break the projection.** Verified: max|div| stays at
  ~10⁻¹³ under sustained actuation, with net wall flux ~10⁻¹⁶.
- **`control_shape` can be coarser than the wall grid** (it must divide it
  exactly). A coarse action grid is usually what you want for RL; it is
  upsampled by nearest-neighbour repeat.
- **`reset()` does not reset the flow field.** It clears the actuation and
  re-baselines the drag. Restoring a specific turbulent state means reloading a
  checkpoint, which you control through the config.
- **Backprop through the solver is not supported.** The solver uses in-place
  ops, preallocated buffers and optional CUDA-graph capture, all of which fight
  autograd. This environment is for gradient-free control (RL, opposition,
  parameter sweeps).
