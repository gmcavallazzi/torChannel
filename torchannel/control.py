"""Drive the solver as a control environment.

torChannel is a PyTorch program, so a simulation can be stepped from Python
rather than only run as a batch job. That makes it usable directly as an
environment for flow control -- opposition control, reinforcement learning,
data assimilation -- without a co-simulation layer or file-based coupling.

    from torchannel.control import ChannelFlowEnv, OppositionControl

    env = ChannelFlowEnv("examples/re180_open/config.yaml", action_interval=0.05)
    obs = env.reset()
    policy = OppositionControl(env, detection_z_plus=15.0)
    for _ in range(200):
        obs, reward, done, info = env.step(policy(obs))
        print(info["drag_reduction"])

The observation is the wall shear-stress field, the action is wall-normal
blowing/suction, and the reward is drag reduction relative to the uncontrolled
baseline. A Gymnasium adapter is available via `to_gym()` if gymnasium is
installed; the core class deliberately has no such dependency.

Actions are mean-subtracted by the solver (see ChannelFlow.set_wall_velocity):
with periodic x/y and no flux through the top, a non-zero net wall flux has no
consistent pressure solution.
"""

import numpy as np
import torch

from torchannel.solver import ChannelFlow


class ChannelFlowEnv:
    """A steppable channel-flow environment with wall blowing/suction.

    Args:
        config_file: path to a torChannel YAML config.
        action_interval: physical time each action is held for. The solver may
            take many (adaptively sized) sub-steps within one action.
        control_shape: (nx_c, ny_c) resolution of the action. Defaults to the
            full wall grid; a coarser grid is usually what you want for RL, and
            is upsampled to the wall by nearest-neighbour repeat.
        max_action: actions are clipped to +/- this, in units of U_bulk. Wall
            transpiration in control studies is typically a few percent of the
            bulk velocity; larger values are unphysical and destabilising.
        flow: an existing ChannelFlow to wrap instead of constructing one.
    """

    def __init__(self, config_file=None, action_interval=0.05,
                 control_shape=None, max_action=0.05, flow=None):
        if flow is None and config_file is None:
            raise ValueError("provide either config_file or flow")
        self.flow = flow if flow is not None else ChannelFlow(config_file=config_file)

        if self.flow.time_scheme != 'IMEX':
            raise NotImplementedError(
                f"the control API drives step_imex; time.scheme is "
                f"{self.flow.time_scheme!r}. Use IMEX.")

        self.action_interval = float(action_interval)
        self.max_action = float(max_action)
        self.control_shape = tuple(control_shape) if control_shape else (
            self.flow.nx, self.flow.ny)

        for n, full in zip(self.control_shape, (self.flow.nx, self.flow.ny)):
            if full % n != 0:
                raise ValueError(
                    f"control_shape {self.control_shape} must divide the wall "
                    f"grid {(self.flow.nx, self.flow.ny)} exactly")

        self._baseline_tau = None
        self._t0 = self.flow.time
        self._initial_state = None

    # -- observation ------------------------------------------------------

    def wall_shear(self):
        """Streamwise wall shear stress tau_w(x, y) at the bottom wall.

        nu * du/dz at the wall, one-sided from the first interior centre (u=0
        at the wall by no-slip). Shape (nx, ny).
        """
        u = self.flow.u
        dist = float(self.flow.z_c[1])
        # u is staggered in x: average the two faces to get a cell-centred value.
        u1 = 0.5 * (u[0:self.flow.nx, 1:self.flow.ny + 1, 1]
                    + u[1:self.flow.nx + 1, 1:self.flow.ny + 1, 1])
        return self.flow.nu * u1 / dist

    def observe(self):
        """Observation: wall shear stress, normalised by its own mean."""
        tau = self.wall_shear()
        return (tau / tau.mean()).to(torch.float32)

    def velocity_at_z_plus(self, z_plus, component='w'):
        """Wall-parallel plane of a velocity component at a given z+.

        The detection plane an opposition controller needs. z+ is measured with
        the CURRENT u_tau, so the plane index tracks the actual flow state.
        """
        u_tau = float(self.current_u_tau())
        z_target = z_plus * self.flow.nu / max(u_tau, 1e-12)
        z_int = self.flow.z_c[1:self.flow.nz + 1]
        k = int(torch.argmin(torch.abs(z_int - z_target)).item())
        nx, ny = self.flow.nx, self.flow.ny
        if component == 'w':
            # w is staggered in z: interpolate the two faces bounding cell k.
            return 0.5 * (self.flow.w[1:nx + 1, 1:ny + 1, k]
                          + self.flow.w[1:nx + 1, 1:ny + 1, k + 1])
        if component == 'u':
            return 0.5 * (self.flow.u[0:nx, 1:ny + 1, k + 1]
                          + self.flow.u[1:nx + 1, 1:ny + 1, k + 1])
        raise ValueError(f"unknown component {component!r}")

    def current_u_tau(self):
        from torchannel.utils import compute_u_tau
        return compute_u_tau(self.flow.u, self.flow.z_c, self.flow.nu,
                             top_wall_bc_type=self.flow.top_wall_bc_type)

    # -- dynamics ---------------------------------------------------------

    def reset(self, baseline_tau=None):
        """Reset actuation and the episode clock; return the first observation.

        Note this does NOT reset the flow field -- restoring a turbulent state
        means reloading a checkpoint, which the caller controls via the config.
        Continuing from the current state is the usual and cheaper choice.
        """
        self.flow.set_wall_velocity(None)
        self._t0 = self.flow.time
        self._baseline_tau = (float(self.wall_shear().mean())
                              if baseline_tau is None else float(baseline_tau))
        return self.observe()

    def _expand_action(self, action):
        a = torch.as_tensor(action, dtype=self.flow.dtype, device=self.flow.device)
        if a.shape != self.control_shape:
            a = a.reshape(self.control_shape)
        a = a.clamp(-self.max_action, self.max_action)
        rx = self.flow.nx // self.control_shape[0]
        ry = self.flow.ny // self.control_shape[1]
        if rx > 1 or ry > 1:
            a = a.repeat_interleave(rx, 0).repeat_interleave(ry, 1)
        return a

    def step(self, action):
        """Hold `action` for action_interval of physical time.

        Returns (observation, reward, done, info). `done` is always False --
        a channel flow has no terminal state; cap the episode length yourself.
        """
        self.flow.set_wall_velocity(self._expand_action(action))

        t_end = self.flow.time + self.action_interval
        n_sub = 0
        while self.flow.time < t_end:
            dt = self.flow.compute_cfl_dt()
            dt = min(float(dt), t_end - self.flow.time)
            self.flow.step_imex(dt)
            self.flow.time += dt
            self.flow.current_step += 1
            n_sub += 1
            if not np.isfinite(float(self.flow.u.abs().max())):
                raise RuntimeError(
                    f"solution diverged at t={self.flow.time:.4f}; "
                    f"max_action={self.max_action} may be too large")

        tau = float(self.wall_shear().mean())
        if self._baseline_tau is None:
            self._baseline_tau = tau
        dr = (self._baseline_tau - tau) / self._baseline_tau

        info = {
            'drag_reduction': dr,
            'tau_wall': tau,
            'baseline_tau': self._baseline_tau,
            'u_tau': float(self.current_u_tau()),
            'time': self.flow.time,
            'substeps': n_sub,
        }
        return self.observe(), dr, False, info

    # -- optional Gymnasium adapter ---------------------------------------

    def to_gym(self):
        """Wrap as a gymnasium.Env. Requires gymnasium (not a core dependency)."""
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:
            raise ImportError(
                "to_gym() needs gymnasium: pip install gymnasium") from exc

        env_self = self

        class _GymChannelFlow(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(env_self.flow.nx, env_self.flow.ny), dtype=np.float32)
                self.action_space = spaces.Box(
                    low=-env_self.max_action, high=env_self.max_action,
                    shape=env_self.control_shape, dtype=np.float32)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return env_self.reset().cpu().numpy(), {}

            def step(self, action):
                obs, reward, done, info = env_self.step(action)
                return obs.cpu().numpy(), float(reward), done, False, info

        return _GymChannelFlow()


class OppositionControl:
    """Choi, Moin & Kim (1994) opposition control.

    Sets the wall-normal wall velocity to the negative of the wall-normal
    velocity at a detection plane: v_wall(x, z) = -v(x, y_d, z). With a
    detection plane near y+ ~ 10-15 this reliably gives ~20-25% drag reduction
    at Re_tau = 180, which makes it a self-validating check on the control API.
    """

    def __init__(self, env, detection_z_plus=15.0, gain=1.0):
        self.env = env
        self.detection_z_plus = float(detection_z_plus)
        self.gain = float(gain)

    def __call__(self, observation=None):
        w_d = self.env.velocity_at_z_plus(self.detection_z_plus, component='w')
        action = -self.gain * w_d
        cs = self.env.control_shape
        if tuple(action.shape) != cs:
            # Average down to the control resolution.
            rx = action.shape[0] // cs[0]
            ry = action.shape[1] // cs[1]
            action = action.reshape(cs[0], rx, cs[1], ry).mean(dim=(1, 3))
        return action
