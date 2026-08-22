import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path("step_responses")
OUTPUT_DIR2 = Path("second_order_step_responses")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR2.mkdir(exist_ok=True)


# ── Generate N random first-order systems ──────────────────────────────────────
def random_first_order_systems(
    n: int = 10,
    K_range: tuple = (0.5, 5.0),
    tau_range: tuple = (0.1, 5.0),
    theta_range: tuple = (0.5, 3.0),
    dead_time_probability: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n random first-order systems.

    With probability dead_time_probability the system includes dead time θ:
        H(s) = K * exp(-θs) / (τs + 1)
    Otherwise a plain first-order system:
        H(s) = K / (τs + 1)

    Dead time is NOT handled by scipy.signal.TransferFunction (which has no
    delay support). Instead θ is stored separately and applied at simulation
    time by shifting the step-response vector.
    """
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(n):
        K   = rng.uniform(*K_range)
        tau = rng.uniform(*tau_range)

        has_delay = rng.random() < dead_time_probability
        theta     = rng.uniform(*theta_range) if has_delay else 0.0

        sys = signal.TransferFunction([K], [tau, 1])
        systems.append({
            "id":        i,
            "K":         K,
            "tau":       tau,
            "theta":     theta,
            "has_delay": has_delay,
            "sys":       sys,
            "pole":      -1 / tau,
            "bandwidth": 1 / tau,
        })
    return systems


# ── Generate N random second-order systems ────────────────────────────────────
def random_second_order_systems(
    n: int = 10,
    K_range: tuple = (0.5, 5.0),
    wn_range: tuple = (0.5, 10.0),
    zeta_range: tuple = (0.1, 1.5),
    seed: int = 42,
) -> list[dict]:
    """
    Generate n random second-order systems:

        H(s) = K * wn² / (s² + 2*ζ*wn*s + wn²)

    Parameters
    ----------
    K_range    : (min, max) DC gain
    wn_range   : (min, max) natural frequency in rad/s
    zeta_range : (min, max) damping ratio
                   ζ < 1  → underdamped  (oscillatory)
                   ζ = 1  → critically damped (fastest non-oscillatory)
                   ζ > 1  → overdamped   (sluggish, no overshoot)

    Returns a list of dicts with keys:
        id, K, wn, zeta, sys, damping_type, poles
    """
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(n):
        K    = rng.uniform(*K_range)
        wn   = rng.uniform(*wn_range)
        zeta = rng.uniform(*zeta_range)

        # H(s) = K*wn² / (s² + 2ζwn·s + wn²)
        num = [K * wn**2]
        den = [1, 2 * zeta * wn, wn**2]
        sys = signal.TransferFunction(num, den)

        # Compute poles: s = wn(-ζ ± sqrt(ζ²-1))
        discriminant = zeta**2 - 1
        if discriminant < 0:
            # Complex conjugate pair — underdamped
            wd = wn * np.sqrt(-discriminant)          # damped natural frequency
            poles = [complex(-zeta * wn,  wd),
                     complex(-zeta * wn, -wd)]
            damping_type = "underdamped"
        elif discriminant == 0:
            poles = [complex(-zeta * wn, 0)] * 2
            damping_type = "critically damped"
        else:
            # Two distinct real poles — overdamped
            sq = wn * np.sqrt(discriminant)
            poles = [complex(-zeta * wn + sq, 0),
                     complex(-zeta * wn - sq, 0)]
            damping_type = "overdamped"

        systems.append({
            "id":           i,
            "K":            K,
            "wn":           wn,
            "zeta":         zeta,
            "sys":          sys,
            "poles":        poles,
            "damping_type": damping_type,
        })
    return systems


# ── Save second-order step response plots ─────────────────────────────────────
def save_second_order_step_responses(systems: list[dict], t_end: float = 20.0) -> None:
    """
    Save one PNG per second-order system showing its step response.
    No legend labels — only axis labels are shown.
    """
    # Damping-type colour map
    color_map = {
        "underdamped":      "#378ADD",
        "critically damped":"#EF9F27",
        "overdamped":       "#D85A30",
    }

    t = np.linspace(0, t_end, 1000)

    for s in systems:
        _, y = signal.step(s["sys"], T=t)

        fig, ax = plt.subplots(figsize=(6, 4))

        color = color_map[s["damping_type"]]
        ax.plot(t, y, color=color, linewidth=1.8)

        # Steady-state value = K (DC gain)
        ax.axhline(s["K"], color="#888780", linewidth=1.0, linestyle="--")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("y(t)")

        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linewidth=0.4, alpha=0.5)

        plt.tight_layout()

        path = OUTPUT_DIR2 / f"second_order_system_{s['id']:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        print(
            f"Saved: {path}  "
            f"(K={s['K']:.3f}, ωn={s['wn']:.3f}, ζ={s['zeta']:.3f}, {s['damping_type']})"
        )


# ── Simulate step response with optional dead time ────────────────────────────
def step_with_dead_time(sys, theta: float, t: np.ndarray) -> np.ndarray:
    """
    Compute the step response of sys over t, then shift it right by theta.

    Dead time shifts the response: y_delayed(t) = y(t - theta) * u(t - theta)
    where u is the unit step. Points before theta stay at zero.
    """
    _, y = signal.step(sys, T=t)

    if theta == 0.0:
        return y

    dt = t[1] - t[0]
    shift_samples = int(round(theta / dt))

    y_delayed = np.zeros_like(y)
    if shift_samples < len(y):
        y_delayed[shift_samples:] = y[: len(y) - shift_samples]

    return y_delayed


# ── Save individual step response plots ───────────────────────────────────────
def save_step_responses(systems: list[dict], t_end: float = 20.0) -> None:
    """
    Save one PNG per system showing its step response.
    No legend labels — only axis labels are shown.
    """
    t = np.linspace(0, t_end, 1000)

    for s in systems:
        y = step_with_dead_time(s["sys"], s["theta"], t)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t, y, color="#378ADD", linewidth=1.8)

        # τ + θ marker — response rises at t = theta, settles around theta + tau
        ax.axvline(s["theta"] + s["tau"], color="#EF9F27", linewidth=1.0, linestyle="--")

        # Dead-time boundary — vertical line at t = theta (only if delay exists)
        if s["has_delay"]:
            ax.axvline(s["theta"], color="#888780", linewidth=1.0, linestyle=":")

        # Steady-state marker
        ax.axhline(s["K"], color="#D85A30", linewidth=1.0, linestyle="--")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("y(t)")

        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linewidth=0.4, alpha=0.5)

        plt.tight_layout()

        path = OUTPUT_DIR / f"system_{s['id']:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        delay_str = f"θ={s['theta']:.3f}" if s["has_delay"] else "no delay"
        print(f"Saved: {path}  (K={s['K']:.3f}, τ={s['tau']:.3f}, {delay_str})")


# ── Run ────────────────────────────────────────────────────────────────────────
systems = random_first_order_systems(n=150, K_range=(0.5, 4.0), tau_range=(0.2, 4.0))

# Print summary table
print(f"{'ID':>3} | {'K':>6} | {'τ (s)':>6} | {'θ (s)':>6} | {'Delay':>5} | {'Pole':>8} | {'BW (rad/s)':>10}")
print("-" * 58)
for s in systems:
    print(
        f"{s['id']:>3} | {s['K']:>6.3f} | {s['tau']:>6.3f} | "
        f"{s['theta']:>6.3f} | {'yes' if s['has_delay'] else 'no':>5} | "
        f"{s['pole']:>8.3f} | {s['bandwidth']:>10.3f}"
    )

print(f"\nSaving first-order step response plots to '{OUTPUT_DIR}/'...")
save_step_responses(systems)

# ── Second-order systems ───────────────────────────────────────────────────────
second_order_systems = random_second_order_systems(
    n=150, K_range=(0.5, 4.0), wn_range=(0.5, 8.0), zeta_range=(0.1, 1.5)
)

print(f"\n{'ID':>3} | {'K':>6} | {'ωn':>6} | {'ζ':>6} | {'Type':<18} | Poles")
print("-" * 72)
for s in second_order_systems:
    p = s["poles"]
    pole_str = f"{p[0].real:+.3f}{p[0].imag:+.3f}j,  {p[1].real:+.3f}{p[1].imag:+.3f}j"
    print(
        f"{s['id']:>3} | {s['K']:>6.3f} | {s['wn']:>6.3f} | {s['zeta']:>6.3f} | "
        f"{s['damping_type']:<18} | {pole_str}"
    )

print(f"\nSaving second-order step response plots to '{OUTPUT_DIR}/'...")
save_second_order_step_responses(second_order_systems)
print("Done.")