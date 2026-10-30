#!/usr/bin/env python3
"""Compare spherical and anisotropic dust collapse in flat Lambda-CDM.

The independent variable is N = ln(a).  Expansion and shear variables are
made dimensionless with H0:

    q       = Theta_L / H0
    s_i     = sigma_i / H0
    e_i     = E_i / H0**2

The anisotropic calculation uses the local ``silent-universe`` closure
H_ab = curl(E)_ab = 0 and assumes that sigma_ab and E_ab remain diagonal in
the same Fermi-propagated eigenframe.  This is a local tidal approximation,
not the full non-local Einstein-dust system.

Outputs
-------
collapse_theta_vs_time.png
    Total expansion Theta = 3H + Theta_L and the three local axis factors.
collapse_histories.csv
    All plotted histories, convenient for further analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Force a standard font to avoid small-caps / all-caps rendering from custom fonts.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana"]


# ---------------------------------------------------------------------------
# Cosmology and numerical choices
# ---------------------------------------------------------------------------
H0_KM_S_MPC = 70.0
OMEGA_M0 = 0.30
OMEGA_L0 = 1.0 - OMEGA_M0

A_INITIAL = 0.02          # z_i = 49; safely matter dominated
A_FINAL = 2.0             # evolve slightly into the future if no collapse
DELTA_INITIAL =-0.045     # 0.045     # spherical nonlinear model collapses near the present epoch
DELTA_STOP = 1.0e4        # practical proxy for the dust singularity
AXIS_STOP = 1.0e-4        # stop if any local axis / initial axis reaches this
N_SAMPLES = 2400

# 1/H0 in Gyr.  1 Mpc = 3.085677581491367e19 km.
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
H0_SI = H0_KM_S_MPC / 3.085677581491367e19
HUBBLE_TIME_GYR = 1.0 / H0_SI / SECONDS_PER_GYR


def h(a: np.ndarray | float) -> np.ndarray | float:
    """Dimensionless Hubble rate H(a)/H0 for flat matter + Lambda."""
    return np.sqrt(OMEGA_M0 * np.asarray(a) ** -3 + OMEGA_L0)


def omega_m(a: float) -> float:
    return OMEGA_M0 * a**-3 / h(a) ** 2


def growth_rate(a: float) -> float:
    """Accurate late-time approximation to d ln D_+ / d ln a."""
    return omega_m(a) ** 0.55


def cosmic_time_gyr(a: np.ndarray | float) -> np.ndarray | float:
    """Cosmic proper time since the big bang for flat matter + Lambda."""
    a = np.asarray(a)
    tau = (
        2.0
        / (3.0 * np.sqrt(OMEGA_L0))
        * np.arcsinh(np.sqrt(OMEGA_L0 / OMEGA_M0) * a ** 1.5)
    )
    return HUBBLE_TIME_GYR * tau


def matter_source(a: float) -> float:
    """(kappa rho_bar / 2) / H0^2 = (3/2) Omega_m0 a^-3."""
    return 1.5 * OMEGA_M0 * a**-3


# ---------------------------------------------------------------------------
# Spherical models
# ---------------------------------------------------------------------------
def rhs_spherical_linear(n: float, y: np.ndarray) -> np.ndarray:
    """Linear continuity + Raychaudhuri: omit q^2 and all shear."""
    delta, q = y
    a = np.exp(n)
    hh = h(a)
    # d/dN = (1/h) d/d(H0 t)
    return np.array([-q / hh, (-2.0 * hh * q - matter_source(a) * delta) / hh])


def rhs_spherical_nonlinear(n: float, y: np.ndarray) -> np.ndarray:
    """Exact local spherical dust equations before shell crossing."""
    delta, q = y
    a = np.exp(n)
    hh = h(a)
    return np.array(
        [
            -(1.0 + delta) * q / hh,
            (-2.0 * hh * q - q * q / 3.0 - matter_source(a) * delta) / hh,
        ]
    )


# ---------------------------------------------------------------------------
# Anisotropic silent-universe model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TidalCase:
    name: str
    # Fractions of the initial deformation eigenvalues; sum must equal one.
    # Positive entries are initially compressive in the growing mode.
    p: tuple[float, float, float]
    color: str


TIDAL_CASES = (
    TidalCase("Pancake (1 compressive direction)", (1.40, -0.20, -0.20), "#7b2cbf"),
    TidalCase("Filament (2 compressive directions)", (0.75, 0.55, -0.30), "#e76f51"),
    # A nearly isotropic, but nonzero-shear, node.  Larger anisotropy generally
    # turns one direction extensive before the dust singularity in this local
    # silent-universe model (the familiar instability toward sheets/spindles).
    TidalCase(
        "Node (3 compressive directions)",
        (1.0 / 3.0 + 1.0e-3, 1.0 / 3.0, 1.0 / 3.0 - 1.0e-3),
        "#2a9d8f",
    ),
)


def anisotropic_initial_state(case: TidalCase) -> np.ndarray:
    """Growing-mode initial data based on deformation eigenvalues.

    At high redshift, q = -h f delta.  The directional peculiar expansion is
    assigned in fractions p_i, and its trace-free part supplies s_i.  During
    matter domination the growing tidal mode gives approximately
    e_i = -(3/2) h s_i.
    """
    p = np.asarray(case.p, dtype=float)
    if not np.isclose(p.sum(), 1.0):
        raise ValueError(f"The deformation fractions for {case.name!r} must sum to one")

    hh = h(A_INITIAL)
    q0 = -hh * growth_rate(A_INITIAL) * DELTA_INITIAL
    s0 = q0 * (p - 1.0 / 3.0)
    e0 = -1.5 * hh * s0

    # L_i are local physical principal-axis factors, normalized to one at a_i.
    ln_l0 = np.zeros(3)
    return np.concatenate(([DELTA_INITIAL, q0], s0, e0, ln_l0))


def rhs_anisotropic(n: float, y: np.ndarray) -> np.ndarray:
    """Nonlinear local silent-universe equations in a shared eigenframe."""
    delta, q = y[:2]
    s = y[2:5]
    e = y[5:8]
    a = np.exp(n)
    hh = h(a)

    s2_mean = np.mean(s * s)
    se_mean = np.mean(s * e)

    ddelta_dtau = -(1.0 + delta) * q
    dq_dtau = -2.0 * hh * q - q * q / 3.0 - np.dot(s, s) - matter_source(a) * delta
    ds_dtau = (
        -(2.0 / 3.0) * (3.0 * hh + q) * s
        - (s * s - s2_mean)
        - e
    )
    de_dtau = (
        -(3.0 * hh + q) * e
        + 3.0 * (s * e - se_mean)
        - matter_source(a) * (1.0 + delta) * s
    )

    # d ln(L_i)/d(H0 t) = h + q/3 + s_i.
    dln_l_dtau = hh + q / 3.0 + s
    return np.concatenate(
        ([ddelta_dtau / hh, dq_dtau / hh], ds_dtau / hh, de_dtau / hh, dln_l_dtau / hh)
    )


def collapse_event(n: float, y: np.ndarray) -> float:
    """Stop near the dust singularity rather than integrating through it."""
    return DELTA_STOP - y[0]


collapse_event.terminal = True
collapse_event.direction = -1


def axis_event(n: float, y: np.ndarray) -> float:
    """Stop if a principal axis becomes extremely small."""
    if y.size < 11:
        return 1.0
    return np.min(np.exp(np.clip(y[8:11], -700.0, 700.0))) - AXIS_STOP


axis_event.terminal = True
axis_event.direction = -1


def integrate(rhs, y0: np.ndarray, *, anisotropic: bool = False):
    n0, n1 = np.log(A_INITIAL), np.log(A_FINAL)
    events = (collapse_event, axis_event) if anisotropic else (collapse_event,)
    sol = solve_ivp(
        rhs,
        (n0, n1),
        y0,
        method="DOP853",
        rtol=2.0e-9,
        atol=1.0e-11,
        events=events,
        dense_output=True,
        max_step=0.01,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    n_eval = np.linspace(n0, sol.t[-1], N_SAMPLES)
    return n_eval, sol.sol(n_eval)


def describe_collapse(case: TidalCase, n: np.ndarray, y: np.ndarray) -> str:
    """Report the initial web type and the axes contracting at termination."""
    positive = int(np.count_nonzero(np.asarray(case.p) > 0.0))
    web_type = {1: "pancake", 2: "filament", 3: "node/three-axis"}[positive]
    axes = np.exp(np.clip(y[8:11, -1], -700.0, 700.0))
    directional_rates = h(np.exp(n[-1])) + y[1, -1] / 3.0 + y[2:5, -1]
    contracting = int(np.count_nonzero(directional_rates < 0.0))
    return (
        f"{case.name}: initial classification={web_type}; "
        f"contracting axes at stop={contracting}; "
        f"L/L_i={np.array2string(axes, precision=3)}; "
        f"a_stop={np.exp(n[-1]):.4f}, t_stop={cosmic_time_gyr(np.exp(n[-1])):.3f} Gyr"
    )


def initial_conditions_rows() -> list[dict]:
    """Complete numerical initial conditions at a = A_INITIAL for every model."""
    hh = h(A_INITIAL)
    q0 = -hh * growth_rate(A_INITIAL) * DELTA_INITIAL
    rows = [
        dict(model="spherical (linear and nonlinear)", p=None, delta=DELTA_INITIAL,
             q=q0, s=(0.0, 0.0, 0.0), e=(0.0, 0.0, 0.0)),
    ]
    for case in TIDAL_CASES:
        y0 = anisotropic_initial_state(case)
        rows.append(dict(model=case.name, p=case.p, delta=y0[0], q=y0[1],
                         s=tuple(y0[2:5]), e=tuple(y0[5:8])))
    return rows


def format_initial_conditions() -> str:
    hh = h(A_INITIAL)
    lines = [
        f"a_i = {A_INITIAL:g} (z_i = {1.0 / A_INITIAL - 1.0:.0f}),  "
        f"H_i/H0 = {hh:.4f},  f_i = {growth_rate(A_INITIAL):.4f}",
        f"delta_i = {DELTA_INITIAL:g},  q_i = Theta_L/H0 = {initial_conditions_rows()[0]['q']:.6f},  "
        f"L_i/L_i(a_i) = 1",
        "",
    ]
    for r in initial_conditions_rows()[1:]:
        lines.append(f"{r['model']}:")
        lines.append("  p_i   = (" + ", ".join(f"{v:+.4f}" for v in r["p"]) + ")")
        lines.append("  s_i/H0   = (" + ", ".join(f"{v:+.3e}" for v in r["s"]) + ")")
        lines.append("  E_i/H0^2 = (" + ", ".join(f"{v:+.3e}" for v in r["e"]) + ")")
    return "\n".join(lines)


def save_initial_conditions_csv(output: Path) -> None:
    header = ("model,p1,p2,p3,delta_i,q_i,s1_i,s2_i,s3_i,e1_i,e2_i,e3_i,"
              "a_i,z_i,Omega_m0,Omega_L0,H0_km_s_Mpc")
    lines = [header]
    for r in initial_conditions_rows():
        p = r["p"] if r["p"] is not None else (np.nan,) * 3
        vals = [*p, r["delta"], r["q"], *r["s"], *r["e"], A_INITIAL,
                1.0 / A_INITIAL - 1.0, OMEGA_M0, OMEGA_L0, H0_KM_S_MPC]
        lines.append(f'"{r["model"]}",' + ",".join(f"{v:.10g}" for v in vals))
    output.write_text("\n".join(lines) + "\n")


def plot_results(results: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 11.5,
    })

    fig, ax0 = plt.subplots(1, 1, figsize=(8.0, 5.2), constrained_layout=True)

    for case in TIDAL_CASES:
        n, y = results[case.name]
        a = np.exp(n)
        theta_gyr = (3.0 * h(a) + y[1]) / HUBBLE_TIME_GYR
        p_txt = ", ".join(f"{v:+.3f}" for v in case.p)
        ax0.plot(cosmic_time_gyr(a), theta_gyr, color=case.color, lw=2.2, zorder=4,
                 label=rf"{case.name}, $p_i=({p_txt})$")

    styles = {
        "linear spherical": dict(color="#264653", ls="--", lw=2.4),
        "nonlinear spherical": dict(color="#1d3557", ls="-", lw=4.0),
    }
    for label in ("linear spherical", "nonlinear spherical"):
        n, y = results[label]
        a = np.exp(n)
        theta_gyr = (3.0 * h(a) + y[1]) / HUBBLE_TIME_GYR
        ax0.plot(cosmic_time_gyr(a), theta_gyr, label=label.capitalize(), **styles[label], zorder=3)

    ax0.axhline(0.0, color="black", lw=1.0, ls=":", zorder=1)

    ax0.set_yscale("symlog", linthresh=0.15, linscale=0.8)
    ax0.set_xlabel(r"Cosmic proper time $t$ [Gyr]")
    ax0.set_ylabel(r"$\Theta=3H+\Theta_L$ [Gyr$^{-1}$]")
    ax0.set_xlim(0.0, 15.0)

    # Complete initial conditions (common to all models) in a boxed annotation.
    hh = h(A_INITIAL)
    q0 = -hh * growth_rate(A_INITIAL) * DELTA_INITIAL
    ic = (
        rf"Flat $\Lambda$CDM: $\Omega_{{m0}}={OMEGA_M0:.2f}$, "
        rf"$H_0={H0_KM_S_MPC:.0f}$ km s$^{{-1}}$ Mpc$^{{-1}}$" "\n"
        rf"Initial data at $a_i={A_INITIAL:g}$ ($z_i={1/A_INITIAL-1:.0f}$):" "\n"
        rf"$\delta_i={DELTA_INITIAL:g}$, $q_i=\Theta_{{L,i}}/H_0={q0:.5f}$," "\n"
        r"$s_{a,i}=q_i\,(p_a-1/3)$, $e_{a,i}=-\frac{3}{2}\,(H_i/H_0)\,s_{a,i}$"
    )
    ax0.text(0.98, 0.97, ic, transform=ax0.transAxes, fontsize=10.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.3", lw=0.8), zorder=5)

    fig.legend(*ax0.get_legend_handles_labels(), loc="outside lower center", ncols=1,
               frameon=False, handlelength=2.8)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_csv(results: dict, output: Path) -> None:
    """Save all histories on separate rows; unused tensor entries are NaN."""
    rows = []
    for label, (n, y) in results.items():
        a = np.exp(n)
        for k in range(n.size):
            row = [
                label,
                cosmic_time_gyr(a[k]),
                a[k],
                y[0, k],
                y[1, k] / HUBBLE_TIME_GYR,
                (3.0 * h(a[k]) + y[1, k]) / HUBBLE_TIME_GYR,
            ]
            if y.shape[0] >= 11:
                row.extend(y[2:5, k] / HUBBLE_TIME_GYR)
                row.extend(np.exp(np.clip(y[8:11, k], -700.0, 700.0)))
            else:
                row.extend([np.nan] * 6)
            rows.append(row)

    header = (
        "model,t_Gyr,a,delta,Theta_L_Gyr^-1,Theta_total_Gyr^-1,"
        "sigma1_Gyr^-1,sigma2_Gyr^-1,sigma3_Gyr^-1,L1_over_L1i,L2_over_L2i,L3_over_L3i"
    )
    np.savetxt(output, np.asarray(rows, dtype=object), delimiter=",", fmt="%s", header=header, comments="")


def main() -> None:
    q0 = -h(A_INITIAL) * growth_rate(A_INITIAL) * DELTA_INITIAL
    results = {
        "linear spherical": integrate(rhs_spherical_linear, np.array([DELTA_INITIAL, q0])),
        "nonlinear spherical": integrate(rhs_spherical_nonlinear, np.array([DELTA_INITIAL, q0])),
    }

    for case in TIDAL_CASES:
        results[case.name] = integrate(
            rhs_anisotropic,
            anisotropic_initial_state(case),
            anisotropic=True,
        )

    here = Path(__file__).resolve().parent
    figure_path = here / "collapse_theta_vs_time_underdense.png"
    #figure_path = here / "collapse_theta_vs_time_overdense.png"
    csv_path = here / "collapse_histories.csv"
    plot_results(results, figure_path)
    ic_path = here / "collapse_initial_conditions.csv"
    save_initial_conditions_csv(ic_path)
    save_csv(results, csv_path)

    print(f"Initial redshift: z_i = {1.0 / A_INITIAL - 1.0:.1f}")
    print(f"Initial density contrast: delta_i = {DELTA_INITIAL:g}")
    print(f"Hubble time: H0^(-1) = {HUBBLE_TIME_GYR:.4f} Gyr")
    for case in TIDAL_CASES:
        print(describe_collapse(case, *results[case.name]))
    print("\nInitial conditions:\n" + format_initial_conditions() + "\n")
    print(f"Saved {ic_path}")
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
