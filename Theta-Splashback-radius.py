import numpy as np
import matplotlib.pyplot as plt

from colossus.cosmology import cosmology
from colossus.halo import profile_composite, concentration  # DK14 includes truncation/infall, so a splashback radius exists

# 1. SETUP COSMOLOGY IN COLOSSUS
cosmo = cosmology.setCosmology('planck18')
z = 0.5  # Evaluation redshift
H_z = cosmo.Hz(z) # Hubble parameter at z in km/s/Mpc
c_light = 299792.458 # km/s

mdef = '200m'  # Overdensity definition relative to mean matter density


def make_profile(M):
    """DK14 profile plus mean-density and infalling power-law outer terms,
    which are required for a steepest-slope (splashback) radius to exist."""
    c = concentration.concentration(M, mdef, z)
    return profile_composite.compositeProfile(
        'dk14', outer_names=['mean', 'pl'], M=M, c=c, z=z, mdef=mdef,
        norm=1.0, slope=1.5)


# Define halo masses to explore
masses = [1e12, 1e13, 1e14, 1e15] # Msun / h

plt.figure(figsize=(14, 10))

# Subplot 1: Theta(z, r) vs r
plt.subplot(2, 1, 1)

results_summary = []

for M in masses:
    # Initialize Einasto profile (which captures the physical outer steepening / splashback better than pure NFW)
    # Colossus automatically computes alpha (shape parameter) and concentration based on mass/redshift.
    p = make_profile(M)
    
    # Define radial range from 0.1 Mpc to 5 Mpc (in physical Mpc/h or converted)
    # Colossus internal radii are in physical/comoving kpc/h depending on convention; let's use physical Mpc.
    r_vals = np.logspace(-1, 1.2, 200) # in Mpc
    r_kpc_h = r_vals * cosmo.h * 1000.0 # convert to kpc/h for Colossus
    
    # Get logarithmic slope: d ln(rho) / d ln(r)
    dlnrho_dlnr = p.densityDerivativeLog(r_kpc_h)
    
    # Calculate Theta(z, r)
    # Theta = 3*H(z) + (c / r) * (d ln rho / d ln r)
    theta_vals = 3.0 * H_z + (c_light / r_vals) * dlnrho_dlnr
    
    # Identify zero-crossing of Theta(z, r)
    zero_crossings = np.where(np.diff(np.sign(theta_vals)))[0]
    r_zero = np.nan
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        # Linear interpolation for precise root finding
        r_zero = r_vals[idx] - theta_vals[idx] * (r_vals[idx+1] - r_vals[idx]) / (theta_vals[idx+1] - theta_vals[idx])

    # Identify splashback radius (minimum of the logarithmic slope for Einasto/outer profile)
    # Colossus has a built-in method Rsteepest() or we can find the argmin of densityDerivativeLog
    r_sp_kpc_h = p.Rsteepest()
    r_sp = r_sp_kpc_h / (cosmo.h * 1000.0) # convert back to Mpc
    
    results_summary.append((M, r_zero, r_sp))
    
    # Plot Theta
    plt.plot(r_vals, theta_vals, lw=2, label=f'M = {M:.0e} $M_\\odot$ (Zero-cross: {r_zero:.2f} Mpc)')
    if not np.isnan(r_zero):
        plt.scatter([r_zero], [0], zorder=5)

plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.xscale('log')
plt.xlabel(r'Radius $r$ [Mpc]', fontsize=12)
plt.ylabel(r'$\Theta(z, r)$ [km/s/Mpc]', fontsize=12)
plt.title(r'Evaluation of $\Theta(z,r)$ Across Different Halo Masses at $z = %.1f$' % z, fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10)

# Subplot 2: Logarithmic slope d ln rho / d ln r and Splashback Radius
plt.subplot(2, 1, 2)
for i, M in enumerate(masses):
    p = make_profile(M)
    r_vals = np.logspace(-1, 1.2, 200)
    r_kpc_h = r_vals * cosmo.h * 1000.0
    dlnrho_dlnr = p.densityDerivativeLog(r_kpc_h)
    
    r_sp = p.Rsteepest() / (cosmo.h * 1000.0)
    
    plt.plot(r_vals, dlnrho_dlnr, lw=2, label=f'M = {M:.0e} $M_\\odot$ (R_sp: {r_sp:.2f} Mpc)')
    plt.scatter([r_sp], [p.densityDerivativeLog(p.Rsteepest())], marker='X', s=80, zorder=5)

plt.axhline(-3, color='gray', linestyle=':', label='Reference slope (-3)')
plt.xscale('log')
plt.xlabel(r'Radius $r$ [Mpc]', fontsize=12)
plt.ylabel(r'$d \ln \rho / d \ln r$', fontsize=12)
plt.title(r'Logarithmic Density Slope and Splashback Radii Markers ($\times$)', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

# Print comparison table
print("\n--- COMPARISON TABLE ---")
print(f"{'Halo Mass (Msun)':<20} | {'Zero-Crossing Radius (Mpc)':<28} | {'Splashback Radius (Mpc)':<24}")
print("-" * 76)
for M, r_z, r_s in results_summary:
    z_str = f"{r_z:.2f}" if not np.isnan(r_z) else "No crossing"
    print(f"{M:<20.0e} | {z_str:<28} | {r_s:<24.2f}")