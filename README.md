# Essential paper

This repository contains code intended to reproduce some of the results in:

- [An essential building block for cosmological zoom-in perturbation theory](https://arxiv.org/abs/2601.19812)
- [Cosmological zoom-in perturbation theory as a consistent beyond point-particle approximation framework](https://arxiv.org/abs/2604.01112)

The scripts and notebooks are research materials and are not, by themselves, a complete or independently validated reproduction of either paper. Numerical results may depend on the parameters and assumptions set in each file.

## Contents

- `shear_collapse_lcdm.py` — compares spherical and anisotropic dust-collapse models in flat ΛCDM. Running it writes a plot, a collapse-history CSV, and the initial-conditions CSV in the repository directory.
- `Theta-Splashback-radius.py` — evaluates and plots a halo density profile, its logarithmic slope, and the corresponding splashback radius.
- `collapse_initial_conditions.csv` — initial conditions for the collapse calculations.
- `FLRW_CPT.ipynb`, `Matter_Horizon_galaxy_and_star.ipynb`, and `Matter_horizon_spaclike_geodesic.ipynb` — Jupyter notebooks.

## Running the scripts

Use Python with `numpy`, `scipy`, and `matplotlib` installed to run the collapse calculation:

```bash
python shear_collapse_lcdm.py
```

The halo-profile script additionally requires [Colossus](https://bdiemer.bitbucket.io/colossus/):

```bash
python Theta-Splashback-radius.py
```

Run the notebooks with Jupyter. The scripts use parameters defined in their source; see those files when comparing a run with a particular figure or result in the papers.
