# Model order reduction for space-time DPG approximations of the Schrödinger equation

Code and results of the numerical experiments (Section 4) of the paper

> T. Keith and M. Ohlberger, *Model order reduction for space-time DPG approximations of the Schrödinger equation*.

The discretization uses [NGSolve](https://ngsolve.org), the reduced basis method uses [pyMOR](https://pymor.org).

## Requirements

Python 3.12 and the package versions listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The results were computed on Windows 11. The full-order solves use the PARDISO solver of MKL, which is installed together with NGSolve.

## Contents

| Script | Paper |
|---|---|
| `dpg_spacetime_schr/convergence_studies_spacetime_schr.py` | Section 4.1, Cases (a) and (c), Figures 1 and 2 |
| `dpg_rb_spacetime_schr/code_rb_spacetime_schr/main_spacetime.py` | Section 4.2, Cases 1 and 2, Figure 3, Table 1 |
| `dpg_rb_spacetime_schr/code_rb_spacetime_schr/plot_solution_spacetime.py` | Section 4.2, Figure 4 |

`main_spacetime.py` uses the modules next to it: `scenario_spacetime.py` (mesh, spaces, the two cases, operators), `fom_spacetime.py` (full-order model), `rom_spacetime.py` (reductor and error estimator), `utils_spacetime.py` (error evaluation and plots) and `complex_ngsolve_bindings.py` (complex NGSolve vectors and matrices for pyMOR).

Next to each script, a folder `sinum_paper_results_*` holds the results of the paper: the data (JSON files, an npz file for Figure 4), the plots and the logs of the runs.

## Reproducing the results

The scripts take no arguments. The settings at the top of each script are those of the paper. Run a script from its own folder, for example

```bash
python main_spacetime.py
```

- **Figures 1 and 2:** `convergence_studies_spacetime_schr.py` runs Case (a) and then Case (c). With `cases = ['a']` and `qprime_order = 'reduced'` it uses the flux degree p - 1 of Demkowicz et al., which the paper compares with.
- **Figure 3 and Table 1:** `main_spacetime.py`, which runs for several hours. Table 1 is the entry `summary` of the two JSON files. Both cases have to run in one go, because they draw their training and test parameters from the same random number generator.
- **Figure 4:** `plot_solution_spacetime.py`, after `main_spacetime.py`.

With `plot_only = True` a script only redraws its plots from the stored results.

## Notes

- The JSON files of Section 4.2 store 179 461 as `trial_dofs`. This is the dimension of the product space including the flux unknowns on horizontal edges, which are not used. The paper counts the remaining 146 701.
- Some of the stored results were computed with earlier revisions of the scripts. The scripts have only been tidied up since, without changing the computations.
- `complex_ngsolve_bindings.py` is adapted from pyMOR and keeps its BSD 2-Clause License, see `LICENSE_pymor.txt` next to it.

## License and citation

The code is distributed under the MIT License, see `LICENSE`. If you use it, please cite the paper.
