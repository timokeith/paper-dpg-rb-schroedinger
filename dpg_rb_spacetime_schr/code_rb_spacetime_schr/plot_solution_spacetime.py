"""
Solution plot for the final reduced-order model of one case of the paper, at the parameter of the test set with the
largest relative u-error: absolute value of the full-order solution u, and absolute value of the difference between
the full-order and the reduced solution u.

main_spacetime.py does not store the reduced basis. It is rebuilt here from the greedy-selected parameters saved in
the results JSON of the case, i.e. by one full-order solve per snapshot with the same Gram-Schmidt settings, so that
neither the greedy search nor the test set validation is repeated. As a check that the rebuilt reduced-order model is
the one of the paper, the test set quantities at the chosen parameter are recomputed and compared with the stored ones.

The sampled solutions are saved to an npz file in sinum_paper_results_rb_spacetime_schr, from which the plots can be
redrawn with plot_only = True. No command line arguments, the settings are set below (mesh, polynomial degrees and
tolerances are those of main_spacetime.py). Run from the shell with the output redirected to a log file.
"""
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ngsolve as ng

from pymor.core.defaults import set_defaults

from main_spacetime import results_dir, script_dir, file_stem, git_commit
from scenario_spacetime import setup_scenario
from rom_spacetime import DPGProjectedReductor
from utils_spacetime import FIG_SIZE, PAPER_RC, pymor_to_gf, solve_and_evaluate_sample, analyze_result

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
case = 2                 # case of the paper whose final reduced-order model is plotted
n_sub = 4                # sample points per element and direction (cell centres of a uniform subgrid)
plot_only = False        # turn on to redraw the plots from the saved npz file without recomputing

# Centres of the Gaussian terms of the potentials (see scenario_spacetime.py), marked in the solution plot
barrier_centres = {1: (0.3, 0.7), 2: (0.5,)}


def rebuild_rom(results):
    """Full-order model and final reduced-order model of a case, rebuilt from its stored greedy parameters."""
    meta, greedy = results['meta'], results['greedy']
    fom, X, mesh, _ = setup_scenario(meta['p'], meta['dp'], meta['N_mesh'], active_scenario=meta['case'])
    reductor = DPGProjectedReductor(fom, product=fom.products['L2'])
    set_defaults({'pymor.algorithms.gram_schmidt.gram_schmidt.rtol': meta['gs_rtol']})

    # max_err_mus[i] became snapshot i + 1, see main_spacetime.py
    snapshot_mus = greedy['max_err_mus'][:greedy['final_rom_size']]
    print(f"\n--- Rebuilding the reduced basis from {len(snapshot_mus)} stored snapshot parameters ---")
    with ng.TaskManager():
        for k, mu in enumerate(snapshot_mus):
            reductor.extend_basis(fom.solve(fom.parameters.parse(mu)), method='gram_schmidt')
            print(f"  Snapshot {k + 1}/{len(snapshot_mus)}", flush=True)
    rom = reductor.reduce()
    return fom, reductor, rom, X, mesh


def sample_on_subgrid(gf_u, mesh, N_mesh, n_sub):
    """Values of a broken volume function at the cell centres of a uniform n_sub x n_sub subgrid of every element,
    as an array with rows in time and columns in space. No point lies on an element boundary."""
    centres = (np.arange(N_mesh * n_sub) + 0.5) / (N_mesh * n_sub)
    xx, tt = np.meshgrid(centres, centres)
    values = gf_u(mesh(xx.ravel(), tt.ravel()))
    return np.asarray(values).reshape(xx.shape)


def plot_field(values, save_path, marks=()):
    """Image of a non-negative space-time field, drawn at the printed size and in the style of the other figures,
    with dashed vertical lines at the positions marks.
    The axes are placed by hand (lengths in pt), so that all panels have the same geometry irrespective of the width
    of the colour bar labels: a square image of 101 pt and a colour bar of 6 pt."""
    plt.rcParams.update(PAPER_RC)
    fig = plt.figure(figsize=FIG_SIZE)
    width, height = 72 * FIG_SIZE[0], 72 * FIG_SIZE[1]
    ax = fig.add_axes([30 / width, 28 / height, 101 / width, 101 / height])
    cax = fig.add_axes([137 / width, 28 / height, 6 / width, 101 / height])

    image = ax.imshow(values, origin='lower', extent=(0, 1, 0, 1), aspect='auto', cmap='viridis', vmin=0,
                      interpolation='nearest')
    for x_mark in marks:
        ax.axvline(x_mark, color='white', lw=1.0, linestyle=(0, (2, 2)))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    colorbar = fig.colorbar(image, cax=cax, format=formatter)
    colorbar.outline.set_linewidth(1.0)
    colorbar.ax.tick_params(width=1.0)

    fig.savefig(save_path, dpi=600)
    plt.close(fig)
    print(f"Saved solution plot to: {os.path.relpath(save_path, script_dir)}")


if __name__ == "__main__":
    stem = file_stem(case)
    data_file = os.path.join(results_dir, f"solution_{stem}.npz")

    if not plot_only:
        print("=" * 60)
        print(f"   Solution plot for the reduced-order model of Case {case}   ")
        print("=" * 60)
        with open(os.path.join(results_dir, f"results_{stem}.json")) as file:
            results = json.load(file)
        test = results['test']
        index = int(np.argmax(test['rel_error_u_L2']))  # test parameter with the largest relative u-error

        # =========================================================================
        # 1. REBUILD THE REDUCED-ORDER MODEL
        # =========================================================================
        fom, reductor, rom, X, mesh = rebuild_rom(results)

        # =========================================================================
        # 2. SOLVE AT THE TEST PARAMETER (same evaluation as for the test set in main_spacetime.py)
        # =========================================================================
        mu = fom.parameters.parse(test['mus'][index])
        sol_rom, sol_rec, sol_fom, t_rom, t_fom, rom_res, fom_res, cond_N = \
            solve_and_evaluate_sample(mu, fom, rom, reductor)
        gf_sol_rom = pymor_to_gf(sol_rec.vectors[0], X)
        gf_sol_fom = pymor_to_gf(sol_fom.vectors[0], X)
        gf_error = ng.GridFunction(X)
        gf_error.vec.data = gf_sol_fom.vec - gf_sol_rom.vec
        rel_error_u_L2 = analyze_result(gf_sol_fom, gf_error, t_rom, t_fom, rom_res, fom_res, mesh)[4]

        # Check that the rebuilt reduced-order model is the one of the paper: the recomputed quantities have to
        # agree with the stored test set results (they agree to all printed digits, 1 % is the admitted deviation)
        print(f"\nTest parameter {index} of {len(test['mus'])}: {test['mus'][index]}")
        recomputed = {'rel_error_u_L2': float(rel_error_u_L2), 'rom_res': float(rom_res), 'fom_res': float(fom_res)}
        for key, value in recomputed.items():
            print(f"  {key:<16} stored {test[key][index]:.6e}   rebuilt {value:.6e}")
            if abs(value - test[key][index]) > 1e-2 * abs(test[key][index]):
                raise RuntimeError(f"The rebuilt reduced-order model does not reproduce the stored value of {key}")

        # =========================================================================
        # 3. SAMPLE THE VOLUME COMPONENT u AND SAVE
        # =========================================================================
        N_mesh = results['meta']['N_mesh']
        u_fom = sample_on_subgrid(gf_sol_fom.components[0], mesh, N_mesh, n_sub)
        u_error = sample_on_subgrid(gf_error.components[0], mesh, N_mesh, n_sub)

        info = {'case': case, 'test_index': index, 'mu': test['mus'][index], 'rom_size': rom.solution_space.dim,
                'date': datetime.now().isoformat(timespec='seconds'), 'commit': git_commit(),
                'results_commit': results['meta']['commit'], **recomputed}
        np.savez_compressed(data_file, u_fom=u_fom.astype(np.complex64), u_error=u_error.astype(np.complex64),
                            n_sub=n_sub, info=json.dumps(info))
        print(f"Saved sampled solutions to: {os.path.relpath(data_file, script_dir)}")

    # =========================================================================
    # 4. PLOTS (always drawn from the saved file)
    # =========================================================================
    data = np.load(data_file)
    u_fom, u_error = np.abs(data['u_fom']), np.abs(data['u_error'])

    plot_field(u_fom, os.path.join(results_dir, f"Solution_fom_{stem}.pdf"), marks=barrier_centres[case])
    plot_field(u_error, os.path.join(results_dir, f"Solution_error_{stem}.pdf"))
