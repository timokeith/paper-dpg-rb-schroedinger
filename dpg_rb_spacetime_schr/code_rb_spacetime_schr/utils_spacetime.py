"""
Helper functions: differential operators, error evaluation, output formatting and plots.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ngsolve as ng
from ngsolve import grad

# =============================================================================
# Differential operators
# =============================================================================
def d_t(func):
    """Temporal derivative (y-component gradient)"""
    return grad(func)[1]

def d_x(func):
    """Spatial derivative (x-component gradient)"""
    return grad(func)[0]

def d_xx(func):
    """Second spatial derivative (Hessian[0,0])"""
    return func.Operator("hesse")[0,0]

def Adj_A(func, V=0):
    """
    Adjoint operator: A* v = i dt v - dxx v + V v
    """
    return 1j * d_t(func) - d_xx(func) + V * func

# =============================================================================
# ROM and FOM solve for one parameter
# =============================================================================
def solve_and_evaluate_sample(mu, fom, rom, reductor):
    """Executes ROM and FOM solves and computes all error estimators for a given parameter.
    Both times are wall-clock times (perf_counter, time.time is too coarse on Windows)."""
    # ROM Solve
    t0 = time.perf_counter()
    sol_rom = rom.solve(mu)
    t_rom = time.perf_counter() - t0

    A_N = rom.operator.assemble(mu).matrix
    cond_N = np.linalg.cond(A_N) if A_N.size > 0 else 0.0

    sol_rec = reductor.reconstruct(sol_rom)

    # ROM Error Estimation
    rom_res = reductor.error_estimator.estimate_error(sol_rom, mu, rom)[0]

    # FOM Solve (the time excludes the residual norm evaluation, see DPGModel._compute)
    sol_fom, fom_res_arr = fom.solve(mu=mu, return_error_estimate=True)
    fom_res = fom_res_arr[0]
    t_fom = fom._last_solve_time

    return sol_rom, sol_rec, sol_fom, t_rom, t_fom, rom_res, fom_res, cond_N

# =============================================================================
# Extract NGSolve GridFunction from pyMOR vector
# =============================================================================
def pymor_to_gf(pymor_vec, space):
    """Convert a natively complex pyMOR vector to an NGSolve GridFunction."""
    gf = ng.GridFunction(space)
    gf.vec.data = pymor_vec.impl.vec
    return gf

# =============================================================================
# Analysis and Validation
# =============================================================================
def mu_to_dict(mu, ranges):
    """Plain dictionary of a pyMOR parameter value, for JSON output."""
    return {name: float(mu[name][0]) for name in ranges}

def summary_statistics(test, keys):
    """Minimum, mean, median and maximum of every per-sample quantity of the test set (table values of the paper)."""
    summary = {}
    for key in keys:
        values = np.array(test[key])
        if key in ('speedup', 'cond_N'):
            values = values[values > 0]  # invalid entries are stored as 0
        summary[key] = {'min': float(np.min(values)), 'mean': float(np.mean(values)),
                        'median': float(np.median(values)), 'max': float(np.max(values))}
    return summary

def compute_3_norms(gf, mesh):
    """L2 norm of u and the edge norms of q+ and q' (sums over the element boundaries, interior edges are counted twice)."""
    u_comp = gf.components[0]
    norm_u_L2 = np.sqrt(max(ng.Integrate(u_comp * ng.Conj(u_comp), mesh, order=20).real, 0.0))

    n = ng.specialcf.normal(mesh.dim)
    is_vert = ng.IfPos(n[0]**2 - n[1]**2, 1.0, 0.0)

    # Skeleton trace norm
    skel_cf = gf.components[1] * ng.Conj(gf.components[1])
    norm_skel = np.sqrt(abs(ng.Integrate(skel_cf * ng.dx(element_boundary=True, bonus_intorder=20), mesh)))

    # Flux norm on vertical edges
    flux_cf = is_vert * gf.components[2] * ng.Conj(gf.components[2])
    norm_flux = np.sqrt(abs(ng.Integrate(flux_cf * ng.dx(element_boundary=True, bonus_intorder=20), mesh)))

    return norm_u_L2, norm_skel, norm_flux

def analyze_result(gf_sol_fom, gf_error, t_rom, t_fom, rom_res, fom_res, mesh):
    """
    Speedup, residual ratios and ROM-FOM errors of one test parameter, absolute and relative (in percent) in the
    L2 norms of the three solution components.
    """
    norm_u_L2, norm_skel, norm_flux = compute_3_norms(gf_sol_fom, mesh)
    abs_error_u_L2, abs_error_skel, abs_error_flux = compute_3_norms(gf_error, mesh)

    rel_error_u_L2 = (abs_error_u_L2 / norm_u_L2 * 100) if norm_u_L2 > 0 else 0.0
    rel_error_skel = (abs_error_skel / norm_skel * 100) if norm_skel > 0 else 0.0
    rel_error_flux = (abs_error_flux / norm_flux * 100) if norm_flux > 0 else 0.0

    rel_res = (rom_res / norm_u_L2) * 100
    error_estim_ratio = rom_res / abs_error_u_L2 if abs_error_u_L2 > 0 else 0.0

    res_ratio = rom_res / fom_res if fom_res > 0 else 0.0

    speedup = t_fom / t_rom if t_rom > 0 else 0.0

    return speedup, rel_res, res_ratio, abs_error_u_L2, rel_error_u_L2, \
        abs_error_skel, rel_error_skel, abs_error_flux, rel_error_flux, error_estim_ratio

SUMMARY_LABELS = [
    ('t_rom', 'ROM solve time [s]', '.4e'), ('t_fom', 'FOM solve time [s]', '.4e'), ('speedup', 'Speedup', '.1f'),
    ('abs_error_u_L2', 'Absolute L2-error', '.4e'), ('abs_error_skel', 'Absolute qplus L2-error', '.4e'),
    ('abs_error_flux', 'Absolute qprime L2-error', '.4e'), ('rel_error_u_L2', 'Relative L2-error [%]', '.4f'),
    ('rel_error_skel', 'Relative qplus L2-error [%]', '.4f'), ('rel_error_flux', 'Relative qprime L2-error [%]', '.4f'),
    ('rom_res', 'ROM residual', '.4e'), ('rel_res', 'Relative ROM residual [%]', '.4f'), ('fom_res', 'FOM residual', '.4e'),
    ('res_ratio', 'ROM/FOM residual ratio', '.4f'), ('error_estim_ratio', 'ROM resid. / ROM-FOM L2-error', '.4f'),
    ('cond_N', 'ROM matrix condition number', '.2e'),
]

def print_summary(summary, rom_size, n_test):
    """
    Prints the statistical summary of the validation run (table values of the paper).
    """
    print("\n" + "="*60)
    print(f"ROM Size: {rom_size}")
    print("="*60)
    print(f"      FINAL STATISTICS FOR {n_test} RANDOM TEST SAMPLES      ")
    print("="*60)
    for key, label, fmt in SUMMARY_LABELS:
        for stat in ['median', 'mean', 'min', 'max']:
            print(f"  {stat.capitalize():<7}{label:<38}{summary[key][stat]:{fmt}}")
        print("-"*60)


# =============================================================================
# True error computation for convergence analysis
# =============================================================================
def compute_greedy_true_errors(max_err_mus, fom, reductor, rom, X, mesh):
    """
    ROM-FOM errors, FOM residuals and condition numbers along the greedy path: entry i belongs to the basis of
    size i and the parameter max_err_mus[i]. The ROM solutions are computed from the leading blocks of the final ROM.
    """

    full_basis = reductor.bases['RB']
    n_mus = len(max_err_mus)

    u_L2_errors, skel_errors, flux_errors = [], [], []
    fom_residuals = []
    cond_nrs = []

    print(f"\n--- Computing True Errors at {n_mus} Greedy-Selected Parameters ---")

    for basis_size, mu in enumerate(max_err_mus):
        # Solve FOM
        sol_fom, fom_resid = fom.solve(mu, return_error_estimate = True)
        fom_residuals.append(fom_resid[0])

        # Reconstruct ROM solution using truncated basis
        if basis_size == 0:
            rom_sol_rec_vec = fom.solution_space.zeros()
        else:
            A_N = rom.operator.assemble(mu).matrix[:basis_size, :basis_size]
            cond_N = np.linalg.cond(A_N)
            cond_nrs.append(cond_N)
            f_N = rom.rhs.as_range_array(mu).to_numpy().flatten()[:basis_size]
            u_i = np.linalg.solve(A_N, f_N)

            rom_sol_rec_vec = full_basis[:basis_size].lincomb(u_i)

        gf_rom = pymor_to_gf(rom_sol_rec_vec.vectors[0], X)

        # Compute Error Field
        gf_fom = pymor_to_gf(sol_fom.vectors[0], X)

        # Create explicit difference GridFunction
        gf_err = ng.GridFunction(X)
        gf_err.vec.data = gf_fom.vec - gf_rom.vec
        err_u_L2, err_skel, err_flux = compute_3_norms(gf_err, mesh)

        u_L2_errors.append(err_u_L2)
        skel_errors.append(err_skel)
        flux_errors.append(err_flux)

    return u_L2_errors, skel_errors, flux_errors, fom_residuals, cond_nrs

# =============================================================================
# Plots of the greedy convergence and the condition numbers from the results dictionary
# =============================================================================
def plot_case_results(results, results_dir, stem):
    """Draw all plots of one case from its results dictionary (as saved in the JSON file)."""
    greedy = results['greedy']
    plot_greedy_convergence(greedy, os.path.join(results_dir, f"Greedy_convergence_{stem}.pdf"))
    if 'cond_nrs_from_N1' in greedy:
        plot_condition_numbers(greedy, os.path.join(results_dir, f"Conditioning_{stem}.pdf"))

# The figures are drawn at their printed size, i.e. the width of a subfigure with 0.48\textwidth in the SIAM class
# (text width 5.125in), so that line widths and font sizes in the paper are exactly the ones set here.
# Same style block as the FOM convergence scripts: 1.0 pt lines, viridis colours with distinct brightness for
# grayscale printing, hollow markers for dashed curves, legend above the axes, no bbox_inches='tight'.
FIG_SIZE = (0.48 * 5.125, 2.0)
PAPER_RC = {
    'font.size': 7,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
}

def plot_greedy_convergence(greedy, save_path):
    """ROM residual, ROM-FOM errors and FOM residual along the greedy path, against the basis size N >= 1."""
    max_errs = greedy['max_errs']
    n_last = len(max_errs)
    if greedy['stop_reason'] == 'extension_failed':
        n_last -= 1  # the last selected snapshot was linearly dependent, its ROM-FOM error is not meaningful
    N = range(1, n_last)
    every = max(1, (n_last - 1) // 20)  # about 20 markers per curve

    plt.rcParams.update(PAPER_RC)
    plt.figure(figsize=FIG_SIZE, constrained_layout=True)

    # ROM residual (solid, filled markers) and FOM residual (dashed, hollow markers), both black
    plt.semilogy(N, max_errs[1:n_last], '-o', color='black', lw=1.0, markersize=3.5, markevery=every,
                 zorder=5, label=r"ROM residual")
    if 'u_L2_errs' in greedy:
        plt.semilogy(N, greedy['fom_res'][1:n_last], '--o', color='black', lw=1.0, markersize=3, mfc='white',
                     markevery=every, zorder=4, label=r"FOM residual")
        # ROM-FOM errors in the three viridis colours of the FOM plots with distinct markers
        plt.semilogy(N, greedy['u_L2_errs'][1:n_last], '-^', color='#31688e', lw=1.0, markersize=3.5,
                     markevery=every, zorder=3, label=r"$u$-error")
        plt.semilogy(N, greedy['skel_errs'][1:n_last], '-s', color='#35b779', lw=1.0, markersize=3.5,
                     markevery=every, zorder=3, label=r"$q^+$-error")
        plt.semilogy(N, greedy['flux_errs'][1:n_last], '-D', color='#90d743', lw=1.0, markersize=3,
                     markevery=every, zorder=3, label=r"$q'$-error")

    plt.xlabel(r"Reduced basis size $N$")
    plt.ylabel(r"Residuals / $L^2$-errors")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
    plt.grid(True, which="major", linestyle='-', lw=0.4, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved greedy convergence plot to: {os.path.relpath(save_path, os.path.dirname(os.path.abspath(__file__)))}")

def plot_condition_numbers(greedy, save_path):
    """Condition numbers of the reduced normal equations against the basis size N >= 1."""
    cond_nrs = greedy['cond_nrs_from_N1']
    every = max(1, len(cond_nrs) // 20)

    plt.rcParams.update(PAPER_RC)
    plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    plt.semilogy(range(1, len(cond_nrs) + 1), cond_nrs, '-o', color='black', lw=1.0, markersize=3.5, markevery=every)
    plt.xlabel(r"Reduced basis size $N$")
    plt.ylabel(r"Condition number")
    plt.grid(True, which="major", linestyle='-', lw=0.4, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved conditioning plot to: {os.path.relpath(save_path, os.path.dirname(os.path.abspath(__file__)))}")
