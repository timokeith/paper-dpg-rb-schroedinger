"""
Main script: greedy RB construction and validation for the ultraweak space-time DPG formulation of the Schrödinger
equation with parametrized potential, Cases 1 and 2 of the paper.

Uses the hybrid approach of computing FOM solutions by static condensation, while reducing the coercive normal
equations by standard Galerkin projection.

All greedy and test set results of a case are saved to a JSON file in sinum_paper_results_rb_spacetime_schr
(after the greedy phase and again after the test loop), from which the plots can be redrawn with plot_only = True.
No command line arguments, the settings are set below. Run from the shell with the output redirected to a log file.
"""
import os
import json
import subprocess
from datetime import datetime
import ngsolve as ng

from pymor.core.defaults import set_defaults
from pymor.algorithms.greedy import rb_greedy

from scenario_spacetime import setup_scenario
from rom_spacetime import DPGProjectedReductor
from utils_spacetime import (analyze_result, compute_greedy_true_errors, mu_to_dict, plot_case_results, print_summary,
                             pymor_to_gf, solve_and_evaluate_sample, summary_statistics)

# ------------------------------------------------------------------------------
# Settings of the paper
# ------------------------------------------------------------------------------
cases = [1, 2]           # Cases 1 and 2 of the paper (Case 2 has the random sets of the paper only if run after Case 1)
p, dp = 3, 2             # trial space degree (full flux degree for Q_prime) and test space enrichment
N_mesh = 90              # uniform N_mesh x N_mesh mesh, h = 1 / N_mesh
n_training = 10000       # size S of the random training set
max_ext = 100            # maximal number of greedy steps
atol = 1e-4              # greedy tolerance for the ROM residual
gs_rtol = 1e-11          # Gram-Schmidt tolerance (relaxed from 1e-13 to ignore solver noise of about 1e-12)
n_test = 200             # size of the random test set
compute_true_errs = True # ROM-FOM errors and FOM residuals along the greedy path (one FOM solve per greedy step)
plot_only = False        # turn on to redraw the plots from the saved JSON results without recomputing

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'sinum_paper_results_rb_spacetime_schr')
os.makedirs(results_dir, exist_ok=True)


def file_stem(case):
    return f"rb_spacetime_case_{case}_N{N_mesh}_p{p}_dp{dp}_maxExt{max_ext}"


def save_results(results, path):
    with open(path, 'w') as file:
        json.dump(results, file, indent=1)
    print(f"Saved results to: {os.path.relpath(path, script_dir)}")


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=script_dir,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


if __name__ == "__main__":
    print("=" * 60)
    print("   Ultraweak space-time DPG with RB for the Schrödinger equation with parametrized potential   ")
    print("=" * 60)

    for case in cases:
        results_file = os.path.join(results_dir, f"results_{file_stem(case)}.json")

        if plot_only:
            with open(results_file) as file:
                results = json.load(file)
            plot_case_results(results, results_dir, file_stem(case))
            continue

        print("\n" + "#" * 60)
        print(f"CASE {case}")
        print("#" * 60)
        results = {
            'meta': {
                'case': case, 'date': datetime.now().isoformat(timespec='seconds'), 'commit': git_commit(),
                'p': p, 'dp': dp, 'N_mesh': N_mesh, 'h': 1 / N_mesh, 'n_training': n_training, 'max_ext': max_ext,
                'atol': atol, 'gs_rtol': gs_rtol, 'n_test': n_test,
            }
        }

        # =========================================================================
        # 1. SETUP
        # =========================================================================
        fom, X, mesh, ranges = setup_scenario(p, dp, N_mesh, active_scenario=case)
        param_space = fom.parameters.space(ranges)
        results['meta']['ranges'] = ranges
        results['meta']['trial_dofs'] = X.ndof

        reductor = DPGProjectedReductor(fom, product=fom.products['L2'])  # Gram-Schmidt in the scaled L2 product of X
        print(f"\n--- Sampling Greedy Training Set ({n_training} samples) ---")
        training_set = param_space.sample_randomly(n_training)

        # =========================================================================
        # 2. GREEDY SEARCH
        # =========================================================================
        set_defaults({'pymor.algorithms.gram_schmidt.gram_schmidt.rtol': gs_rtol})

        with ng.TaskManager():
            greedy_data = rb_greedy(
                fom, reductor, training_set,
                max_extensions=max_ext,
                atol=atol,
                use_error_estimator=True,
                extension_params={'method': 'gram_schmidt'}
            )

        rom = greedy_data['rom']
        extensions = greedy_data['extensions']
        max_errs = [float(err) for err in greedy_data['max_errs']]
        if extensions >= max_ext:
            stop_reason = 'max_extensions'
        elif max_errs[-1] <= atol:
            stop_reason = 'tolerance'
        else:
            stop_reason = 'extension_failed'
        print(f"Final ROM Size: {rom.solution_space.dim} (stop reason: {stop_reason})")

        # max_errs[i] is the maximal ROM residual over the training set for the basis of size i, attained at
        # max_err_mus[i], which became snapshot i + 1 (except for the last entry if the greedy stopped by the
        # tolerance or by a failed extension)
        results['greedy'] = {
            'final_rom_size': rom.solution_space.dim,
            'extensions': extensions,
            'stop_reason': stop_reason,
            'greedy_time': greedy_data['time'],
            'max_errs': max_errs,
            'max_err_mus': [mu_to_dict(mu, ranges) for mu in greedy_data['max_err_mus']],
        }

        if compute_true_errs:
            # ROM-FOM errors with the basis of size i at max_err_mus[i], FOM residuals there, and the condition
            # numbers of the reduced normal equations for the basis sizes 1, ..., len(max_err_mus) - 1
            u_L2_errs, skel_errs, flux_errs, fom_res, cond_nrs = \
                compute_greedy_true_errors(greedy_data['max_err_mus'], fom, reductor, rom, X, mesh)
            results['greedy'].update({
                'u_L2_errs': u_L2_errs, 'skel_errs': skel_errs, 'flux_errs': flux_errs,
                'fom_res': fom_res, 'cond_nrs_from_N1': cond_nrs,
            })
        save_results(results, results_file)

        # =========================================================================
        # 3. VALIDATION ON A RANDOM TEST SET
        # =========================================================================
        print("\n" + "=" * 60)
        print(f"      RUNNING VALIDATION ON {n_test} RANDOM SAMPLES      ")
        print("=" * 60)

        test_set = param_space.sample_randomly(n_test)
        test_keys = ['t_rom', 't_fom', 'speedup', 'rom_res', 'rel_res', 'fom_res', 'res_ratio',
                     'abs_error_u_L2', 'rel_error_u_L2', 'abs_error_skel', 'rel_error_skel',
                     'abs_error_flux', 'rel_error_flux', 'error_estim_ratio', 'cond_N']
        test = {key: [] for key in test_keys}
        test['mus'] = []

        for mu in test_set:
            sol_rom, sol_rec, sol_fom, t_rom, t_fom, rom_res, fom_res, cond_N = \
                solve_and_evaluate_sample(mu, fom, rom, reductor)

            gf_sol_rom = pymor_to_gf(sol_rec.vectors[0], X)
            gf_sol_fom = pymor_to_gf(sol_fom.vectors[0], X)
            gf_error = ng.GridFunction(X)
            gf_error.vec.data = gf_sol_fom.vec - gf_sol_rom.vec

            (speedup, rel_res, res_ratio, abs_error_u_L2, rel_error_u_L2,
             abs_error_skel, rel_error_skel, abs_error_flux, rel_error_flux, error_estim_ratio) \
                = analyze_result(gf_sol_fom, gf_error, t_rom, t_fom, rom_res, fom_res, mesh)

            sample = {'t_rom': t_rom, 't_fom': t_fom, 'speedup': speedup, 'rom_res': rom_res, 'rel_res': rel_res,
                      'fom_res': fom_res, 'res_ratio': res_ratio, 'abs_error_u_L2': abs_error_u_L2,
                      'rel_error_u_L2': rel_error_u_L2, 'abs_error_skel': abs_error_skel,
                      'rel_error_skel': rel_error_skel, 'abs_error_flux': abs_error_flux,
                      'rel_error_flux': rel_error_flux, 'error_estim_ratio': error_estim_ratio, 'cond_N': cond_N}
            test['mus'].append(mu_to_dict(mu, ranges))
            for key in test_keys:
                test[key].append(float(sample[key]))

        results['test'] = test
        results['summary'] = summary_statistics(test, test_keys)
        save_results(results, results_file)

        print_summary(results['summary'], rom.solution_space.dim, n_test)

        # =========================================================================
        # 4. PLOTS
        # =========================================================================
        plot_case_results(results, results_dir, file_stem(case))
