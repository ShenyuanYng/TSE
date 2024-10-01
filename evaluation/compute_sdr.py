import numpy as np
from itertools import permutations
import mir_eval.separation
from mir_eval.separation import bss_eval_sources

def compute_sdr(estimate, reference, mixture, permutation_search=False, noisy=False):

    assert estimate.shape == reference.shape
    assert estimate.shape[0] == mixture.shape[0]
    n_srcs = estimate.shape[1]

    if permutation_search:
        pair_wise_sdr = []
        for ref_idx in range(n_srcs):
            ref_sdr = []
            for est_idx in range(n_srcs):
                ref_sdr.append({'SDR': bss_eval_sources(reference[:,ref_idx][None], estimate[:,est_idx][None], compute_permutation=False)[0][0]})
            pair_wise_sdr.append(ref_sdr)

        sdr_permutations = []
        all_scores, all_perms = [], []
        for perm in permutations(range(n_srcs)):
            all_perms.append(perm)
            outputs = []
            for ref_idx, est_idx in enumerate(perm):
                # ref_idx is always in the same order (0, 1, 2, etc...),
                # est_idx is in the permutation order defined by `perm`
                outputs.append(pair_wise_sdr[ref_idx][est_idx])

            # Save avg metrics between srcs for best perm selection later
            sdr_permutations.append(np.mean([o['SDR'] for o in outputs]))
            all_scores.append(outputs)

        # Select best perm based on best SDR (=> index 0 of second axis)
        best_perm = np.asarray(sdr_permutations).argmax()
        scores, perm = all_scores[best_perm], all_perms[best_perm]

    else:
        scores = [{'SDR': bss_eval_sources(reference[:,j][None], estimate[:,j][None], compute_permutation=False)[0][0]} for j in range(n_srcs)]
        perm = [_ for _ in range(n_srcs)]

    if noisy:
        for j in range(n_srcs):
            if len(mixture.shape) == 1:
                noisy_scores = {'SDR': bss_eval_sources(reference[:,j][None], mixture[None], compute_permutation=False)[0][0]}
            else:
                noisy_scores = {'SDR': bss_eval_sources(reference[:,j][None], mixture[:,j][None], compute_permutation=False)[0][0]}
            score_dict = scores[j]
            for k, v in noisy_scores.items():
                k = 'Noisy_' + k
                score_dict[k] = v

    return scores, perm
