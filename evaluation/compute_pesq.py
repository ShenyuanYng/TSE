import numpy as np
from pesq import pesq
from itertools import permutations

def compute_pesq(estimate, reference, mixture, samplerate=16000, wb_or_nb='nb', permutation_search=False, noisy=False):

    assert samplerate in [8000, 16000]
    if samplerate == 8000:
        assert wb_or_nb == 'nb'
    assert estimate.shape == reference.shape
    assert estimate.shape[0] == mixture.shape[0]
    n_srcs = estimate.shape[1]

    if permutation_search:
        pair_wise_pesq = []
        for ref_idx in range(n_srcs):
            ref_pesq = []
            for est_idx in range(n_srcs):
                ref_pesq.append({'PESQ': pesq(samplerate, reference[:,ref_idx], estimate[:,est_idx], wb_or_nb)})
            pair_wise_pesq.append(ref_pesq)

        pesq_permutations = []
        all_scores, all_perms = [], []
        for perm in permutations(range(n_srcs)):
            all_perms.append(perm)
            outputs = []
            for ref_idx, est_idx in enumerate(perm):
                # ref_idx is always in the same order (0, 1, 2, etc...),
                # est_idx is in the permutation order defined by `perm`
                outputs.append(pair_wise_pesq[ref_idx][est_idx])

            # Save avg metrics between srcs for best perm selection later
            pesq_permutations.append(np.mean([o['PESQ'] for o in outputs]))
            all_scores.append(outputs)

        # Select best perm based on best SDR (=> index 0 of second axis)
        best_perm = np.asarray(pesq_permutations).argmax()
        scores, perm = all_scores[best_perm], all_perms[best_perm]

    else:
        scores = [{'PESQ': pesq(samplerate, reference[:,j], estimate[:,j], wb_or_nb)} for j in range(n_srcs)]
        perm = [_ for _ in range(n_srcs)]

    if noisy:
        for j in range(n_srcs):
            if len(mixture.shape) == 1:
                noisy_scores = {'PESQ': pesq(samplerate, reference[:,j], mixture, wb_or_nb)}
            else:
                noisy_scores = {'PESQ': pesq(samplerate, reference[:,j], mixture[:,j], wb_or_nb)}
            score_dict = scores[j]
            for k, v in noisy_scores.items():
                k = 'Noisy_' + k
                score_dict[k] = v

    return scores, perm
