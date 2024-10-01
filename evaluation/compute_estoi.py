import numpy as np
from pystoi import stoi
from itertools import permutations

def compute_estoi(estimate, reference, mixture, samplerate=16000, permutation_search=False, noisy=False, extended=True):

    assert estimate.shape == reference.shape
    assert estimate.shape[0] == mixture.shape[0]
    n_srcs = estimate.shape[1]

    if permutation_search:
        pair_wise_estoi = []
        for ref_idx in range(n_srcs):
            ref_estoi = []
            for est_idx in range(n_srcs):
                ref_estoi.append({'eSTOI': stoi(reference[:,ref_idx], estimate[:,est_idx], samplerate, extended=extended)})
            pair_wise_estoi.append(ref_estoi)

        estoi_permutations = []
        all_scores, all_perms = [], []
        for perm in permutations(range(n_srcs)):
            all_perms.append(perm)
            outputs = []
            for ref_idx, est_idx in enumerate(perm):
                # ref_idx is always in the same order (0, 1, 2, etc...),
                # est_idx is in the permutation order defined by `perm`
                outputs.append(pair_wise_estoi[ref_idx][est_idx])

            # Save avg metrics between srcs for best perm selection later
            estoi_permutations.append(np.mean([o['eSTOI'] for o in outputs]))
            all_scores.append(outputs)

        # Select best perm based on best SDR (=> index 0 of second axis)
        best_perm = np.asarray(estoi_permutations).argmax()
        scores, perm = all_scores[best_perm], all_perms[best_perm]

    else:
        scores = [{'eSTOI': stoi(reference[:,j], estimate[:,j], samplerate, extended=extended)} for j in range(n_srcs)]
        perm = [_ for _ in range(n_srcs)]

    if noisy:
        for j in range(n_srcs):
            if len(mixture.shape) == 1:
                noisy_scores = {'eSTOI': stoi(reference[:,j], mixture, samplerate, extended=extended)}
            else:
                noisy_scores = {'eSTOI': stoi(reference[:,j], mixture[:,j], samplerate, extended=extended)}
            score_dict = scores[j]
            for k, v in noisy_scores.items():
                k = 'Noisy_' + k
                score_dict[k] = v

    return scores, perm
