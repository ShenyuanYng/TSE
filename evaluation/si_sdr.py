import numpy as np
from itertools import permutations


def compute_measures(estimated_signal, reference_signals, j, scaling=True):
    Rss = np.dot(reference_signals.transpose(), reference_signals)
    this_s = reference_signals[:, j]
    if np.any(np.diag(Rss) <= 1e-16):
        return {'SDR': np.nan}

    if scaling:
        a = np.dot(this_s, estimated_signal) / Rss[j, j]
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true ** 2).sum()
    Snn = (e_res ** 2).sum()

    SDR = 10 * np.log10(Sss / Snn)

    return {'SDR': SDR}


def _sdr_permutation_search(estimate, reference, mixture, scaling):
    n_srcs = estimate.shape[1]

    pair_wise_sdr = []
    for ref_idx in range(n_srcs):
        ref_sdr = []
        for est_idx in range(n_srcs):
            ref_sdr.append(_do_si_sdr(estimate[:, est_idx], reference[:, ref_idx], mixture, scaling))
        pair_wise_sdr.append(ref_sdr)

    sdr_permutations = []
    all_scores, all_perms = [], []
    for perm in permutations(range(n_srcs)):
        all_perms.append(perm)
        outputs = []
        for ref_idx, est_idx in enumerate(perm):
            outputs.append(pair_wise_sdr[ref_idx][est_idx])

        sdr_permutations.append(np.mean([o['SDR'] for o in outputs]))
        all_scores.append(outputs)

    best_perm = np.asarray(sdr_permutations).argmax()
    return all_scores[best_perm], all_perms[best_perm]


def mean_center(sig, ax):
    return sig - sig.mean(axis=ax)


def _do_si_sdr(estimate, reference, mixture, scaling):
    noise_ref = mixture - reference
    ref = np.stack([reference, noise_ref]).T

    est = mean_center(estimate, 0)
    ref = mean_center(ref, 0)

    return compute_measures(est, ref, 0, scaling=scaling)


def si_sdr(estimate, reference, mixture, permutation_search=False, scaling=True, noisy=False):
    n_srcs = estimate.shape[1]
    if permutation_search:
        scores, perm = _sdr_permutation_search(estimate, reference, mixture, scaling)
    else:
        if len(mixture.shape) == 1:
            scores = [_do_si_sdr(estimate[:, j], reference[:, j], mixture, scaling) for j in range(n_srcs)]
        else:
            scores = [_do_si_sdr(estimate[:, j], reference[:, j], mixture[:, j], scaling) for j in range(n_srcs)]
        perm = [_ for _ in range(n_srcs)]

    if noisy:
        for j in range(n_srcs):
            if len(mixture.shape) == 1:
                noisy_scores = _do_si_sdr(mixture, reference[:, j], mixture, scaling)
            else:
                noisy_scores = _do_si_sdr(mixture[:, j], reference[:, j], mixture[:, j], scaling)
            score_dict = scores[j]
            for k, v in noisy_scores.items():
                k = 'Noisy_' + k
                score_dict[k] = v

    return scores, perm
