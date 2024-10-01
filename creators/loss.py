from loss_functions.general import LossFuncWrapper


def loss_creator(hparams):
    loss_items = hparams.loss_items.split('+')
    rix2_flag = 'rix2' in loss_items
    mag_flag = 'mag' in loss_items

    stft_istft_proj_flag = 0
    if 'proj' in loss_items:
        stft_istft_proj_flag = 1

    loss_type = hparams.loss_type
    if loss_type in ['ri_inference']:
        if hparams.permutation_invariant == 0:
            from loss_functions.ri_loss import riinf_enh_loss
            loss_func = riinf_enh_loss
        elif hparams.permutation_invariant == 1:
            from loss_functions.ri_loss import riinf_pit_loss
            loss_func = riinf_pit_loss
        else:
            raise
        loss_kwargs = {
            'loss_type': hparams.mask_loss_type,
            'loss_items': loss_items,
            'rix2_flag': rix2_flag, 'mag_flag': mag_flag,
            'masking_or_mapping': hparams.masking_or_mapping,
            'ref_mic': hparams.ref_mic,
            'window_type': hparams.window_type,
            'stft_istft_proj_flag': stft_istft_proj_flag,
            'permutation_invariant': hparams.permutation_invariant,
        }
    else:
        raise

    return LossFuncWrapper(loss_func, **loss_kwargs)
