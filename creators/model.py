from nets import tfgridnet


def model_creator(hparams):
    n_freq_bins = int(hparams.samplerate*hparams.inwintime)//2+1
    rmax = hparams.target_trunc_max
    n_pred_components = 2

    model_name = hparams.model_name

    if not hasattr(hparams, 'mag_sf'):
        hparams.mag_sf = 1.0

    if model_name == 'DPRNNTF':
        class_name = tfgridnet.DPRNNTF
        model = class_name(
            hparams.num_sources, n_freq_bins,
            hparams.num_imics, hparams.num_omics,
            hparams.seqmodel_type, hparams.num_layers, hparams.num_hid_units, hparams.emb_dim, hparams.emb_ks, hparams.emb_hs,
            hparams.masking_or_mapping, hparams.mask_nonlinearity, rmax,
            act_type=hparams.act_type,
            norm_type=hparams.norm_type,
            convnormact_type=hparams.convnormact_type,
            feature_type=hparams.feature_type,
            n_pred_components=n_pred_components,
            first_conv_use_act=hparams.first_conv_use_act,
            mag_sf=hparams.mag_sf,
            )

    else:
        raise

    return model
