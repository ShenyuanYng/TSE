import torch
from utils.audio_utils_torch import nextpoweroftwo
from nets import tfgridnet


def model_creator(hparams):
    n_freq_bins = int(hparams.samplerate*hparams.inwintime)//2+1
    rmax = hparams.target_trunc_max
    n_pred_components = 2

    model_name = hparams.model_name

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
            )

    elif model_name == 'DPRNNTFStack':
        class_name = tfgridnet.DPRNNTFStack
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
            #
            use_pit_feat=hparams.use_pit_feat, use_lp_feat=hparams.use_lp_feat,
            n_layers_stack=hparams.num_layers_stack, n_iters=hparams.num_iters,
            init_epochs_net1=hparams.init_epochs_net1, init_epochs_net_stack=hparams.init_epochs_net_stack,
            #
            )

    else:
        raise

    return model
