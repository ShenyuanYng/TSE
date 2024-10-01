from datasets.paths.mixpaths import get_mixture_paths
from datasets import mc_ri_dataset


def dataset_creator(hparams, partition, path_list=None, is_training=None,):

    if path_list is None:
        path_list = datapath_creator(hparams, partition, is_training=is_training)

    feature_type = hparams.feature_type
    chunk_size = hparams.train_on_chunk if partition == 'tr' else None

    if is_training is None:
        is_training = partition != 'tt'
    else:
        pass

    if feature_type in ['ri',]:
        window_length = int(hparams.inwintime*hparams.samplerate)
        hop_length = int(hparams.hoptime*hparams.samplerate)
        assert window_length % hop_length == 0

        if hparams.fft_size != 0.:
            fft_size = int(hparams.fft_size*hparams.samplerate)
        else:
            fft_size = None

        dataset = mc_ri_dataset.MCRIDataset(
                path_list,
                n_srcs=hparams.num_sources,
                sr=hparams.samplerate, chunk_size=chunk_size,
                rms_norm=hparams.rms_norm,
                n_imics=hparams.num_imics, n_omics=hparams.num_omics,
                max_num_mics=hparams.max_num_mics, max_num_omics=hparams.max_num_omics,
                training=is_training,
                window_length=window_length, hop_length=hop_length, fft_size=fft_size, window_type=hparams.window_type,
                masking_or_mapping=hparams.masking_or_mapping,
                input_feature=feature_type,
                partition=partition,
                )
    else:
        raise

    return dataset


def datapath_creator(hparams, partition, is_training=None):

    sr_str_ = 'wav'+str(int(hparams.samplerate/1000))+'k'

    if hparams.dataset in ['fixarr_h1+n-s1',]:
        path_list = get_mixture_paths(
                hparams.data_path,
                partition=partition, sr=hparams.samplerate,
                mix_dir='mix', src_dir_list=['s1',], reverb_src_dir_list=[],
                sr_str_=sr_str_,
                is_training=is_training,
                )

    elif hparams.dataset in ['wsj0-mix',]:
        min_or_max_or_none = "min"
        path_list = get_mixture_paths(
                hparams.data_path,
                partition=partition, sr=hparams.samplerate,
                mix_dir='mix', src_dir_list=['s1','s2',], reverb_src_dir_list=[],
                sr_str_=sr_str_,
                min_or_max_or_none=min_or_max_or_none,        
                is_training=is_training,
                )

    elif hparams.dataset in ['VoicebankDemandEXP',]:
        path_list = get_mixture_paths(
                hparams.data_path,
                partition=partition, sr=hparams.samplerate,
                mix_dir='mix', src_dir_list=['s1'], reverb_src_dir_list=[],
                sr_str_=sr_str_,
                is_training=is_training,
                )
    else:
        raise

    return path_list
