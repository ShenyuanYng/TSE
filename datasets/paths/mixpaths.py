import os
from loguru import logger
import soundfile as sf
from collections import OrderedDict


def get_mixture_paths(
        data_path,
        partition='tt', sr=16000,
        mix_dir=None, src_dir_list=[], reverb_src_dir_list=[],
        sr_str_="",
        min_or_max_or_none="",
        is_training=None,
        ):

    root_path = os.path.join(data_path, sr_str_, min_or_max_or_none, partition)
    filelist = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir)) if f.endswith('.wav')])
    logger.info('{}, len(filelist)={}'.format(partition, len(filelist)))

    if partition == 'cv':
        # cv sorted from long to short
        lens = [sf.info(os.path.join(root_path,mix_dir,filepath)).frames for filepath in filelist]
        filelist = [filepath for _, filepath in sorted(zip(lens, filelist), reverse=True)]
        logger.info('{}, sorted from long to short'.format(partition))

    return get_path_list(filelist, root_path,
            mix_dir, src_dir_list, reverb_src_dir_list,
            )


def get_path_list(
        filelist, root_dir,
        mix_dir, src_dir_list, reverb_src_dir_list,
        ):
    path_list = [get_path_dict(
            f, root_dir,
            mix_dir, src_dir_list, reverb_src_dir_list,
            ) for f in filelist]
    return path_list


def get_path_dict(
        filename, root_dir,
        mix_dir, src_dir_list, reverb_src_dir_list,
        ):
    id_ = os.path.splitext(filename)[0]
    path_dict = {
        'id': id_,
        'srcs': OrderedDict({s: os.path.join(root_dir, s, filename) for s in src_dir_list}),
        'mix': os.path.join(root_dir, mix_dir, filename),
    }
    return path_dict
