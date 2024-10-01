import os
import soundfile as sf
import torch
from source_separator import SourceSeparator
from creators.dataset import dataset_creator
import torch.utils.data as data
from utils.collate import collate_seq
import glob
from loguru import logger
import numpy as np
from functools import partial
from evaluation.si_sdr import si_sdr
from evaluation.compute_pesq import compute_pesq
from evaluation.compute_estoi import compute_estoi
from evaluation.compute_sdr import compute_sdr


def print_func(all_scores, str_=None):
    if str_ is not None:
        print(str_)
    counter = {}
    for scores in all_scores:
        for score in scores:
            for k, v in score.items():
                if k not in counter:
                    counter[k] = [v, 1.]
                else:
                    a, b = counter[k]
                    counter[k] = [a+v,b+1.]
    for k, v in counter.items():
        logger.info('avg {} = {}, over {} files'.format(k, v[0]/v[1], v[1]))


def print_func_2(all_scores, str_=None):
    if str_ is not None:
        print(str_)
    assert len(all_scores) > 0
    num_sources = len(all_scores[0])
    source_counter = [{} for _ in range(num_sources)]
    for scores in all_scores:
        for ss, score in enumerate(scores):
            for k, v in score.items():
                if k not in source_counter[ss]:
                    source_counter[ss][k] = [v, 1.]
                else:
                    a, b = source_counter[ss][k]
                    source_counter[ss][k] = [a+v,b+1.]
    for ss, counter in enumerate(source_counter):
        for k, v in counter.items():
            logger.info('source {}, avg {} = {}, over {} files'.format(ss, k, v[0]/v[1], v[1]))


def main(args):

    checkpoints = glob.glob(args.checkpoint+'/version_*/_ckpt_epoch_*.ckpt')
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    hparams_yaml = os.path.dirname(checkpoint) + '/hparams.yaml'
    logger.info('loading checkpoint = {}'.format(checkpoint))
    logger.info('loading hparams_yaml = {}'.format(hparams_yaml))
    pyl_model = SourceSeparator.load_from_checkpoint(checkpoint, default_params=hparams_yaml)
    model = pyl_model.model
    hparams = pyl_model.hparams

    hparams.data_path = hparams.data_orig_path
    d = dataset_creator(hparams, args.testset, is_training=False)
    dl = data.DataLoader(
            d, args.batch_size,
            collate_fn=partial(
                collate_seq,
                pad_back_or_front=hparams.pad_back_or_front,
                ),
            shuffle=False)

    #n_imics, n_omics, ref_mic = model.n_imics, model.n_omics, 0
    n_imics, n_omics, ref_mic = model.n_imics, hparams.num_omics, 0
    assert n_imics in [1, 8]

    all_scores = [[], ]

    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    for idx_, batch in enumerate(dl):
        with torch.no_grad():
            inp, targ_list = batch
            targ = targ_list[0]
            samplerate = targ.samplerate
            id_ = targ.id
            mix_cs = targ.mix_stft # [n_imics, n_frames, n_freqs]
            mix_wfm = targ.mix_waveform # [n_imics, -1]
            n_samples = mix_wfm.shape[-1]
            targ_wfm = targ.src_waveforms # [n_srcs, 1, -1]

            if n_imics >= 1 and n_omics == 1:
                mix_ri = targ.mix_ri[None] # [1, n_frames, 1, n_imics, 2, n_freqs]
                target_dict = {
                    'mix_ri': (torch.from_numpy(mix_ri).to(device), ), # [1, n_frames, 1, n_imics, 2, n_freqs]
                    'utt_id': (id_, ),
                }
                net_outputs = model(None, target_dict)
                net_outputs = net_outputs.detach().cpu().numpy() # [1, n_frames, n_srcs, 1, 2, n_freqs]
                assert net_outputs.shape[-3] == 1
                yout = net_outputs[:,:,:,0,0,:] + 1j*net_outputs[:,:,:,0,1,:] # [1, n_frames, n_srcs, n_freqs]
                yout = yout.transpose([2,0,1,3]) # [n_srcs, 1, n_frames, n_freqs]
            else:
                raise

            n_srcs = yout.shape[0]
            yout_wfm = d._istft(yout)[:,:,:n_samples] # [n_srcs, 1, -1]

            all_est_wfm = [yout_wfm*targ.mix_std_,]
            all_targ_wfm = [targ_wfm*targ.mix_std_,]
            all_mix_wfm = [mix_wfm*targ.mix_std_,]
            all_array_str = ['ff',]

            try:
                for mm, (est_wfm, targ_wfm, mix_wfm, array_str) in enumerate(zip(all_est_wfm, all_targ_wfm, all_mix_wfm, all_array_str)):
                    if args.metric_type == 0:
                        scores, perm = si_sdr(est_wfm[:,ref_mic].T, targ_wfm[:,ref_mic].T, mix_wfm[ref_mic],
                                noisy=True, permutation_search=hparams.permutation_invariant)
                        metric_str = 'SDR'
                    elif args.metric_type == 1:
                        scores, perm = compute_pesq(est_wfm[:,ref_mic].T, targ_wfm[:,ref_mic].T, mix_wfm[ref_mic], samplerate=samplerate,
                                noisy=True, permutation_search=hparams.permutation_invariant)
                        metric_str = 'PESQ'
                    elif args.metric_type == 11:
                        scores, perm = compute_pesq(est_wfm[:,ref_mic].T, targ_wfm[:,ref_mic].T, mix_wfm[ref_mic], samplerate=samplerate,
                                noisy=True, permutation_search=hparams.permutation_invariant, wb_or_nb='wb')
                        metric_str = 'PESQ'
                    elif args.metric_type == 2:
                        scores, perm = compute_estoi(est_wfm[:,ref_mic].T, targ_wfm[:,ref_mic].T, mix_wfm[ref_mic], samplerate=samplerate,
                                noisy=True, permutation_search=hparams.permutation_invariant)
                        metric_str = 'eSTOI'
                    elif args.metric_type == 21:
                        scores, perm = compute_estoi(est_wfm[:,ref_mic].T, targ_wfm[:,ref_mic].T, mix_wfm[ref_mic], samplerate=samplerate,
                                noisy=True, permutation_search=hparams.permutation_invariant, extended=False)
                        metric_str = 'eSTOI'
                    elif args.metric_type == 100:
                        scores, perm = compute_sdr(est_wfm[:,ref_mic].T, targ_wfm[:,ref_mic].T, mix_wfm[ref_mic],
                                noisy=True, permutation_search=hparams.permutation_invariant)
                        metric_str = 'SDR'
                    else:
                        raise
                    for ss in range(n_srcs):
                        logger.info('{}, {}, {}, {}: Noisy_{} = {:.3f}, {} = {:.3f}'.format(idx_, id_, array_str, 's{}'.format(ss),
                                    metric_str, scores[ss]['Noisy_'+metric_str], metric_str, scores[ss][metric_str]))
                    all_scores[mm].append(scores)        
            except Exception:
                raise
                logger.info('skipping {}, {}, {}, {}'.format(idx_, id_, array_str, metric_str))
                pass

            #if True:
            if False:
                #
                # to listen
                #
                args.output_dir = args.checkpoint+'/processed/'
                os.system('mkdir -p '+args.output_dir)

                yout_wfm = d._istft(yout)[:,:,:n_samples] # [n_srcs, 1, -1]
                max_ = np.max(np.abs(mix_wfm.flatten()))

                sf.write(args.output_dir+'/{}.mix.wav'.format(id_),(mix_wfm.T/max_*0.9).astype(np.float32), samplerate)
                for cc in range(n_srcs):
                    #sf.write(args.output_dir+'/{}.est.{}.wav'.format(id_,cc),(yout_wfm[cc,ref_mic,:]/np.max(np.abs(yout_wfm))*0.9).astype(np.float32), samplerate)
                    sf.write(args.output_dir+'/{}.est.{}.wav'.format(id_,cc),(yout_wfm[cc,ref_mic,:]/max_*0.9).astype(np.float32), samplerate)


    for mm in range(len(all_scores)):
        logger.info(all_array_str[mm])
        print_func(all_scores[mm])
        print_func_2(all_scores[mm])


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for network processing')
    parser.add_argument('--metric-type', type=int, default=0, choices=[0,1,11,2,21,100,], help='Metric type')
    parser.add_argument('--testset', type=str, default='tt', choices=['cv','tt'], help='testset')
    main(parser.parse_args())
