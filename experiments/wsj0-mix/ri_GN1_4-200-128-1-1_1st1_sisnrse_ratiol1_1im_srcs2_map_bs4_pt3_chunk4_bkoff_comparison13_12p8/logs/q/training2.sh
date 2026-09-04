#!/bin/bash
cd /home/cse30029918/example_code
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python train.py --exp-def exp_defs/ri_default_params.yaml --gpus 1 --num-workers 4 --save-path experiments --save-name wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8 --feature-type ri --rms-norm 2 --model-name DPRNNTF --seqmodel-type 3861 --num-layers 4 --num-sources 2 --loss-items sisnrse --mag-sf 1.0 --masking-or-mapping 1 --samplerate 8000 --num-imics 1 --num-omics 1 --dataset wsj0-mix --max-num-mics 1 --max-num-omics 1 --data-orig-path /data1/cse30029918/wsj0-mix/2speakers/ --data-path /data1/cse30029918//wsj0-mix/2speakers/ --act-type prelu --norm-type LayerNormalization --convnormact-type 2 --batch-size 4 --permutation-invariant True --epochs 150 --inwintime 0.032 --hoptime 0.008 --window-type sqrt_hann --mask-nonlinearity linear --target-trunc-max 5 --train-percent-check 1.0 --patience 3 --val-percent-check 1.0 --hostname ailab01 --gpuidx 0 --train-on-chunk 4 --num-hid-units 200 --emb-dim 128 --emb-ks 1 --emb-hs 1 --first-conv-use-act 1 --ref-mic 0 --pad-back-or-front 0 --mask-loss-type ratiol1 --cudnn-benchmark 0 --seed 8 
EOF
) >experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/training2.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/training2.log
  unset CUDA_VISIBLE_DEVICES.
fi
time1=`date +"%s"`
 ( python train.py --exp-def exp_defs/ri_default_params.yaml --gpus 1 --num-workers 4 --save-path experiments --save-name wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8 --feature-type ri --rms-norm 2 --model-name DPRNNTF --seqmodel-type 3861 --num-layers 4 --num-sources 2 --loss-items sisnrse --mag-sf 1.0 --masking-or-mapping 1 --samplerate 8000 --num-imics 1 --num-omics 1 --dataset wsj0-mix --max-num-mics 1 --max-num-omics 1 --data-orig-path /data1/cse30029918/wsj0-mix/2speakers/ --data-path /data1/cse30029918//wsj0-mix/2speakers/ --act-type prelu --norm-type LayerNormalization --convnormact-type 2 --batch-size 4 --permutation-invariant True --epochs 150 --inwintime 0.032 --hoptime 0.008 --window-type sqrt_hann --mask-nonlinearity linear --target-trunc-max 5 --train-percent-check 1.0 --patience 3 --val-percent-check 1.0 --hostname ailab01 --gpuidx 0 --train-on-chunk 4 --num-hid-units 200 --emb-dim 128 --emb-ks 1 --emb-hs 1 --first-conv-use-act 1 --ref-mic 0 --pad-back-or-front 0 --mask-loss-type ratiol1 --cudnn-benchmark 0 --seed 8  ) 2>>experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/training2.log >>experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/training2.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/training2.log
echo '#' Finished at `date` with status $ret >>experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/training2.log
[ $ret -eq 137 ] && exit 100;
touch experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/q/done.506707
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --ntasks=1  --mem 20G --cpus-per-task 5 --ntasks-per-node=1 --nodelist ailab03 -p RTXPRO5000 --gres=gpu:rtxpro5000:1 --time 2-00:00:00 --qos normal --dependency=afterany:21679  -D /home/cse30029918/example_code --open-mode=append -e experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/q/training2.log -o experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/q/training2.log  /home/cse30029918/example_code/experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/q/training2.sh >>experiments/wsj0-mix//ri_GN1_4-200-128-1-1_1st1_sisnrse_ratiol1_1im_srcs2_map_bs4_pt3_chunk4_bkoff_comparison13_12p8/logs/q/training2.log 2>&1
