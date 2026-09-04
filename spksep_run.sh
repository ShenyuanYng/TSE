#!/usr/bin/env bash

set -e
. path.sh

num_gpus=1
num_workers=4
save_path="experiments"
exp_def="exp_defs/ri_default_params.yaml"
other_options=""

stage=1
dataset="fixarr_h1+n-s1";
rms_norm=0
feature_type="ri"
model_name=SeqUNetDense; seqmodel_type=2; num_layers=4; num_sources=2;
masking_or_mapping=1;
mask_nonlinearity=linear; target_trunc_max=5;
loss_items="rix2+mag";
mag_sf=1.0;
samplerate=16000; num_imics=1; num_omics=1;
metric_type=0

act_type=elu; norm_type=GlobalInstanceNormalization; convnormact_type=2

batch_size=8

tag=""
permutation_invariant=False
epochs=100
inwintime=0.032; hoptime=0.008; window_type="sqrt_hann"

train_percent_check=1.0
val_percent_check=1.0
patience=5

train_on_chunk=4
mask_loss_type=l1

num_hid_units=128; emb_dim=16; emb_ks=8; emb_hs=4;
first_conv_use_act=1;
ref_mic=0;
pad_back_or_front=0;
testset=tt;

cudnn_benchmark=1;
use_slurm=1;
gpu=0;

. tools/parse_options.sh || exit 1;

nj=1 # how many submitted jobs
on_local_machine=1 # run on local machine or not
if [ $use_slurm = "1" ] && [[ "$HOSTNAME" == *.delta.ncsa.illinois.edu ]]; then
    num_workers=4 # for dataloader
    slurm_workers=$((num_workers + num_gpus)) # one thread for each gpu plus number of workers for data loading
    . ./tools/cmd.sh ${num_gpus} ${slurm_workers} ""
    nj=4
    on_local_machine=0
elif [ $use_slurm = "1" ] && [[ "$HOSTNAME" == login0* ]]; then
    num_workers=4
    slurm_workers=$((num_workers + num_gpus)) # one thread for each gpu plus number of workers for data loading
    . ./tools/cmd.sh ${num_gpus} ${slurm_workers} ""
    #
    # start, e.g., 2 jobs, the second one will start (i.e., reload checkpoint and continue training) after the first one finishes.
    # in this case, we say the second job depend on the first one.
    # this is because there is usually a time limit (e.g., 2 days) set by slurm admin for each node,
    # but our experiment could need many days to finish.
    # see the depend_jobid flag in this code for the details.
    #
    # when you use scancel to kill your job, remember to also cancel the dependant jobs.
    #
    nj=2
    on_local_machine=0
elif [ $use_slurm = "1" ] && [[ "$HOSTNAME" == ailab0* ]]; then
    num_workers=4
    slurm_workers=$((num_workers + num_gpus)) # one thread for each gpu plus number of workers for data loading
    . ./tools/cmd.sh ${num_gpus} ${slurm_workers} ""
    nj=2
    on_local_machine=0
else
    export sdr_cmd="run.pl"
    export cuda_cmd="run.pl"
    export CUDA_VISIBLE_DEVICES=$gpu
fi

#local_dir=/scratch/wangzhon/
local_dir=/scratch/`whoami`/
if [[ "$HOSTNAME" == *.delta.ncsa.illinois.edu ]] ; then
    # Depending on the slurm clusters we use, we could have a local SSD drive on the GPU node we submit jobs onto.
    # In this case, we can transfer our training data to the SSD drive beforehand to improve IO.
    local_dir=/scratch/bbjs/wang34/
fi
#
if [ $dataset = "fixarr_h1+n-s1" ]; then
    tmpdataset="fixarr"
    #data_orig_path="/scratch/wangzhon/MERL-dereverb_signals/$tmpdataset/"
    #data_orig_path="/scratch/`whoami`/MERL-dereverb_signals/$tmpdataset/"
    #data_orig_path="$HOME/corpus/MERL-dereverb_signals/$tmpdataset/"
    data_orig_path="/data1/`whoami`/MERL-dereverb_signals/$tmpdataset/"
    if [ $use_slurm = "1" ]; then
        #local_dir="$HOME/corpus/"
        local_dir="/data1/`whoami`/MERL-dereverb_signals"
        #
        # the following code is an example of transfering training data from one slurm cluster to another cluster
        # we use rsync to ensure exactly the same data, and rsync synchronizes data in an incremental way
        #
        #wavxk=wav16k
        #for dd in tr cv tt; do
        #    for ee in mix s1; do
        #        mkdir -p ${local_dir}/$tmpdataset/$wavxk/$dd/$ee
        #        rsync -av --delete --copy-links cse30029918@ailab01:$data_orig_path/$wavxk/$dd/$ee/ ${local_dir}/$tmpdataset/$wavxk/$dd/$ee/ &
        #    done
        #done
        data_path="${local_dir}/$tmpdataset/"
    else
        data_path=$data_orig_path
    fi
    max_num_mics=8
    max_num_omics=$max_num_mics
elif [ $dataset = "wsj0-mix" ]; then
    if [ $samplerate != "8000" ]; then
        exit 1;
    fi
    tmpdataset="wsj0-mix/${num_sources}speakers"
    #data_orig_path="$HOME/corpus/$tmpdataset/"
    data_orig_path="/data1/`whoami`/$tmpdataset/"
    if [ $use_slurm = "1" ]; then
        #local_dir="$HOME/corpus/"
        local_dir="/data1/`whoami`/"
        #wavxk=wav8k
        #for dd in tr cv; do
        #    for ee in mix s1 s2; do
        #    #for ee in mix s1 s2 s3; do
        #        mkdir -p ${local_dir}/$tmpdataset/$wavxk/min/$dd/$ee
        #        rsync -av --delete --links cse30029918@ailab01:$data_orig_path/$wavxk/min/$dd/$ee/ ${local_dir}/$tmpdataset/$wavxk/min/$dd/$ee/
        #        #rsync -av --delete --links $data_orig_path/$wavxk/min/$dd/$ee/ ${local_dir}/$tmpdataset/$wavxk/min/$dd/$ee/
        #    done
        #done
        #rsync -av --delete --links $data_orig_path/for_stacking/ /local-data/zwang/$tmpdataset/for_stacking/
        #data_path="${local_dir}/$tmpdataset/"
        data_path=${local_dir}/$tmpdataset/
    else
        data_path=$data_orig_path
    fi
    max_num_mics=1
    max_num_omics=$max_num_mics
elif [[ $dataset = VoicebankDemandEXP ]]; then
    if [ $samplerate != "16000" ]; then
        exit 1;
    fi
    tmpdataset=VoicebankDemandEXP
    #data_orig_path="$HOME/corpus/$tmpdataset/"
    data_orig_path="/data1/`whoami`/$tmpdataset/"
    if [ $use_slurm = "1" ]; then
        #local_dir="$HOME/corpus/"
        local_dir="/data1/`whoami`/"
        data_path="${local_dir}/$tmpdataset/"
    else
        data_path=$data_orig_path
    fi
    max_num_mics=1
    max_num_omics=$max_num_mics
else
    echo "$dataset not supported. "
    exit 1;
fi

save_name=${dataset}/
save_name=${save_name}/$feature_type
if [ $inwintime != "0.032" ]; then
    save_name=${save_name}_iwt${inwintime}
fi
if [ $hoptime != "0.008" ]; then
    save_name=${save_name}_ht${hoptime}
fi
if [ $rms_norm != "2" ]; then
    save_name=${save_name}_rms${rms_norm}
fi
if [ $model_name = "DPRNNTF" ]; then
    if [ $model_name != "DPRNNTF" ]; then
        save_name=${save_name}_${model_name}
    fi
    if [ $seqmodel_type = "3861" ]; then
        save_name=${save_name}_GN1
    fi
    save_name=${save_name}_${num_layers}-${num_hid_units}-${emb_dim}-${emb_ks}-${emb_hs}
    if [ $first_conv_use_act != "0" ]; then
        save_name=${save_name}_1st${first_conv_use_act}
    fi
fi
#
norm_str=""
if [ $norm_type = "GlobalInstanceNormalization" ]; then
    norm_str="lntf"
elif [ $norm_type = "GlobalLayerNormalization" ]; then
    norm_str="ln"
elif [ $norm_type = "LayerNormalization" ]; then
    norm_str="cln"
else
    echo "$norm_type not supported" && exit 1;
fi
if [ $convnormact_type = "2" ]; then
    #convnormact_str=conv${act_type}${norm_str}
    save_name=${save_name}
elif [ $convnormact_type = "1" ]; then
    convnormact_str=conv${norm_str}${act_type}
    save_name=${save_name}_${convnormact_str}
elif [ $convnormact_type = "0" ]; then
    convnormact_str=${norm_str}${act_type}conv
    save_name=${save_name}_${convnormact_str}
else
    echo "$convnormact_type not supported" && exit 1;
fi
#
save_name=${save_name}_${loss_items}
if [ $mag_sf != "1.0" ]; then
    save_name=${save_name}_magsf${mag_sf}
fi
if [ $mask_loss_type != "l1" ]; then
    save_name=${save_name}_${mask_loss_type}
fi
save_name=${save_name}_${num_imics}im
if [ $num_omics != "1" ]; then
    save_name=${save_name}_${num_omics}om
fi
#
if [ $num_sources != "1" ]; then
    save_name=${save_name}_srcs${num_sources}
fi
#
if [ $pad_back_or_front != "0" ]; then
    save_name=${save_name}_padfront
fi
#
if [ $masking_or_mapping = "0" ]; then
    save_name=${save_name}_mask
    if [ $mask_nonlinearity != "tanh" ]; then
        save_name=${save_name}_${mask_nonlinearity}
    fi
    if [ $target_trunc_max != "5" ] && [ $mask_nonlinearity != "linearNoT" ]; then
        save_name=${save_name}_rmax${target_trunc_max}
    fi
elif [ $masking_or_mapping = "1" ]; then
    save_name=${save_name}_map
fi
if [ $train_percent_check != "1.0" ]; then
    save_name=${save_name}_trn${train_percent_check}
fi
if [ $val_percent_check != "1.0" ]; then
    save_name=${save_name}_val${val_percent_check}
fi
if [ $batch_size != "1" ]; then
    save_name=${save_name}_bs${batch_size}
fi
#
if [ $patience != "2" ]; then
    save_name=${save_name}_pt${patience}
fi
#
save_name=${save_name}_chunk${train_on_chunk}
#
if [ $cudnn_benchmark != "1" ]; then
    save_name=${save_name}_bkoff
fi
save_name=${save_name}$tag
#
if [ "$CONDA_DEFAULT_ENV" != "tums_cuda11p8" ]; then
    if [ "$CONDA_DEFAULT_ENV" = "tums_cuda12p8" ]; then
        save_name=${save_name}_12p8
    else
        exit 1;
    fi
fi
logdir=$save_path/$save_name
logdir=$logdir/logs
mkdir -p $logdir

if [ $stage -le 1 ]; then

explog=$logdir/training.log
echo "Experiment logfile: `pwd`/$explog"
echo "Begin Training"
if [ -e $explog ]; then
    echo "$explog exists, delete before proceed."
    exit 1;
fi

depend_jobid=
for jj in `seq 1 $nj`; do
    if [ $jj = "1" ]; then
        logname=training.log
    else
        logname=training${jj}.log
    fi
    if [ $on_local_machine = "0" ]; then
        . ./tools/cmd.sh ${num_gpus} ${slurm_workers} "$depend_jobid"
    fi
    explog=$logdir/$logname
    $cuda_cmd "$explog" \
        python train.py \
        --exp-def $exp_def \
        --gpus $num_gpus \
        --num-workers $num_workers \
        --save-path $save_path \
        --save-name $save_name \
        --feature-type $feature_type --rms-norm $rms_norm \
        --model-name $model_name --seqmodel-type $seqmodel_type --num-layers $num_layers --num-sources $num_sources \
        --loss-items $loss_items --mag-sf $mag_sf \
        --masking-or-mapping $masking_or_mapping \
        --samplerate $samplerate --num-imics $num_imics --num-omics $num_omics \
        --dataset $dataset --max-num-mics $max_num_mics --max-num-omics $max_num_omics \
        --data-orig-path $data_orig_path --data-path $data_path \
        --act-type $act_type --norm-type $norm_type --convnormact-type $convnormact_type \
        --batch-size $batch_size \
        --permutation-invariant $permutation_invariant \
        --epochs $epochs \
        --inwintime $inwintime --hoptime $hoptime --window-type $window_type \
        --mask-nonlinearity $mask_nonlinearity --target-trunc-max $target_trunc_max \
        --train-percent-check $train_percent_check --patience $patience \
        --val-percent-check $val_percent_check \
        --hostname $HOSTNAME --gpuidx $gpu \
        --train-on-chunk $train_on_chunk \
        --num-hid-units $num_hid_units --emb-dim $emb_dim --emb-ks $emb_ks --emb-hs $emb_hs \
        --first-conv-use-act $first_conv_use_act \
        --ref-mic $ref_mic \
        --pad-back-or-front $pad_back_or_front \
        --mask-loss-type $mask_loss_type \
        --cudnn-benchmark $cudnn_benchmark \
        --seed $(($jj + 6)) \
        $other_options &
        #$other_options || exit 1;
    sleep 10s
    if [ $on_local_machine = "0" ]; then
        depend_jobid=`awk 'NR==1 {print $NF; exit}' $logdir/q/$logname`
    fi
done
wait;
exit 0;
fi

if [ $stage -le 30 ]; then

script_name=separate_simulated

explog=$logdir/testing_${script_name}

if [ $metric_type = "0" ]; then
    explog=${explog}.SISDR
elif [ $metric_type = "1" ]; then
    explog=${explog}.PESQ
elif [ $metric_type = "11" ]; then
    explog=${explog}.PESQwb
elif [ $metric_type = "2" ]; then
    explog=${explog}.eSTOI
elif [ $metric_type = "21" ]; then
    explog=${explog}.STOI
elif [ $metric_type = "100" ]; then
    explog=${explog}.BSSEvalSDR
fi

explog=${explog}.$testset
explog=${explog}.log

echo "Experiment logfile: $explog"
echo "Begin Testing"

if [ -e $explog ]; then
    echo "$explog exists, delete before proceed."
    exit 1; 
fi

$cuda_cmd "$explog" \
    python ${script_name}.py \
    --checkpoint $save_path/$save_name \
    --batch-size 1 \
    --metric-type $metric_type \
    --testset $testset \
    || exit 1;
    exit 0;
fi

echo "Experiment done"
