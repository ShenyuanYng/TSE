# Copyright 2024 Southern University of Science and Technology (Author: Zhong-Qiu Wang).
# Apache 2.0.
#
#
# Please do not upload the code online without my permission
#
#
# 1. Be aware, every experiment you run should be reproducible. This is VERY VERY VERY important for academic research. This code can help you achieve that.
# 2. The code creates a directory for each experiment you run, based on your experiment configuration. You can use any flags (e.g., model_name, masking_or_mapping) you consider important to form the directory name.
# 3. This code can also help you manage all your experiments you have run. For example, if you have run 50 experiments for your study over half a year, you may forget what you did for each experiment. Based on the directory name, you can likely figure out what you did. For me, based on this code, I can figure out what I did years ago.
# 4. This code writes important information about training progress into log files. The saved log file can be very helpful in many cases, especially for the purpose of reproducibility. Please consider following this style.
#
#
use_slurm=0; # if the GPU server is a local one, need to configure which GPU to use
#use_slurm=1; # if the GPU server is managed by slurm
#
# If you use slurm, here is a list of very useful commands (all can be found in my .bashrc at /home/wangzq_lab/cse30029918/.bashrc). They are customized by me.
# 1. ALLsqueue           <-- list all the jobs in the queue (by all the users on all the nodes)
# 2. mysq                <-- list your jobs
# 3. jobinfo $slurmjobid <-- show slurm details of your job
# 4. sparse              <-- list the available GPUs of all nodes, you can check how I implement it in /home/wangzq_lab/cse30029918/tools/sparse.py
# 5. gmem $slurmjobid    <-- check GPU memory consuption of a job. It is very useful, as you need to have an idea about how much GPU memory a job uses. Please use the available memory of your requested GPU as much as possible.
# 6. mysinfo             <-- check info of gpu servers
# 7. scancel $slurmjobid <-- to kill a job
# 8. more about slurm can be found at https://slurm.schedmd.com/overview.html
#
# When using slurm-managed systems, do not run any computation-intensive code on the login node (i.e., the head node). Every job should be submitted to clusters via the login node.
# Example things you can do on login node include, e.g., code editing, installing software.
# Example code of submitting non-GPU jobs can be found at submit_slurm_jobs.sh.
#
# Your will need to understand tools/slurm.conf, tools/cmd.sh and path.sh on how to set paritition, nodelist etc.
# A good practice to test your understanding is to figure out how to run your job on a particular node (with a particular type of GPU, such as a100).
#
#
if [ $HOSTNAME = "zqgpuserver1" ]; then
    gpu=0 #RTX4090
    gpu=1 #RTX4090
    gpu=2 #RTX4090
    gpu=3 #RTX4090
else
    gpu=0
fi
#
#
# segment length for training
# 1. For each mixture signal, we randomly cut a segment (with length equal to train_on_chunk) from it for training in each epoch.
# 2. We select batch_size (e.g., 4) segments to form each minibatch
# 2. The selected segments are then padded to the length of the selected segment with the maximum length.
# 3. Some mixture signal could be shorter than, e.g., 4 second. In this case, the selected segment is shorter than 4 second. The padded length (i.e., max of all the segment lengths in a mini-batch) could be actually shorter than 4 second. This means that the number of frames we feed for DNN training could vary (be careful with the torch.backends.cudnn.benchmark flag in this case), and the GPU memory comsuption could fluctuate.
# Please see utils/collate.py for details, which is called in _build_data_loader in SourceSeparator in source_separator.py
train_on_chunk=4;
#train_on_chunk=8;
#batch_size=1;
batch_size=4;
#
#
# reduce learning rate if no improvement is obtained in validation loss for [patience] epochs
#patience=2; 
patience=3;
#
#
train_percent_check=1.0; val_percent_check=1.0; # using full training mixtures in each epoch
#train_percent_check=0.25; val_percent_check=1.0; # sample 25% training mixtures in each epoch
train_percent_check=0.01; val_percent_check=0.01; train_on_chunk=2; batch_size=1; patience=1; # for debug purpose
#
mask_loss_type=ratiol1; # use L1 distance in loss
#mask_loss_type=ratiol2; # use L2 distance in loss
act_type=prelu; norm_type=LayerNormalization; convnormact_type=2; #convpreluln
#first_conv_use_act=0;
first_conv_use_act=1;
#
epochs=150;
target_trunc_max=5; # truncate mask values to [-target_trunc_max, +target_trunc_max]
masking_or_mapping=1; mask_nonlinearity=linear; #<-- mapping
#masking_or_mapping=0; mask_nonlinearity=linear; #<-- masking
#masking_or_mapping=0; mask_nonlinearity=tanh; #<-- masking
#
#inwintime=0.016; hoptime=0.008; # 50% overlap
inwintime=0.032; hoptime=0.008; # 75% overlap, consider using 50% overlap, which can reduce computation by 50%
#
rms_norm=2; # normalize mixture standard deviation to 1.0. We do this for each microphone separately.
#
#num_imics=8; # 8-channel input
num_imics=1; # 1-channel input
#
model_name=DPRNNTF; # TF-GridNet
#num_hid_units=200; emb_dim=64; emb_ks=2; emb_hs=2; seqmodel_type=3861; #
num_hid_units=200; emb_dim=128; emb_ks=1; emb_hs=1; seqmodel_type=3861; # to reduce computation, consider reducing num_hid_units or emb_dim, increasing emb_ks or emb_hs
#
cudnn_benchmark=0; # this flag may speed up training, if the padded signal length in each mini-batch is always fixed, e.g., always 4 seconds
tag=""; # you can give a custom name to your experiment, but I seldomly use it
#tag="_comparison0"; # you can give a custom name to your experiment, but I seldomly use it
#
num_sources=1; # number of target sources to predict
#
# number of layers (e.g., number of TF-GridNet blocks)
num_layers=2;
num_layers=3;
num_layers=4;
#
ref_mic=0; # reference microphone index, usually just 0 (i.e., the first microphone)
#loss_items="rix2+mag";
#loss_items="rix2+mag+proj";
loss_items="sisnrse";
#loss_items="rix2+mag+proj+rix2W0p1"; # 0.1 weight for rix2
#loss_items="rix2+mag+proj+mc"; # only for num_sources > 1
#
mag_sf=1.0;
#mag_sf=0.3; masking_or_mapping=0; # power compression to magnitude for both input and loss. NOTE masking is still performed on original STFT.
#
# Here we provide examples of using this code to do three different tasks, try them
dataset="wsj0-mix"; samplerate=8000; num_sources=2; permutation_invariant=True; num_imics=1; # for speaker separation
#dataset="fixarr_h1+n-s1"; samplerate=16000; num_sources=1; permutation_invariant=False; num_imics=1; # for speech dereverb
#dataset="VoicebankDemandEXP"; samplerate=16000; num_sources=1; permutation_invariant=False; num_imics=1; # for speech enhancement. Note that for VoicebanDemand, we use its test set as validation set. This is not right, but existing studies use this way.
#dataset="VoicebankDemandEXP"; samplerate=16000; num_sources=2; permutation_invariant=False; num_imics=1; # for speech enhancement. Note that for VoicebanDemand, we use its test set as validation set. This is not right, but existing studies use this way. Predict both speech and noise
#
# what evaluation metric to use, run-time parameter
metric_type=0 # SI-SDR
#metric_type=11 # PESQwb
#metric_type=21 # STOI
#metric_type=100 # BSSEvalSDR
#metric_type=1 # PESQ
#metric_type=2 # eSTOI
#
# what testset to evaluate, run-time parameter
testset=cv;
testset=tt;
#
# For training, usually I stop jobs once the learning rate is decayed serveral times (e.g., from the initial 1e-3 down to 6.25-05).
# This can save computation, as the model would be almost converged. Please follow this way.
stage=1; #training
#stage=30; # evaluation, separate_simulated
#
./spksep_run.sh \
        --stage $stage \
        --dataset $dataset \
        --rms_norm $rms_norm \
        --model_name $model_name --seqmodel_type $seqmodel_type --num_layers $num_layers --num_sources $num_sources \
        --loss_items $loss_items --mag_sf $mag_sf \
        --samplerate $samplerate \
        --masking_or_mapping $masking_or_mapping \
        --mask_nonlinearity $mask_nonlinearity --target_trunc_max $target_trunc_max \
        --num_imics $num_imics \
        --metric_type $metric_type \
        --act_type $act_type --norm_type $norm_type --convnormact_type $convnormact_type \
        --batch_size $batch_size \
        --tag "$tag" \
        --permutation_invariant $permutation_invariant \
        --epochs $epochs \
        --inwintime $inwintime --hoptime $hoptime \
        --patience $patience \
        --train_percent_check $train_percent_check --val_percent_check $val_percent_check \
        --train_on_chunk $train_on_chunk \
        --mask_loss_type $mask_loss_type \
        --num_hid_units $num_hid_units --emb_dim $emb_dim --emb_ks $emb_ks --emb_hs $emb_hs \
        --first_conv_use_act $first_conv_use_act \
        --ref_mic $ref_mic \
        --testset $testset \
        --cudnn_benchmark $cudnn_benchmark \
        --use_slurm $use_slurm \
        --gpu $gpu \
        || exit 1;
