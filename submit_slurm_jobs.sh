#
# 1. Code for simulating reverberant mixtures
# 2. Example code for sumitting slurm jobs for code that does not use GPU
#
. path.sh

num_gpus=0
slurm_workers=16 # need to be the same as the NUM_PROCESSES flag in, e.g., create-mixtures/generate_all_RIRs.py
. ./tools/cmd.sh ${num_gpus} ${slurm_workers} ""

# generate RIRs
#$cuda_cmd "create-mixtures/generate_all_RIRs.log" \
#    python create-mixtures/generate_all_RIRs.py
 
# spatialize WSJ0CAM to create reverb mixtures
#$cuda_cmd "create-mixtures/WSJCAM0_spatialize.log" \
#    python create-mixtures/WSJCAM0_spatialize.py

# downsample VoicebankDemand from 48k to 16k
#$cuda_cmd "create-mixtures/VoicebankDemand_downsample.log" \
#    python create-mixtures/VoicebankDemand_downsample.py

# VoicebankDemand only has speech, generate noise by using mixture - speech
$cuda_cmd "create-mixtures/VoicebankDemand_generate_noise.log" \
    python create-mixtures/VoicebankDemand_generate_noise.py
