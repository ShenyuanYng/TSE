# Download Anaconda3
# cd ~
# curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
# bash Anaconda3-2020.02-Linux-x86_64.sh

# create env
conda create -n tums_cuda12p8 python=3.9.18 # make sure the conda is the installed one

# install software
pip install -r requirements_cuda12p8.txt --no-cache-dir

# remove env
# conda remove --name tums_cuda12p8 --all

# use software source in China to speed up the installation
# --ignore-installed
# -i https://pypi.tuna.tsinghua.edu.cn/simple

# In case of error:
#   OSError: ctypes.util.find_library() did not manage to locate a library called 'sndfile'
# In your env, run
#   conda install conda-forge::libsndfile

# run.sh provides code for training and evaluating models for speech dereverb (on the fixarr_h1+n-s1 dataset), enhancement (on VoicebanDemand), and separation (on wsj0-2mix).
# For generating the fixarr_h1+n-s1 dataset, check submit_slurm_jobs.sh for details.
# The code is heavily commented.
