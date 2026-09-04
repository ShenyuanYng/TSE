mkdir $HOME/corpus/VoicebankDemandEXP/
mkdir $HOME/corpus/VoicebankDemandEXP/wav16k/
mkdir $HOME/corpus/VoicebankDemandEXP/wav16k/tr/
ln -s $HOME/corpus/VoicebankDemand/clean_trainset_28spk_wav_16k $HOME/corpus/VoicebankDemandEXP/wav16k/tr/s1
ln -s $HOME/corpus/VoicebankDemand/noisy_trainset_28spk_wav_16k $HOME/corpus/VoicebankDemandEXP/wav16k/tr/mix
mkdir $HOME/corpus/VoicebankDemandEXP/wav16k/cv/
ln -s $HOME/corpus/VoicebankDemand/clean_testset_wav_16k $HOME/corpus/VoicebankDemandEXP/wav16k/cv/s1
ln -s $HOME/corpus/VoicebankDemand/noisy_testset_wav_16k $HOME/corpus/VoicebankDemandEXP/wav16k/cv/mix
# use cv set as the tt set
ln -s $HOME/corpus/VoicebankDemandEXP/wav16k/cv $HOME/corpus/VoicebankDemandEXP/wav16k/tt
