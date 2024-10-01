#!/usr/bin/env python

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.parameters import constants
#constants.set('room_isinside_max_iter', 100) #not useful
import os
import pickle

from multiprocessing import Pool

#data_root = '/scratch/wangzhon/MERL-dereverb_signals/'
data_root = os.path.expanduser('~') + '/corpus/MERL-dereverb_signals/'

fs = 16000
rir_root = data_root + '/fixarr/RIR_' + str(fs//1000) + 'k'
os.system('mkdir -p ' + rir_root)

room_l_range = [5., 10.] # in meters
room_w_range = [5., 10.] # in meters
room_h_range = [3., 4.] # in meters

array_z_range = [1., 2.] # in meters
array_xy_jittering = 0.5 # in meters

spk_z_range = [1., 2.] # in meters
spk_dist_range = [0.75, 2.5] # in meters

T60_range = [0.2, 1.3] # in seconds 

M = 8 # on a circler
R = 0.1 # in meters, following REVERB 
S = 4 # number of sources

def draw(range):
    assert range[1] > range[0]
    return (range[1]-range[0])*np.random.rand()+range[0]

def check_spks_angle_pos(all_spk_angle, spk_pos, room_dim):
    for ii in range(S):
        for jj in range(ii+1,S):
            if np.abs(all_spk_angle[ii]-all_spk_angle[jj]) < 15:
                return False
    for ii in range(len(spk_pos)):
        if spk_pos[ii][0] < 0.5 or spk_pos[ii][0] > room_dim[0] - 0.5:
            return False
        if spk_pos[ii][1] < 0.5 or spk_pos[ii][1] > room_dim[1] - 0.5:
            return False
    return True

MAX_NUM_RIRS = 48000
#MAX_NUM_RIRS = 10000 # for testing, rollback to the above one

def simulate_rir(rr):

    print("%d/%d"%(rr, MAX_NUM_RIRS))
     
    if rr == 17060:
        # There is likely some problems with pyroomacoustics
        # For this example, if nseed == rr*7, the first channel of h_anecho of the first source is all-zero
        # don't know why
        nseed = rr*7 + 1
    else:
        nseed = rr*7
    np.random.seed(nseed)

    room_l = draw(room_l_range)
    room_w = draw(room_w_range)
    room_h = draw(room_h_range)
    room_dim = [room_l, room_w, room_h]

    array_x = room_l/2. + (np.random.rand()*2.-1.)*array_xy_jittering
    array_y = room_w/2. + (np.random.rand()*2.-1.)*array_xy_jittering
    array_z = draw(array_z_range)
    array_center = [array_x, array_y, array_z]

    angle_step = 360. / M
    first_mic_angle = draw([0., angle_step])
    mic_pos = []
    for mm in range(M):
        mic_angle = (first_mic_angle+mm*angle_step)/180.*np.pi
        mic_pos.append([array_x+R*np.cos(mic_angle), array_y+R*np.sin(mic_angle), array_z])

    while True:
        spk_pos = []
        all_spk_angle = []
        all_spk_dist = []
        for ss in range(S):
            spk_dist = draw(spk_dist_range)
            spk_angle = draw([0.,360.])
            all_spk_angle.append(spk_angle)
            all_spk_dist.append(spk_dist)
            spk_angle = spk_angle/180.*np.pi
            spk_z = draw(spk_z_range)
            spk_pos.append([array_x+spk_dist*np.cos(spk_angle), array_y+spk_dist*np.sin(spk_angle), spk_z])
        flag = check_spks_angle_pos(all_spk_angle, spk_pos, room_dim)
        if flag:
            break

    epsilon = 0.1
    for ii in range(len(spk_pos)):
        if spk_pos[ii][0] <= epsilon or spk_pos[ii][0] >= room_dim[0]-epsilon:
            raise
        if spk_pos[ii][1] <= epsilon or spk_pos[ii][1] >= room_dim[1]-epsilon:
            raise
        if spk_pos[ii][2] <= epsilon or spk_pos[ii][2] >= room_dim[2]-epsilon:
            raise
        
    T60 = draw(T60_range)
    try:
        absorption, max_order = pra.inverse_sabine(T60, room_dim)
    except ValueError:
       return
    room = pra.ShoeBox(room_dim, fs=fs,
            materials=pra.Material(absorption), max_order=max_order)
    try:
        for tmp in spk_pos:
            room.add_source(tmp)
    except ValueError:
       return
    room.add_microphone_array(np.asarray(mic_pos).T)
    room.compute_rir()
    # room.rir: outer list is for microphones and inner is for sources
    rir_len = min([min([len(tmptmp) for tmptmp in tmp]) for tmp in room.rir])
    rir_len = min(np.ceil((T60+0.1)*fs).astype(int),rir_len)
    h_reverb = np.stack([np.stack([tmptmp[:rir_len] for tmptmp in tmp],axis=0) for tmp in room.rir],axis=0) # [M, S, -1]

    room = pra.ShoeBox(room_dim, fs=fs,
            materials=pra.Material(absorption), max_order=0)
    try:
        for tmp in spk_pos:
            room.add_source(tmp)
    except ValueError:
       return
    room.add_microphone_array(np.asarray(mic_pos).T)
    room.compute_rir()
    rir_len = max([max([len(tmptmp) for tmptmp in tmp]) for tmp in room.rir])
    h_anecho = np.stack(
            [np.stack(
                [np.pad(tmptmp,(0,rir_len-len(tmptmp)),'constant',constant_values=(0,0)) for tmptmp in tmp],
                axis=0) for tmp in room.rir],
            axis=0) # [M, S, -1]

    """
    print(T60)
    print(mic_pos)
    print(spk_pos)
    for tmp in room.rir:
        for tmptmp in tmp:
            print(np.sum(np.abs(tmptmp)))
    print(np.sum(np.abs(h_anecho[:,0]), axis=-1))
    """

    """
    import matplotlib.pyplot as plt
    #row, col = 2, 1
    #plt.subplot(row, col, 1)
    #plt.imshow(np.flipud(xx.T), cmap="jet")
    plt.plot(h_reverb[0,0],'rx-')
    #plt.subplot(row, col, 2)
    plt.plot(h_anecho[0,0],'b*-')
    plt.show()
    input("")
    """

    pickle.dump({
            'h_anecho': h_anecho, 'h_reverb': h_reverb,
            'T60': T60, 'spk_pos': spk_pos, 'mic_pos': mic_pos, 'room_dim': room_dim,
            'R': R, 'M': M, 'S': S, 'all_spk_angle': all_spk_angle, 'all_spk_dist': all_spk_dist,
            },
            open(rir_root + '/RIR_%06d.pickle'%rr, 'wb'),-1)

#NUM_PROCESSES = 8 #catbus
#NUM_PROCESSES = 12 #totoro
#NUM_PROCESSES = 16 #zqgpuserver1
#NUM_PROCESSES = 4 #sustech_cse_cluster_2.sh
NUM_PROCESSES = 16 #sustech_cse_cluster_2.sh
p = Pool(NUM_PROCESSES)
#p.map(simulate_rir, range(0,MAX_NUM_RIRS//2))
#p.map(simulate_rir, range(MAX_NUM_RIRS//2,MAX_NUM_RIRS))
p.map(simulate_rir, range(MAX_NUM_RIRS))
#for rr in range(MAX_NUM_RIRS):
#    simulate_rir(rr)
"""
for rr in [17060]:
    simulate_rir(rr)
"""
