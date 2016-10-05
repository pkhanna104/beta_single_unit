'''
The goal is to produce a .mat file that has 1.5 sec before GO cue and 1 sec after go cue. 
We will then use the 'neural signal' mat to look at alignment of cells to filtered LFP
'''
import scipy.io as sio
import numpy as np
import scipy.signal
import collections
import matplotlib.pyplot as plt
import math
from load_files import master_save_directory
import load_files
import un2key

cmap = ['darkcyan', 'royalblue', 'gold', 'orangered']

def parse_spks_and_LFP(cell_dict, days, blocks, mc_indicator, bef=1.5, aft=1., animal='grom'):
    ''' method to spit out cell & LFPs to go to state_space_beta_bursts. 

    inputs: 
        cell_dict: dictionary with dates as keys (e.g. '022715')
        days: list of strings of dates: ['022715', '022815']
        blocks: list of strings of blocks: ['abcd','acde']
        bef : how much time to take before Go cue. NOTE. DO NOT CHANGE THIS --
            all other time slices in state_space_beta_bursts are with reference
            to this time (assuming t = 0 is 1.5 sec before the Go cue)

        after: How much time after the Go cue to take. Less fatal if changes are made. 

    '''

    #Make sure same number of blocks / days etc.
    assert len(days) == len(blocks) == len(cell_dict.keys())

    #For index storage: 
    trl_cnt = 0
    lfp_lab = {}
    go_times = {}
    rt_dict = {}

    #For storage of spikes / lfps

    units = ['a', 'b', 'c', 'd']
    spk_dict = {} #Spikes dict
    lfp_dict = {} #LFP dict

    go_times_dict = {}
    
    #Cycle through the days: 
    for i_d, day in enumerate(days):
        cell_list = cell_dict[day]
        mc = mc_indicator[i_d]

        #Cycle through blocks to get go times first: 
        for ib, b in enumerate(blocks[i_d]):
            loaded = False
            fx = 0
            t = load_files.load(b, day, subj=animal, include_hdfstuff=True)

            if t is not None:
                if animal=='grom':
                    lfp_lab[b, day], rt_dict[b, day], go_times[b], trl_cnt, lfp_offset = get_go_times_grom(t, mc[ib])
                elif animal == 'cart':
                    lfp_lab[b, day], rt_dict[b, day], go_times[b], trl_cnt, lfp_offset = get_go_times_cart(t, mc[ib])
                
                spk = t #Copy loaded file to a more attractive name ;) 
                keys = spk.keys()
                spk_keys = cell_list
                cnt = -1 #Reset cnt before each block: 

                spk_dict[b, day] = dict()
                lfp_dict[b, day] = np.zeros((trl_cnt, int((bef+aft)*1000)))

                for i, go_ in enumerate(go_times[b]):
                    cnt += 1 #trial counter

                    for un in spk_keys:
                        if un[-2:] == 'wf':
                            #Trim key down: 
                            un = un[:-3]

                        #Add correct number of zeros
                        if animal=='grom':
                            un = un2key.convert(un)
                                
                        #Init spike counts if necessary: 
                        if un not in spk_dict[b, day].keys():
                            spk_dict[b, day][un] = np.zeros((trl_cnt, int((bef+aft)*1000)))

                        if np.any(np.array(spk[un]).shape > 1):
                            ts_arr = np.squeeze(spk[un])
                        else:
                            ts_arr = spk[un]

                        #Spikes: ASSUME TS_ARR is same time stamps as LFP ts
                        spk_dict[b, day][un][cnt,:] = bin1ms(ts_arr, go_, bef=bef, aft=aft)

                    #LFPs
                    if animal == 'grom':
                        lfp_aligned_go = go_ - lfp_offset
                        gg = int(np.round(lfp_aligned_go*1000.))
                        try:
                            lfp_dict[b, day][cnt, :] = spk['AD74'][gg-(float(bef*1000)):gg+(float(aft*1000)),0]
                        except:
                            print 'skipping LFP addition:', cnt, day, b
                            try:
                                print spk['AD74'].shape
                            except:
                                print 'No channel 74 in ', day, b

                    elif animal == 'cart':
                        #Make sure there are 124 cahnnels: 
                        go_ix_ = np.argmin(np.abs(spk['ad124_ts'] - go_))

                        if tmp[1] == 256:
                            lfp_dict[b, day][cnt, :] = spk['ad124'][go_ix_ - (bef*1000): go_ix_ + (aft*1000)] 


                #Spike times, LFP times
                go_times_dict[b, day] = [go_times[b] , go_times[b] - lfp_offset]

    return spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, bef, aft, go_times_dict



def get_go_times_grom(t, mci):
    strobed = t['Strobed']

    #Get time delays 
    try:
        ts_key = 'AD33_ts'
        offset = t[ts_key]
    except:
        tmp = [k for ik, k in enumerate(t.keys()) if k[:2]=='AD' and k[-2:]=='ts']
        ts_key = tmp[0]
        offset = t[ts_key]
    if np.any(np.array(offset.shape) > 1): #Any size is nonzero
        offset = np.squeeze(offset)
        lfp_offset = offset[1] #Second one is the LFP offset
    else:
        lfp_offset = offset[0, 0]

    #Make sure no strobed events occur before:
    assert np.all(strobed[:,0] - lfp_offset > 0.)

    #Make sure length of AD channel is > last event - offset: 
    assert (len(t[ts_key[:-3]])/1000.) > (strobed[-1, 0] - lfp_offset)

    #MAKE SURE ALIGNED TO BEGINNING OF FILE: 
    rew_ix = np.nonzero(strobed[:, 1]==9)[0] 

    if int(mci) == 1:
        go_ix = rew_ix - 4
        ix_valid = np.nonzero(go_ix >=0)[0]
        lfp_lab[b, day] = strobed[rew_ix[ix_valid] - 6, 1]
        rt_dict[b, day] = strobed[go_ix[ix_valid]+1, 0] - strobed[go_ix[ix_valid], 0]
    
    else:
        go_ix = rew_ix - 3
        ix_valid = np.nonzero(go_ix >=0)[0]
        lfp_lab = strobed[rew_ix[ix_valid] - 5, 1]
        rt_dict = strobed[go_ix[ix_valid]+1, 0] - strobed[go_ix[ix_valid], 0]
    
    go_times = strobed[go_ix[ix_valid], 0] #FOR SPIKES ONLY
    trl_cnt = len(go_ix[ix_valid])

    return lfp_lab, rt_dict, go_times, trl_cnt, lfp_offset

def get_go_times_cart(t, mci):
    #get go times: 
    hdf = t['hdf']
    ts_func = t['ts_func']

    rew_hdf_ix = np.array([i['time'] for i in hdf.root.task_msgs[:] if i['msg'] == 'reward'])
    if int(mci) == 1:
        #Manual control
        go_ix = np.array([hdf.root.task_msgs[i-3]['time'] for i, j in enumerate(hdf.root.task_msgs[:]) if j['msg'] == 'reward'])
        reach_targ_loc = hdf.root.task[go_ix.astype(int)+8]['target']
        reach_targ_ang = np.array([math.atan2(y,x) for i, (x, y) in enumerate(reach_targ_loc[:, [0, 2]])])
        
        #Convert angle to kinarm codes:
        map_dict = np.array([[64, 0.], [66, np.pi/2.], [68, np.pi], [70, -1*np.pi/2.]])
        lfp_lab_ix = np.array([np.argmin(np.abs(i - map_dict[:, 1])) for i in reach_targ_ang])
        lfp_lab = map_dict[lfp_lab_ix, 0]

        rt_dict = []
    else:
        #BMI
        go_ix = np.array([hdf.root.task_msgs[i-2]['time'] for i, j in enumerate(hdf.root.task_msgs[:]) if j['msg'] == 'reward'])
        lfp_targ_ix = np.array([hdf.root.task_msgs[i-2]['time'] for i, j in enumerate(hdf.root.task_msgs[:]) if j['msg'] == 'mc_target'])
        lfp_targ = hdf.root.task[lfp_targ_ix.astype(int)]['lfp_target'][:, 2]
        lfp_lab = ((lfp_targ + 4.875)/4.875)+84
        rt_dict = []

    trl_cnt = len(go_ix)
    lfp_offset = 0
    go_times = ts_func(go_ix, 'plx')

    return lfp_lab, rt_dict, go_times, trl_cnt, lfp_offset



def bin1ms(ts_arr, go_, bef=1.5, aft=1):
    tmp = np.zeros(( (bef+aft)*1000., ))
    ix = np.nonzero(np.logical_and(ts_arr < go_+aft, ts_arr>= go_-bef))[0]
    if len(ix) > 0:
        ts_recenter = (ts_arr[ix] - go_)+ bef
        ts_bins = np.round(ts_recenter*1000).astype(int) 
        for t in ts_bins:
            if t<len(tmp):
                tmp[t] += 1
            else:
                print 'rounding means skipping: ', t, ' on trial w/ go:', go_

    return tmp

def lfp_lab_to_lfp_lab_arr(lfp_lab, blocks):
    lfp_lab_arr = []
    for b in blocks:
        lfp_lab_arr.append(lfp_lab[b])
    lfp_lab_arr = np.hstack((lfp_lab_arr))
    return lfp_lab_arr

def plot_mn_sd_FR_during_MC_vs_BetaC(d, lfp_lab, blocks, plot_set=1):
    lfp_lab_arr = []
    for b in blocks:
        lfp_lab_arr.append(lfp_lab[b])
    lfp_lab_arr = np.hstack((lfp_lab_arr))

    if plot_set == 1:
        f, ax = plt.subplots(nrows = 2)
    elif plot_set == 2:
        f, ax = plt.subplots(nrows = 4, ncols =4)
        hist = {}

    for iu, u in enumerate(d.keys()):
        arr = d[u]
        lfp = {}
        lfp_mc = {}

        proceed = 1
        mc = (1000*arr[:, 1500:]).reshape(-1)
        if np.sum(mc) == 0:
            proceed = 0
    
        for il, l in enumerate(np.sort(np.unique(lfp_lab_arr))):
            ix = np.nonzero(lfp_lab_arr==l)[0]
            lfp[l] = (1000*arr[ix, :1500]).reshape(-1)
            if np.sum(lfp[l]) == 0:
                proceed = 0

        if proceed:
            if plot_set == 1:
                for il, l in enumerate(np.sort(np.unique(lfp_lab_arr))):
                    ax[0].plot(iu + (0.2*(il+1)), np.mean(lfp[l]), 
                        color=cmap[il], marker='.',markersize=10)
                    
                    ax[1].plot(iu + (0.2*(il+1)), np.std(lfp[l]),
                        color=cmap[il], marker='.',markersize=10)

                ax[0].plot(iu, np.mean(mc), color='k', marker='.', markersize=10)
                ax[1].plot(iu, np.std(mc), color='k', marker='.', markersize=10)


            elif plot_set == 2:
                for il, l in enumerate(np.sort(np.unique(lfp_lab_arr))):
                    ax[il, 0].plot(np.mean(mc), np.mean(lfp[l]), '.')
                    try:
                        hist['mn', l].append(d_to_unity(np.mean(mc), np.mean(lfp[l])))
                    except:
                        hist['mn', l] = [d_to_unity(np.mean(mc), np.mean(lfp[l]))]

                    ax[il, 2].plot(np.std(mc), np.std(lfp[l]), '.')

                    try:
                        hist['std', l].append(d_to_unity(np.std(mc), np.std(lfp[l])))
                    except:
                        hist['std', l] = [d_to_unity(np.std(mc), np.std(lfp[l]))]
    if plot_set==2:
        for il, l in enumerate(np.sort(np.unique(lfp_lab_arr))):
            ax[il, 1].hist(hist['mn', l], 20)
            ax[il, 1].set_xlim([-3.5, 3.5])
            ax[il, 1].set_ylim([0, 20])
            ax[il, 1].plot(np.mean(hist['mn', l]), 0, 'r.')
            ax[il, 1].set_title(str(np.mean(hist['mn', l])))

            ax[il, 3].hist(hist['std',l], 20)
            ax[il, 3].set_xlim([-25, 25])
            ax[il, 3].set_ylim([0, 25])
            ax[il, 3].plot(np.mean(hist['std', l]), 0, 'r.')
            ax[il, 3].set_title(str(np.mean(hist['std', l])))

            ax[il, 0].plot([0, 50], [0, 50], 'k-')
            ax[il, 2].plot([0, 50], [0, 50], 'k-')
        plt.tight_layout()

def d_to_unity(mc, lfp):
    u = np.array([1, 1])
    u_ang = math.atan2(1,1)
    v = np.array([mc, lfp])
    cos_thet = np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))
    dist = np.sqrt(np.linalg.norm(v)**2 - (np.linalg.norm(v)*cos_thet)**2)
    if math.atan2(v[1], v[0]) > u_ang:
        return dist
    else:
        return -1*dist

def dig_lfp_d(lfp_d, bp_filt=[20, 45]):
    ''' 
    Filter and decide if it's an oscillatory episode
    '''
    Fs = 1000
    nyq = 0.5* Fs
    bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')

    #Use default padding:
    data_filt = scipy.signal.filtfilt(bw_b, bw_a, lfp_d, axis=1)

    #Now get the envelope:
    hilb = scipy.signal.hilbert(data_filt, axis=1)
    env = np.abs(hilb)

    thresh = np.percentile(env, 75)
    env[env>thresh] = 1
    env[env<=thresh] = 0

    return get_dict_of_islands(env)

def get_dict_of_islands(env):
    islands = {}
    for i in range(env.shape[0]):
        arr = env[i,:]
        cnt = 0
        used_ix = []
        for j, jx in enumerate(arr):
            if jx == 1 and j not in used_ix:
                islands[i, cnt] = [j]
                used_ix.append(j)
                on_island = True
                island_cnt = 0
                while on_island:
                    try:
                        if arr[j+island_cnt] == 1:
                            islands[i, cnt].append(j+island_cnt)
                            used_ix.append(j+island_cnt)
                            island_cnt+=1
                        else:
                            on_island= False
                            cnt += 1
                    except:
                        on_island = False
                        cnt += 1
    len_ = []
    for k in islands.keys():
        len_.append(len(islands[k]))

    #Keep everything over 50 ms: 
    env0 = env.copy()
    env0[:,:] = 0

    for k in islands.keys():
        if len(islands[k]) > 50:
            x, _ = k
            env0[x, islands[k]] = 1
    return env0

def plot_dig_by_lfp_targ(env0, lfp_lab_arr):
    f, ax = plt.subplots(nrows=2)
    for i, il in enumerate(np.unique(lfp_lab_arr)):
        ix = np.nonzero(lfp_lab_arr==il)[0]
        ax[0].plot(np.mean(env0[ix,:], axis=0), color=cmap[i])
        ax[1].bar(i, np.mean(env0[ix, 500:1500].reshape(-1)), color=cmap[i])

def fft_of_ACH(d, env0, lfp_d):
    ACH_in = np.zeros((len(d.keys()), 250))
    ACH_out = np.zeros((len(d.keys()), 250))
    ACH_all = np.zeros((len(d.keys()), 250))

    LFP_in = np.zeros((len(d.keys()), 500))
    LFP_out = np.zeros((len(d.keys()), 500))
    LFP_all = np.zeros((len(d.keys()), 500))

    ord_un = collections.OrderedDict()
    for k in d.keys():
        ord_un[k] = d[k]

    #Trials:
    for i in range(env0.shape[0]):
        # Time point w/in trial:
        for j in range(250, env0.shape[1]-250):
            #Keys:
            for ik, k in enumerate(ord_un.keys()):
                #If oscillatory episode:
                # if d[k][i, j] and env0[i, j]:
                #   for z in range(int(d[k][i, j])):
                #       ACH_in[ik, :]+= d[k][i, j-125:j+125]
                #       LFP_in[ik, :]+= lfp_d[i, j-125:j+125]
                # elif d[k][i, j] and not env0[i, j]:
                #   for z in range(int(d[k][i, j])):
                #       ACH_out[ik, :]+= d[k][i, j-125:j+125]
                #       LFP_out[ik, :]+= lfp_d[i, j-125:j+125]
                for z in range(int(d[k][i, j])):
                    #ACH_all[ik, :]+= d[k][i, j-125:j+125]
                    LFP_all[ik, :]+= lfp_d[i, j-250:j+250]

    ## FFT: 
    n_out = np.tile(ACH_out[:, 125], [250, 1]).T
    ACH_out[:, 125] = 0
    ACH_out = ACH_out / n_out

    n_in = np.tile(ACH_in[:, 125], [250, 1]).T
    ACH_in[:, 125] = 0
    ACH_in = ACH_in / n_in

    n_all = np.tile(ACH_all[:, 125], [250, 1]).T
    ACH_all[:, 125] = 0
    ACH_all = ACH_all / n_all

    LFP_all = LFP_all - np.tile(np.mean(LFP_all, axis=1), [250, 1]).T

    n_un = len(ord_un.keys())
    PSD_in = np.zeros((n_un, 126))
    PSD_out = np.zeros((n_un, 126))
    PSD_all = np.zeros((n_un, 129))
    Fs = 1000
    for i in range(n_un):
        f, PSD_in[i,:] = scipy.signal.welch(ACH_in[i,:], fs=Fs)
        f, PSD_out[i,:] = scipy.signal.welch(ACH_out[i,:], fs=Fs)
        f, PSD_all[i, :] = scipy.signal.welch(LFP_all[i, :], fs=Fs)

    #Find trials w/ PSD_in > PSD_out
    plt.pcolormesh(np.arange(n_un),f, PSD_in.T-PSD_out.T, vmin=0,vmax=5e-7)
    f_ix = np.nonzero(np.logical_and(f>=20, f<=45))[0]
    diff = PSD_in - PSD_out

    cnt_n_un = np.nonzero(n_in[:,0]>=100)[0]

    ix_ = np.ix_(cnt_n_un, f_ix)
    thresh = np.percentile(diff[ix_].reshape(-1), 95)
    ix1, ix2 = np.nonzero(diff[ix_] > thresh)
    mod_un = cnt_n_un[np.unique(ix1)]
    
#def spk_trig_LFP(env0, lfp_d):




            







