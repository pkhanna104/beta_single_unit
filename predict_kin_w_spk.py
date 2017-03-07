
''' Predict kinematic signal using a history of spikes from MC (022715a)

   Then, apply filter to Beta BMI data, see if it predicts the slow movement onset

Learnings so far: 
- Factor 1 seems to capture the movement onset state
- Factor does not seem to care about pre-move state
- What about the BMI task makes movement onset slow? 
    - Is there something the beta is doing to the spikes? 
    - Or could it be an attentional process that isn't able to switch tasks as quickly?

- Beta states (on vs. off) are not very different in factor weightings
- Seems like off-beta FR are higher than on-beta
    - could be data bias --> many 'non-beta' events, few 'beta events'

    '''

#Re-extract kin signals: 
import tables
import numpy as np
import scipy.signal
import scipy.stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import state_space_spks as sss
import state_space_cart as sssc
import state_space_w_beta_bursts as ssbb
import load_files
import pickle
import sklearn.lda

from riglib.bmi import train
from riglib.bmi import state_space_models
from riglib.bmi import ppfdecoder, kfdecoder

from sklearn.lda import LDA
from sklearn import svm

import gc
import multiprocessing as mp
import datetime
import un2key
import re
import scipy.io as sio
import scipy.signal
from utils import sav_gol_filt as sg_filt
from scipy import ndimage
#import fcns

import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5,style='whitegrid')

cmap = ['green', 'royalblue', 'gold', 'orangered', 'black']
bp_filt = ssbb.master_beta_filt
# hdf_bcd = tables.openFile('/Volumes/TimeMachineBackups/v2_Beta_pap_rev_data/2016-05-09pap_rev_grom_behav_t1lfp_mod_mc_reach_out.h5')
# ix = hdf_bcd.root.behav[:]['task_entry']
# ix2 = np.array([i for i,j in enumerate(ix) if j[:6]=='022715'])
# kin_sig = hdf_bcd.root.kin[ix2]['kin_sig']

# cmap = ['darkcyan', 'royalblue', 'gold', 'orangered']

def get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize, smooth=-1, animal='grom'):
    signal_type = 'shenoy_jt_vel' #Used to get full endpt hand pos and hand vel: 
    kin_signal = dict()
    binned_kin_signal = dict()
    bin_kin_signal = dict()
    binned_rt = dict()

    sz =  int((bef+aft)*1000/float(binsize))

    for i_d, day in enumerate(days):
        for ib, b in enumerate(blocks[i_d]):

            go_cue_times_lfp = go_times_dict[b, day][1]
            labs = lfp_lab[b, day].copy()
            labs[labs > 80] = 64 #All LFP labels are to MC target 64

            #Cursor trajectory: 
            if animal == 'grom':
                from utils import psycho_metrics as pm
                from utils import spectral_metrics as sm
                fname = load_files.load(b, day, return_name_only=True)
                kw = dict(target_coding='standard', fnames = [fname], all_kins=True)

                try:
                    
                    #kin_sig is signal2 in sm.get_sig and is [ xpos, ypos, xvel, yvel, speed] and is metric x trials x time
                    proj_vel, kin_sig, targ_dir = sm.get_sig([day], [[b]], go_cue_times_lfp, [len(go_cue_times_lfp)], signal_type=signal_type, mc_lab = labs, prep=True, anim = 'grom', **kw)
                    kin_feats = pm.kin_feat = pm.get_kin_sig_shenoy(proj_vel) #Kin 
                    rt = kin_feats[:, 2] #1500 is Go cue

                except:
                    print 'No AD33 key in ', day, b

            elif animal == 'cart':
                #Get movement onset times for CART: 

                kin_sig, rt = get_cart_kin(b, day, go_cue_times_lfp, bef, aft)

            #Smooth signal: 
            if smooth > 0:  
                window = scipy.signal.gaussian(151, std=smooth)
                window = window/np.sum(window)

                tmp = []
                if signal_type=='speed':
                    for i in range(len(labs)): tmp.append(np.convolve(window, kin_sig[i,:], mode='same'))
                    Y = np.vstack((tmp))
                    Y = Y[:, :, np.newaxis]
                elif signal_type == 'shenoy_jt_vel':

                    for j in [2, 3]: #Xvel, Yvel
                        tmp2 = []
                        for i in range(len(labs)): tmp2.append(np.convolve(window, kin_sig[j, i,:], mode='same'))
                        tmp.append(tmp2)
                    y = np.dstack((tmp))
                    Y = np.zeros((y.shape[0], y.shape[1], 4))
                    Y[:, :, 2:] = y
                    Y[:, :, :2] = np.swapaxes(kin_sig[:2, :, :].T, 0, 1)

            else:
                Y = kin_sig.copy()

            #Bin signal
            Y_tmp = []

            #Assumed bef == 1.5 (this is what sm.get_sig assumes)
            if bef == 1.5:
                for s in range(sz):
                    Y_tmp.append(np.mean(Y[:, s*binsize:(s+1)*binsize, :], axis=1))
                if signal_type in ['endpt', 'shenoy_jt_vel']:
                    Y_down = np.dstack((Y_tmp))
                else:
                    Y_down = np.vstack((Y_tmp)).T

                kin_signal[b, day] = kin_sig
                binned_kin_signal[b, day] = Y_down

                t_bins = np.arange(0, 2501, binsize)
                #Choose bin right after RT: 
                rt_bins = []
                for r in rt:
                    _tmp = np.nonzero(t_bins>=r)[0]
                    if len(_tmp) > 0:
                        _i = _tmp[0]
                    else:
                        _i = len(t_bins) - 1
                    rt_bins.append(_i)
                binned_rt[b, day] = rt_bins

                if signal_type in ['endpt', 'shenoy_jt_vel']:
                    spd = np.sqrt(np.sum(Y_down[:, [2, 3], :]**2, axis=1))
                    bin_kin_signal[b, day] = np.zeros_like(spd)
                    if animal == 'grom':
                        bin_kin_signal[b, day][spd >= (3.5/100.)] = 1
                    elif animal == 'cart':
                        #Already in CM / sec. boom. 
                        bin_kin_signal[b, day][spd >= (3.5)] = 1
            else:
                raise Exception

    return kin_signal, binned_kin_signal, bin_kin_signal, binned_rt

def get_cart_kin(block, day, go_cue_times_lfp, bef, aft):
    ''' Must return kin_sig, rt where kin_sig is a 5 x nbins matrix'''

    #from utils import sav_gol_filt as sg_filt
    from utils import psycho_metrics as pm

    t = load_files.load(block, day, animal='cart', include_hdfstuff=True)
    hdf = t['hdf']
    ts_func = t['ts_func']

    #Transform LFP marker into HDF rows: 
    go_cue_times_lfp_hdf_ix = ts_func(go_cue_times_lfp, 'hdf')

    KIN_SIG = []
    RT = []

    #Must resample data to be 1000 Hz
    for ig, go_ in enumerate(go_cue_times_lfp_hdf_ix):
        hdf_ix = np.arange(go_ - 90, go_+60)
        lfp_ix = (1000*ts_func(hdf_ix, 'plx')).astype(int)
        lfp_ix = lfp_ix - lfp_ix[0]
        cursor_init = np.zeros((2500, 2))
        for i, hdfi in enumerate(hdf_ix):
            if i == len(lfp_ix)-1:
                cursor_init[lfp_ix[i]:] = hdf.root.task[hdfi]['cursor'][[0, 2]]
            else:
                cursor_init[lfp_ix[i]:lfp_ix[i+1],:] = hdf.root.task[hdfi]['cursor'][[0, 2]]


        #In cm
        cursor2 = hdf.root.task[int(go_-((bef)*60)):int(go_+((aft)*60.))]['cursor'][:,[0,2]]
       
        #In cm / sec
        vel = np.diff(cursor2,axis=0)/(1./60.)
        vel_init = np.zeros((2500, 2))
        for i, hdfi in enumerate(hdf_ix):
            if i == len(lfp_ix)-1:
                vel_init[lfp_ix[i]:] = vel[i-1,:]
            else:
                vel_init[lfp_ix[i]:lfp_ix[i+1],:] = vel[i,:]


        # filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)

        #Get RT - old fashioned way: 
        curs_b = np.arange(-1500, 1000, 1000./60.)
        vel = np.diff(cursor2, axis=0)/(1000./60.)
        filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
        vel_bins = curs_b[:-1] + .5*(curs_b[1]-curs_b[0])
        mc_vect = np.array([1, 0])
        mc_vect_mat = np.tile(mc_vect[np.newaxis, :], [filt_vel.shape[0], 1])
        proj_vel = np.sum(np.multiply(mc_vect_mat, filt_vel), axis=1)
        start_bin  = 89;

        kin_feat = pm.get_kin_sig_shenoy(proj_vel[np.newaxis], bins=vel_bins, start_bin=start_bin, 
            first_local_max_method=True, after_start_est=200, kin_est=700)

        RT.append(kin_feat[0, 2])

        #Get Speed:
        spd = np.sqrt(vel_init[:,0]**2 + vel_init[:, 1]**2)
        sub_kin_sig = np.vstack((cursor_init.T, vel_init.T, spd[np.newaxis, :]))

        KIN_SIG.append(sub_kin_sig)

    kin_sig = np.dstack((KIN_SIG))
    rt = np.hstack((RT))
    #Yield metric x trial x time
    return np.swapaxes(kin_sig, 1, 2), rt

class ssm_spd(state_space_models.StateSpaceEndptVel2D):
    def __init__(self):
        super(self, ssm.StateSpaceEndptVel2D).__init__()
        self.drives_obs = np.array([True, True])
        self.drives_obs_inds = [0, 1]
        self.state_order = np.array([1., np.nan])
        self.is_stochastic = np.array([True, False])
        ssm.train_inds = np.array([0, 1])

def get_kf_trained_from_full_mc(keep_dict, days, blocks, mc_indicator, decoder_only=False, kin_type='endpt', animal='grom', binsize=25, include_speed=False):

    ''' Goal is to use entire MC file (not just trial epochs) to train decoder'''

    R2_jt = {}
    R2_lpf_jt = {}
    R2_lpf_spd_jt = {}
    R2_lpf_beta = {}
    Pred_Kin = {}
    Full_Kin = {}
    Pred_Kin_lpf = {}
    
    decoder_dict = {}
    print animal
    if animal == 'grom':
        spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = get_full_blocks(keep_dict, days, blocks, mc_indicator, kin_type=kin_type)
    elif animal == 'cart':
        spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = get_full_blocks_cart(keep_dict, days, blocks, mc_indicator, kin_type=kin_type)


    mc_blocks = []
    for i_d, d in enumerate(days):
        ix = np.array([m.start() for m in re.finditer('1', mc_indicator[i_d])])
        str_ = ''
        for i in ix: str_ = str_+blocks[i_d][i]
        mc_blocks.append(str_)

    #Smooth stuff: 
    window = scipy.signal.gaussian(151, std=50)
    window = window/np.sum(window)
    smooth_kin_dict = {}
    for k in kin_dict.keys():
        smooth_kin_dict[k] = np.zeros_like(kin_dict[k])
        for j in range(4):
            smooth_kin_dict[k][:, j] = np.convolve(window, kin_dict[k][:,j], mode='same')



    #Bin stuff: 
    Bin_K = {}
    Bin_N = {}
    Bin_B = {}
    Bin_H = {}
    for k in spk_dict.keys():
        Bin_N[k] = {}
        for u in spk_dict[k].keys():
            Bin_N[k][u] = bin_(spk_dict[k][u], binsize, mode='cnts')
        Bin_K[k] = bin_(smooth_kin_dict[k], binsize, mode='mean')
    
        if decoder_only is False:
            Bin_B[k] = bin_(beta_dict[k][1,:], binsize, mode='mode')
            #Bin_H[k] = bin_(hold_dict[k], binsize, mode='mode')

    lpf_window = scipy.signal.gaussian(401, std=151)
    t_ix = np.arange(0, 401, binsize)
    lpf_window = lpf_window[t_ix]
    lpf_window = lpf_window/np.sum(lpf_window)

    #Train MC decoder: 
    for k in Bin_K.keys():

        mat_Bin_N_tmp = []
        for u in Bin_N[k]:
            mat_Bin_N_tmp.append(Bin_N[k][u])
        obs = np.hstack((mat_Bin_N_tmp))
        kin = Bin_K[k]
        z = np.zeros((len(kin), ))
        
        kin_vel = kin[:, [2,3]]
        
        
        if include_speed:
            spd = np.sum(kin[:, [2,3]]**2, axis=1)
            kin_full = np.vstack((kin[:, 0], z, kin[:, 1], kin[:, 2], spd, kin[:,3])).T
            ssm = state_space_models.StateSpaceEndptVel3D()
            kin_vel = np.hstack((kin[:, [2]], spd[:, np.newaxis], kin[:, [3]]))
        else:
            ssm = state_space_models.StateSpaceEndptVel2D()
            kin_full = np.vstack((kin[:, 0], z, kin[:, 1], kin[:, 2], z, kin[:,3])).T
        nbins, n_units = obs.shape
        

        A, B, W = ssm.get_ssm_matrices(update_rate=1./binsize)
        if animal == 'cart':
            W *= 10
        # #Actually calculated A, W: 
        # x_t_minus_1 = kin_full[:-1, [2, 3]] #t x 2
        # x_t_0 = kin_full[1:, [2, 3]] # t x 2

        # x_t_minus_1 = kin_full[:-1, [3, 5]] #t x 2
        # x_t_0 = kin_full[1:, [3, 5]] # t x 2
        # sub_A = np.zeros((2, 2))
        # sub_A_d = np.mean(np.diag(np.linalg.lstsq(x_t_minus_1, x_t_0)[0]))
        # sub_A[0,0] = sub_A_d
        # sub_A[1,1] = sub_A_d

        # A[3,3] = sub_A_d
        # A[5,5] = sub_A_d

        # sub_W = np.cov(x_t_0.T - np.mat(sub_A)*x_t_minus_1.T)
        # W[3,3] = np.max(np.diag(sub_W))
        # W[5,5] = np.max(np.diag(sub_W))

        C = np.zeros([n_units, ssm.n_states])

        for model in ['kf']:#['ppf', 'kf']:
            print 'Starting model: ', model, ' for key: ', k, ' for binsize: ',binsize
            if model == 'kf':
                C[:, ssm.drives_obs_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin_vel.T, obs.T, include_offset=True)
            
                # instantiate KFdecoder
                filter_ = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=ssm.is_stochastic)
                filter_._init_state()

            elif model == 'ppf':
                obs_ppf = obs.copy()
                obs_ppf[obs_ppf > 1] = 1

                C[:, ssm.drives_obs_inds], pvals = ppfdecoder.PointProcessFilter.MLE_obs_model(kin_vel.T, obs_ppf.T, include_offset=True)
                
                filter_ = ppfdecoder.PointProcessFilter(A, W, C, B=B, dt=1./binsize, is_stochastic=ssm.is_stochastic)
                filter_._init_state()

            decoder_dict[str(binsize), k[1], k[0], model] = filter_
            print 'adding decoder dict: ', str(binsize), k[1], k[0], model

            if decoder_only is False:
                #Get training: 
                s = filter_.state
                nbins, n_ft = kin.shape

                st = []

                #Cycle through bins
                for t in range(nbins):
                    obs_ = obs[t, :][:, np.newaxis]
                    if model == 'ppf':
                        obs_[obs_>1] = 1

                    s = filter_._forward_infer(s, obs_)
                    st.append(s.mean)
                pred_kin = np.array(np.hstack((st))).T

                # f, ax = plt.subplots(nrows = 2)
                # ax[0].plot(pred_kin[:, 3], label='pred x vel')
                # ax[0].plot(kin_full[:, 3], label='act x vel')
                # ax[0].legend()
                # ax[0].set_title('Binsize: '+str(binsize)+' Model: '+model+' Key: '+k[1]+k[0])
                # ax[1].plot(pred_kin[:, 5], label='pred y vel')
                # ax[1].plot(kin_full[:, 5], label='act x vel')
                # ax[1].legend()

                #Calc R2 after 1 minute in to -1 min out: 
                n_samp_buffer = int(60*1000./binsize)
                y_hat = pred_kin[n_samp_buffer:-n_samp_buffer, [3, 5]]
                y = kin_full[n_samp_buffer:-n_samp_buffer, [3, 5]]
                beta_sig = Bin_B[k][0, n_samp_buffer:-n_samp_buffer]
                kin_sig = Bin_K[k][n_samp_buffer:-n_samp_buffer, [2, 3]]
                
                binary_kin_sig = np.array([int(np.logical_or(np.abs(kin_sig[i, 0])>0.5, np.abs(kin_sig[i, 1])>0.5)) for i in range(len(kin_sig))])

                y_mean = np.tile(np.mean(y, axis=0)[np.newaxis, :], [y.shape[0], 1])
                R2_ = 1 - (np.sum((y - y_hat)**2)/np.sum((y - y_mean)**2))

                R2_jt[binsize, k[1], k[0], model] = R2_
                #Pred_Kin[binsize, k[1], k[0], model] = y_hat.copy()
                #Full_Kin[binsize, k[1], k[0], model] = y.copy()
                lpf_x_vel = np.convolve(lpf_window, y_hat[:, 0], mode='same')
                lpf_y_vel = np.convolve(lpf_window, y_hat[:, 1], mode='same')
                
                y_hat_lpf = np.vstack((lpf_x_vel, lpf_y_vel)).T

                R2_lpf_ = 1 - (np.sum((y - y_hat_lpf)**2)/np.sum((y - y_mean)**2))

                f, ax = plt.subplots(nrows = 2)
                ax[0].plot(y_hat_lpf[:, 0], label='pred x vel')
                ax[0].plot(y[:, 0], label='act x vel')
                ax[0].plot(20+ (2*beta_sig), 'r-')
                ax[0].legend()
                ax[0].set_title('Binsize: '+str(binsize)+' Model: '+model+' Key: '+k[1]+k[0])
                ax[1].plot(y_hat_lpf[:, 1], label='pred y vel')
                ax[1].plot(y[:, 1], label='act x vel')
                ax[1].plot(20+ (2*beta_sig), 'r-')
                ax[1].legend()

                Pred_Kin_lpf[binsize, k[1], k[0], model] = y_hat_lpf
                R2_lpf_jt[binsize, k[1], k[0], model] = R2_lpf_

                lpf_spd = np.sqrt(lpf_x_vel**2 + lpf_y_vel**2)
                act_spd = np.sqrt(y[:,0]**2 + y[:,1]**2)
                R2_lpf_spd_ = 1 - (np.sum((act_spd - lpf_spd)**2)/np.sum((act_spd - np.mean(act_spd))**2))
                R2_lpf_spd_jt[binsize, k[1], k[0], model] = R2_lpf_spd_

                hold_periods = Bin_H[k][0, n_samp_buffer:-n_samp_buffer]

                hold_ix = np.nonzero(binary_kin_sig==0)[0]
                beta_hold = beta_sig[hold_ix]

                #y_hold = y.copy()
                #y_pred = y_hat_lpf.copy()

                y_hold = y[hold_ix, :]
                y_pred = y_hat_lpf[hold_ix, :]

                for beta_ix_ in [0, 1]:
                    ix = np.nonzero(beta_hold==beta_ix_)[0]
                    print 'beta ix: ', len(ix)
                    
                    for vel in [0, 1]:
                        r2_ = 1 - (np.sum((np.abs(y_pred[ix, vel]) - np.abs(y_hold[ix, vel]))**2)/(np.sum((np.abs(y_hold[ix, vel])- np.abs(np.mean(y_hold[ix, vel])))**2)))
                        R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel] = r2_

                        MN = np.mean(np.abs(y_pred[ix, vel]) - np.abs(y_hold[ix, vel]))
                        print 'beta ix: ', beta_ix_, ' vel: ', vel, MN

                        R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel, 'mean_abs_err'] = MN
                        R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel, 'y_pred'] = y_pred[ix, vel]
                        R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel, 'y_act'] = y_hold[ix, vel]
                R2_lpf_beta[binsize, k[1], k[0], model, 'full_R2_lpf'] = R2_lpf_
                R2_lpf_beta[binsize, k[1], k[0], model, 'y_hat_lpf'] = y_hat_lpf
                R2_lpf_beta[binsize, k[1], k[0], model, 'y'] = y
    if decoder_only is False:
        import pickle
        # pickle.dump(R2, open('mc_fit_R2.pkl', 'wb'))
        # pickle.dump(R2_lpf, open('mc_fit_R2_lpf.pkl', 'wb'))
        # pickle.dump(R2_lpf_spd, open('mc_fit_R2_lpf_spd.pkl', 'wb'))
        pickle.dump(R2_lpf_beta, open('mc_fit_R2_lpf_beta_hold_v4_endpt_bc_v3_was_jts.pkl', 'wb'))
    else:
        return decoder_dict

def bin_(x, binsize, mode='cnts'):
    ''' X must be time x number of features'''
    if len(x.shape) > 1:
        nf = np.min(x.shape)
    else:
        nf = 1
        x = x[:, np.newaxis]

    n_b = x.shape[0]/binsize
    x_binned = []
    for n in range(nf):
        x_ = []
        for bi in range(n_b-1):
            sub = x[(bi*binsize):((bi+1)*binsize), n]
            if mode == 'cnts':
                _tmp = np.sum(sub)
            elif mode == 'mean':
                _tmp = np.mean(sub)
            elif mode == 'mode':
                _tmp, nmb = scipy.stats.mstats.mode(sub)
            else:
                raise Exception
            x_.append(_tmp)
        x_binned.append(x_)
    return np.vstack((x_binned)).T

def get_full_blocks_cart(keep_dict, days, blocks, mc_indicator, mc_only=True, nf_only = False, kin_type='endpt', decoder=None):
    #Make sure same number of blocks / days etc.
    assert len(days) == len(blocks) == len(keep_dict.keys())    
    spk_dict = {} #Spikes dict
    lfp_dict = {} #LFP dict
    kin_dict = {}
    beta_dict = {}
    hold_dict = {}

    for i_d, day in enumerate(days):
        if mc_only == True:
            ix = np.array([m.start() for m in re.finditer('1', mc_indicator[i_d])])
        elif nf_only == True:
            ix = np.array([m.start() for m in re.finditer('0', mc_indicator[i_d])])
        else:
            ix = np.arange(len(mc_indicator[i_d]))

        units = ['a', 'b', 'c', 'd']
        cell_list = keep_dict[day]

        #For each block:
        for i_b in ix:
            loaded = False
            t = load_files.load(blocks[i_d][i_b], day, animal='cart', include_hdfstuff=True)

            if t is not None:

                spk = t #Copy loaded file to a more attractive name ;) 
                keys = spk.keys()
                spk_keys = cell_list

                #Get time delays 
                lfp_offset = 0.

                hdf = spk['hdf']
                hdf_lims = np.array([0, len(hdf.root.task)-1])
                ts_func = spk['ts_func']
                plx_lims = ts_func(hdf_lims, 'plx') 

                cursor_init = np.zeros((int(plx_lims[-1]*1000) - int(plx_lims[0]*1000), 2))
                hdf_ts = (1000*ts_func(np.arange(0, len(hdf.root.task)), 'plx')).astype(int)
                hdf_ts = hdf_ts - hdf_ts[0]

                #Make sure no strobed events occur before:
                #assert np.all(strobed[:,0] - lfp_offset > 0.)

                #Make sure length of AD channel is > last event - offset: 
                #assert (len(t[ts_key[:-3]])/1000.) > (strobed[-1, 0] - lfp_offset)



                spk_dict[blocks[i_d][i_b], day] = dict()

                for un in spk_keys:
                        
                    if np.any(np.array(spk[un].shape) > 1):
                        ts_arr = np.squeeze(spk[un])
                    else:
                        ts_arr = spk[un]

                    #good_ts_arr = np.nonzero(np.logical_and(ts_arr>plx_lims[0], ts_arr<plx_lims[1]))

                    #Spikes
                    spk_dict[blocks[i_d][i_b], day][un] = bin1ms_full(ts_arr, plx_lims[0], plx_lims[1])

                #LFPs
                lfp_dict[blocks[i_d][i_b], day] = spk['ad124'][int(plx_lims[0]*1000):int(plx_lims[1]*1000)]

                perc_beta=60
                min_beta_burst_len = 125

                nyq = 0.5* 1000
                bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
                data_filt = scipy.signal.filtfilt(bw_b, bw_a, lfp_dict[blocks[i_d][i_b], day])
                

                ### scipy.signal.hilbert stupidly doesn't pad zeros to get to nfft ##
                nfft = 1<<(len(data_filt)-1).bit_length()
                data_filt_pad = np.zeros(nfft)
                data_filt_pad[:len(data_filt)] = data_filt

                sig_pad = np.abs(scipy.signal.hilbert(data_filt_pad))
                sig = sig_pad[:len(data_filt)]
                sig_bin = np.zeros_like(sig)
                sig_bin[sig > np.percentile(sig.reshape(-1), perc_beta)] = 1

                #Get only blobs >= 50 ms: 
                #see http://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array
                sig_bin_filt = np.zeros((3, len(sig_bin)))
                sig_bin_filt[1,:] = sig_bin.copy()
                struct = np.zeros((3,3))
                struct[1,:] = 1 #Get patterns that only are horizontal
                id_regions, num_ids = ndimage.label(sig_bin_filt, structure=struct)
                id_sizes = np.array(ndimage.sum(sig_bin, id_regions, range(num_ids + 1)))
                area_mask = (id_sizes <= min_beta_burst_len )
                sig_bin_filt[area_mask[id_regions]] = 0
                beta_dict[blocks[i_d][i_b], day] = sig_bin_filt

                #Get cart kin: 
                #In cm
                cursor = hdf.root.task[:]['cursor'][:,[0,2]]

                assert hdf_ts.shape[0] == cursor.shape[0]

                for it, ts in enumerate(hdf_ts[:-1]): cursor_init[ts:hdf_ts[it+1], :] = cursor[it,:]
                c_filt = cursor_init
                #c_filt = scipy.signal.resample_poly(cursor, 1000, 60, axis=0)
                xpos = c_filt[:, 0]
                ypos = c_filt[:, 1]
                #In cm / sec
                vel = np.diff(cursor,axis=0)/(1./60.)
                vel_init = np.zeros_like(cursor_init)
                for it, ts in enumerate(hdf_ts[:-1]): vel_init[ts:hdf_ts[it+1], :] = vel[it,:]
                filt_vel = sg_filt.savgol_filter(vel_init, 81, 5, axis=0)
                #v_filt = scipy.signal.resample_poly(filt_vel, 1000, 60, axis=0)
                xvel = filt_vel[:, 0]
                yvel = filt_vel[:, 1]

                kin_dict[blocks[i_d][i_b], day] = np.hstack((xpos[:, np.newaxis], ypos[:, np.newaxis], 
                    xvel[:, np.newaxis], yvel[:, np.newaxis]))
                
                hold_dict = {}
    return spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict

def get_full_blocks(keep_dict, days, blocks, mc_indicator, mc_only=True, kin_type='endpt', decoder=None):
    #Make sure same number of blocks / days etc.
    assert len(days) == len(blocks) == len(keep_dict.keys())

    spk_dict = {} #Spikes dict
    lfp_dict = {} #LFP dict
    kin_dict = {}
    beta_dict = {}
    hold_dict = {}
    import kinarm
    kinarm_params = sio.loadmat('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/KinematicsParameters_seba.mat')
    kinarm_calib = kinarm.calib(kinarm_params['sho_pos_x'][0,0], kinarm_params['sho_pos_y'][0,0], 
        kinarm_params['L1'][0,0], kinarm_params['L2'][0,0], kinarm_params['L2ptr'][0,0])

    for i_d, day in enumerate(days):
        if mc_only == True:
            ix = np.array([m.start() for m in re.finditer('1', mc_indicator[i_d])])
        else:
            ix = np.arange(len(mc_indicator[i_d]))

        units = ['a', 'b', 'c', 'd']
        cell_list = keep_dict[day]

        for i_b in ix:
            loaded = False
            t = load_files.load(blocks[i_d][i_b], day)
            if t is not None:
                #Get time delays 
                strobed = t['Strobed']

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

                spk = t #Copy loaded file to a more attractive name ;) 
                keys = spk.keys()
                spk_keys = cell_list

                spk_dict[blocks[i_d][i_b], day] = dict()

                for un in spk_keys:
                    if un[-2:] == 'wf':
                        #Trim key down: 
                        un = un[:-3]

                    #Add correct number of zeros
                    un = un2key.convert(un)
                        
                    if np.any(np.array(spk[un].shape) > 1):
                        ts_arr = np.squeeze(spk[un])
                    else:
                        ts_arr = spk[un]

                    #Spikes
                    spk_dict[blocks[i_d][i_b], day][un] = bin1ms_full(ts_arr, lfp_offset, lfp_offset+(len(spk['AD74'])/1000.))

                #LFPs
                lfp_dict[blocks[i_d][i_b], day] = spk['AD74'][:, 0]
                perc_beta=60
                min_beta_burst_len = 100

                nyq = 0.5* 1000
                bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
                data_filt = scipy.signal.filtfilt(bw_b, bw_a, lfp_dict[blocks[i_d][i_b], day])
                
                sig = np.abs(scipy.signal.hilbert(data_filt, N=None))
                sig_bin = np.zeros_like(sig)
                sig_bin[sig > np.percentile(sig.reshape(-1), perc_beta)] = 1

                #Get only blobs >= 50 ms: 
                #see http://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array
                sig_bin_filt = np.zeros((3, len(sig_bin)))
                sig_bin_filt[1,:] = sig_bin.copy()
                struct = np.zeros((3,3))
                struct[1,:] = 1 #Get patterns that only are horizontal
                id_regions, num_ids = ndimage.label(sig_bin_filt, structure=struct)
                id_sizes = np.array(ndimage.sum(sig_bin, id_regions, range(num_ids + 1)))
                area_mask = (id_sizes <= min_beta_burst_len )
                sig_bin_filt[area_mask[id_regions]] = 0
                beta_dict[blocks[i_d][i_b], day] = sig_bin_filt

                if kin_type=='endpt':
                    xpos,ypos,xvel,yvel = kinarm.calc_endpt(kinarm_calib,spk['AD33'][:, 0], spk['AD34'][:, 0], 
                        sh_vel = spk['AD35'][:, 0], el_vel = spk['AD36'][:, 0])
                    xpos *= 100
                    ypos *= 100
                    xvel *= 100
                    yvel *= 100
                elif kin_type=='jts':
                    xpos = spk['AD33'][:, 0]
                    ypos = spk['AD34'][:, 0]
                    xvel = spk['AD35'][:, 0]
                    yvel = spk['AD36'][:, 0]

                # Hold dict: 
                rew_ix = np.nonzero(strobed[:,1]==9)[0]
                st_hold1 = strobed[rew_ix - 5, 0] - lfp_offset
                end_hold1 = strobed[rew_ix - 4, 0] - lfp_offset
                
                st_hold2 = strobed[rew_ix-2, 0] - lfp_offset
                end_hold2 = strobed[rew_ix-1, 0] - lfp_offset

                hld = np.zeros((len(spk['AD74']), ))
                for hset in [zip(st_hold1, end_hold1), zip(st_hold2, end_hold2)]:
                    for i, (s, e) in enumerate(hset):
                        hld[int(s*1000):int(e*1000)] = 1

                hold_dict[blocks[i_d][i_b], day] = hld

                kin_dict[blocks[i_d][i_b], day] = np.hstack((xpos[:, np.newaxis], ypos[:, np.newaxis], 
                    xvel[:, np.newaxis], yvel[:, np.newaxis]))

    return spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict

def bin1ms_full(ts_arr, start, stop):
    tmp = np.zeros(( (stop-start)*1000., ))
    ix = np.nonzero(np.logical_and(ts_arr < stop, ts_arr>= start))[0]
    if len(ix) > 0:
        ts_recenter = (ts_arr[ix] - start)
        ts_bins = np.round(ts_recenter*1000).astype(int) 
        for t in ts_bins:
            if t<len(tmp):
                tmp[t] += 1
            else:
                print 'rounding means skipping: ', t, ' on full trial'
    return tmp

def get_unbinned(test=False, days=None, animal='grom', all_cells=False, blocks=None, mc_indicator=None):
    #Get good cells (ensure they're all in all blocks)
    if animal=='grom':
        keep_dict, days, blocks, mc_indicator = sss.get_cells(plot=False, test=test, days=days)
    
    elif animal == 'cart':
        keep_dict, days, blocks, mc_indicator = sssc.get_cells(plot=False, test=test, days=days, 
            blocks = blocks, mc_indicator=mc_indicator, all_cells=all_cells, only3478=False)
    
    #Get spike / lfp / beta dictionaries with data
    spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict = ssbb.get_beta_bursts(keep_dict, 
        days, blocks, mc_indicator, animal=animal)

    return keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator

def test_beta_binary(lfp_dict, beta_dict, beta_cont_dict):
    k = lfp_dict.keys()
    k = k[0]
    k2 = tuple(np.hstack((k,'filt')))

    ntrials = lfp_dict[k].shape[0]
    ix = np.random.permutation(ntrials)
    for j in ix[:5]:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(lfp_dict[k][j, :], 'k-', label='Raw')
        ax.plot(beta_cont_dict[k][j, :], 'b-', label='Pwr')
        ax.plot(beta_cont_dict[k2][j, :], 'g-', label='filt')
        ax.plot(beta_dict[k][j, :]/10., 'r-', label='Events')
        plt.legend(fontsize='x-small')
        plt.show()

def main_process(kin_signal_dict, binned_kin_signal_dict, B_bin, BC_bin, S_bin, Params, days, blocks, mc_indicator, lfp_lab, 
    binsize, train_trial_type='mc', test_trial_type='beta', spike_model='gaussian', predict='kin'):

    # #Get good cells (ensure they're all in all blocks)
    # keep_dict, days, blocks, mc_indicator = sss.get_cells()

    # #Get spike / lfp / beta dictionaries with data
    # spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, bef, aft, go_times_dict = ssbb.get_beta_bursts(keep_dict, 
    #     days, blocks, mc_indicator)

    # #Bin / smooth spike and beta dictionaries
    # B_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, bef, aft, 
    # smooth=-1, binsize=binsize)

    # #Get kin signal
    # kin_signal_dict, binned_kin_signal_dict = get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize)

    #Now correlate kin signals and neural signals on a day-by-day basis: 
    for i_d, d in enumerate(days):
        day_spks = S_bin[d]
        day_bc = np.dstack((BC_bin[d], BC_bin[d, 'phz']))
        day_bc[:, :, 0] = np.log10(day_bc[:, :, 0])

        day_b = B_bin[d]
        day_kin = []
        day_trial_type = []
        day_lfp_lab = []


        for ib, b in enumerate(blocks[i_d]):
            if predict== 'kin' or predict == 'kin_and_beta':
                if np.logical_and(len(binned_kin_signal_dict[b, d].shape) > 2, binned_kin_signal_dict[b, d].shape[1] < binned_kin_signal_dict[b, d].shape[2]):
                        day_kin.append(np.swapaxes(binned_kin_signal_dict[b, d], 1, 2))
                else:
                    day_kin.append(binned_kin_signal_dict[b, d])
 
            n_trials = len(lfp_lab[b, d])

            if int(mc_indicator[i_d][ib]) == 1:
                tt = ['mc']*n_trials
            elif int(mc_indicator[i_d][ib]) == 0:
                tt = ['beta']*n_trials
            else:
                print 'Error! Mc indicator [i_d][ib] is not a 1 or 0: i_d: ', i_d, d, 'i_b: ', i_b, b, 'mc[i_d][ib]: ', mc_indicator[i_d][ib]

            day_trial_type.append(tt)
            day_lfp_lab.append(lfp_lab[b, d])

        day_trial_type = np.hstack(day_trial_type)

        if predict == 'kin' or predict == 'kin_and_beta':
            day_kin = np.vstack((day_kin))
        day_lfp_lab = np.hstack((day_lfp_lab))

        ntrials, nbins, nunits = day_spks.shape

        #Training: 
        train_ix = np.nonzero(day_trial_type == train_trial_type)[0]
        test_ix = np.nonzero(day_trial_type == test_trial_type)[0]

        day_resh_spks_train = day_spks[train_ix, :, :].reshape(len(train_ix)*nbins, nunits)
        day_resh_spks_test = day_spks[test_ix, :, :].reshape(len(test_ix)*nbins, nunits)

        day_beta_binary = day_b[train_ix, :].reshape(len(train_ix)*nbins)
        day_beta_binary_test = day_b[test_ix, :].reshape(len(test_ix)*nbins)


        ff = filter_fcns()
        if spike_model == 'ssm_filters':
            filter_arr = ['kf', 'ppf']
        else:
            filter_arr = ['none']

        for filt_name in filter_arr:
            master_metrics = {}

            if predict == 'kin':        
                if len(day_kin.shape) <3 :
                    day_resh_X_train = day_kin[train_ix, :].reshape(len(train_ix)*nbins)
                    day_resh_X_test = day_kin[test_ix, :].reshape(len(test_ix)*nbins)
                    day_X = day_kin[:, :, np.newaxis]
                else:
                    nft = day_kin.shape[2]
                    day_resh_X_train = day_kin[train_ix, :, :].reshape(len(train_ix)*nbins, nft)
                    day_resh_X_test = day_kin[test_ix, :, :].reshape(len(test_ix)*nbins, nft)                
                    day_X = day_kin.copy()    
            elif predict == 'beta':
                day_resh_X_train = day_bc[train_ix, :, 0].reshape(len(train_ix)*nbins, 1)
                day_resh_X_test = day_bc[test_ix, :, 0].reshape(len(test_ix)*nbins, 1)
                day_X = day_bc
            elif predict == 'beta_binary':
                day_resh_X_train = day_b[train_ix, :].reshape(len(train_ix)*nbins)
                day_resh_X_test = day_b[test_ix, :].reshape(len(test_ix)*nbins)
                day_X = day_b
            elif predict == 'kin_and_beta':
                day_resh_X_train = np.hstack((day_bc[train_ix, :, :].reshape(len(train_ix)*nbins, 2), 
                    day_kin[train_ix, :].reshape(len(train_ix)*nbins, 1) ))

                day_resh_X_test = np.hstack((day_bc[test_ix, :, :].reshape(len(test_ix)*nbins, 2), 
                    day_kin[test_ix, :].reshape(len(test_ix)*nbins, 1) ))
            if spike_model == 'gaussian':

                #Linear regression: 
                a, rez, ndim, rcond = np.linalg.lstsq(day_resh_spks_train, day_resh_X_train)
                if len(day_resh_X_train.shape) == 1:
                    n_X_ft = 1
                else:
                    n_X_ft = day_resh_X_train.shape[1]

                if n_X_ft == 1:
                    X_pred_from_testing = np.array((np.mat(day_resh_spks_test)*np.mat(a).T).reshape(len(test_ix), nbins))
                    X_pred_from_training = np.array((np.mat(day_resh_spks_train)*np.mat(a).T).reshape(len(train_ix), nbins))
                else:
                    X_pred_from_testing = np.array((np.mat(day_resh_spks_test)*np.mat(a))).reshape(len(test_ix), nbins, n_X_ft)
                    X_pred_from_training = np.array((np.mat(day_resh_spks_train)*np.mat(a))).reshape(len(train_ix), nbins, n_X_ft)

                # j = 0
                # f, ax = plt.subplots(nrows = 3, ncols=3) 
                # for i in range(j, j+9):
                #     #for k in range(3):
                #     ax[i/3, i%3].plot(day_X[test_ix[i], :], 'g', label='hand speed')
                #     ax[i/3, i%3].plot(X_pred_from_testing[i, :], 'r', label='Gaussian: 10 ms')
                # ax[i/3, i%3].legend()
            elif spike_model == 'ssm_filters':

                # kin_train = day_resh_X_train
                # kin_test = day_resh_X_test
                # spk_train = day_spks[train_ix, :, :]
                # spk_test = day_spks[test_ix, :, :]
                # n_units = day_spks.shape[2]
                fcn_name = filt_name+'_make'
                fcn = getattr(ff, fcn_name)

                #Make sure no infs or np.nans: 
                day_resh_X_train[day_resh_X_train == np.nan] = 0
                day_resh_X_test[day_resh_X_test == np.nan] = 0
                day_resh_X_train[np.abs(day_resh_X_train) == np.inf] = 0
                day_resh_X_test[np.abs(day_resh_X_test) == np.inf] = 0
                
                X_pred_from_training, X_pred_from_testing, n_X_ft = fcn(day_resh_X_train, day_resh_X_test,
                    day_spks[train_ix, :, :], day_spks[test_ix, :, :], binsize, predict, nunits, nbins)

                #X_pred_from_training = np.swapaxes(X_pred_from_training, 1, 2)
                #X_pred_from_testing = np.swapaxes(X_pred_from_testing, 1, 2)
            elif spike_model == 'binary_lda':
                confusion_trn, confusion_test, feature_importance = predict_beta_LDA(day_b[train_ix, :], day_b[test_ix, :], 
                    day_spks[train_ix, :, :], day_spks[test_ix, :, :])
                # Beta_train = day_b[train_ix, :]
                # Beta_test = day_b[test_ix, :]
                # spks_train = day_spks[train_ix, :, :]
                # spks_test = day_spks[test_ix, :, :]

            ### METRICS: MEAN, STD, N1, SEM ###
            if len(X_pred_from_training.shape) < 3:
                X_pred_from_training = X_pred_from_training[:, :, np.newaxis]
                X_pred_from_testing = X_pred_from_testing[:, :, np.newaxis]

            for ft in range(n_X_ft):
                ax, sub_metrics = plot_errs(day_X[train_ix, :, ft], X_pred_from_training[:, :, ft], day_X[test_ix, :, ft], X_pred_from_testing[:, :, ft], 
                    B_bin[d][train_ix, :], B_bin[d][test_ix, :], day_lfp_lab[train_ix], day_lfp_lab[test_ix], plot=False)

                master_metrics[d, binsize, spike_model, ft] = sub_metrics
                master_metrics[d, binsize, spike_model, ft, 'test_pred'] = X_pred_from_testing[:, :, ft] 
                master_metrics[d, binsize, spike_model, ft, 'test_act'] = day_X[test_ix, :, ft]
                master_metrics[d, binsize, spike_model, ft, 'train_pred'] = X_pred_from_training[:, :, ft] 
                master_metrics[d, binsize, spike_model, ft, 'train_act'] = day_X[train_ix, :, ft]

            if predict == 'kin':
                spd_trn_pred = np.sqrt(X_pred_from_training[:, :, 2]**2 + X_pred_from_training[:, :, 3]**2)
                spd_tst_pred = np.sqrt(X_pred_from_testing[:, :, 2]**2 + X_pred_from_testing[:, :, 3]**2)

                spd_trn = np.sqrt(day_X[train_ix, :, 2]**2 + day_X[train_ix, :, 3]**2)
                spd_tst = np.sqrt(day_X[test_ix, :, 2]**2 + day_X[test_ix, :, 3]**2)

                ax, sub_metrics = plot_errs(spd_trn, spd_trn_pred, spd_tst, spd_tst_pred, 
                    B_bin[d][train_ix, :], B_bin[d][test_ix, :], day_lfp_lab[train_ix], day_lfp_lab[test_ix], plot=False)                

                master_metrics[d, binsize, spike_model, 'spd'] = sub_metrics
                master_metrics[d, binsize, spike_model, 'spd', 'test_pred'] = spd_tst_pred
                master_metrics[d, binsize, spike_model, 'spd', 'test_act'] = spd_tst              
                master_metrics[d, binsize, spike_model, 'spd', 'train_pred'] = spd_trn_pred
                master_metrics[d, binsize, spike_model, 'spd', 'train_act'] = spd_trn 
            # conf_train, conf_test = predict_beta(B_bin[d][train_ix, :], B_bin[d][test_ix, :], day_spks[train_ix, :, :],
            #  day_spks[test_ix, :, :])
            # master_metrics[d, binsize] = [conf_train, conf_test]

            #Save parameter metrics: 
            iso = datetime.datetime.now()
            iso_date = iso.isoformat()
            fname = ssbb.master_save_directory2 + 'beta_regressions/'+iso_date[:10]+'_'+d+'_'+str(binsize)+'_'+spike_model+'_predicting_'+predict+'_'+filt_name+'.pkl'
            pickle.dump(master_metrics, open(fname, 'wb'))
            print 'Saved metrics for binsize: '+ str(binsize) + ' and model: ' + spike_model + ' with filter: '+filt_name+ ' while predicting: '+predict+'!'

def psth(train_ix, test_ix, day_spks, Params_day):
    f, ax = plt.subplots(ncols = 5)
    t = np.linspace(-1500, 1000, day_spks.shape[1])
    nunits = day_spks.shape[2]

    mc = np.mean(day_spks[train_ix, :, :], axis=0)
    mx = np.max(mc, axis=0)
    mc = mc / np.tile(mx[:, np.newaxis].T, (mc.shape[0], 1))
    ax[0].pcolormesh(t, np.arange(nunits), mc.T, vmin=0, vmax=1.)
    ax[0].set_title('MC')

    lfp_lab = Params_day['day_lfp_labs'][test_ix]
    for i in range(84, 88):
        axi = ax[i-84+1]
        ix_ = np.nonzero(lfp_lab==i)
        beta = np.mean(day_spks[test_ix[ix_], :, :], axis=0)
        mx = np.max(beta, axis=0)
        beta2 = beta / np.tile(mx[:, np.newaxis].T, (beta.shape[0], 1))
        axi.pcolormesh(t, np.arange(nunits), beta2.T, vmin=0, vmax=1.)
        axi.set_title('Beta Targ: '+str(i-84+1))

def rt_aligned_psth(test, binsize, blocks, days, rt_dict, S_bin, Params, pre=1000, post=500):
    # keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test)
    # B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
    #     beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=binsize)

    rpre = round(pre/binsize)
    rpost = round(post/binsize)

    cmap = fcns.get_cmap()
    f, ax = plt.subplots(nrows=6)

    for i_d, d in enumerate(days):
        
        pre_post_same_diff = np.zeros((2, 2))

        rt_stack = []
        for i_b, b in enumerate(blocks[i_d]):
            rt_stack.append(rt_dict[b, d])
        rt_stack = np.hstack((rt_stack))
        rt_stack_ix = ((rt_stack*1000.)+1500)/binsize
        assert len(rt_stack_ix) == S_bin[d].shape[0]

        ntrials, nbins, nunits = S_bin[d].shape
        Spk_Day = S_bin[d]*binsize

        for iu, un in enumerate(Params[d]['sorted_un']):
            rt_aligned = []            
        
            for t in range(len(rt_stack_ix)):

                r_rt = round(rt_stack_ix[t])
                if r_rt > nbins:
                    tmp = np.zeros((rpre+rpost))+np.nan

                elif r_rt > nbins - rpost:
                    tmp = np.hstack(( Spk_Day[t, :, iu], np.zeros((rpost))+np.nan ))
                    tmp = tmp[r_rt-rpre:r_rt+rpost] 

                elif r_rt < rpre:
                    tmp = Spk_Day[t, :r_rt+rpost, iu]
                    tmp = np.hstack((tmp, np.zeros((rpre - r_rt )) + np.nan))
                else: 
                    tmp = Spk_Day[t, r_rt-rpre:r_rt+rpost, iu]
                rt_aligned.append(tmp)

            rt_aligned = np.vstack((rt_aligned))
            lfp_lab = Params[d]['day_lfp_labs']
            lfp_lab[lfp_lab == 64] = 88

            ix = np.nonzero(lfp_lab<=87)[0]
            ix2 = np.nonzero(lfp_lab==88)[0]

            u1, p1 = scipy.stats.mannwhitneyu(np.sum(rt_aligned[ix, :rpre], axis=1), np.sum(rt_aligned[ix2, :rpre], axis=1))
            u2, p2 = scipy.stats.mannwhitneyu(np.sum(rt_aligned[ix, rpre:(rpre+rpost)], axis=1), np.sum(rt_aligned[ix2, rpre:(rpre+rpost)], axis=1))
            if np.nanmean(rt_aligned) > 1.5/float(binsize):
                if p1 < 0.05:
                    if p2 < 0.05:
                        pre_post_same_diff[1, 1] += 1
                    else:
                        pre_post_same_diff[1, 0] += 1
                else:
                    if p2 <0.05:
                        pre_post_same_diff[0, 1] += 1
                    else:
                        pre_post_same_diff[0, 0] += 1

        print pre_post_same_diff
        tot = float(np.sum(pre_post_same_diff))
        ax[i_d].bar(0, (pre_post_same_diff[0, 0]/tot) + (pre_post_same_diff[1, 0]/tot)) #Same same & Diff Same
        ax[i_d].bar(1, (pre_post_same_diff[1, 1]/tot) + (pre_post_same_diff[0, 1]/tot)) #Diff diff & Same diff

def plot_ex_beta_psth():
    d = '022315'
    Spk_Day = S_bin[d]*binsize
    nunits = Spk_Day.shape[2]
    mc = np.zeros((10,nunits))
    mc_n = np.zeros((10, nunits ))
    nf = np.zeros((10,nunits))
    nf_n = np.zeros((10, nunits))

    J = [0, 6, 38, 32]
    Jdict = {}

    for iu, un in enumerate(Params[d]['sorted_un']):
        if iu in J:
            Jdict[iu, 'mc'] = []
            Jdict[iu, 'nf'] = []
        lfp_lab = Params[d]['day_lfp_labs']

        ix = np.nonzero(np.logical_and(lfp_lab > 80, lfp_lab<=87))[0]
        ix2 = np.nonzero(np.logical_or(lfp_lab==88, lfp_lab==64))[0]
        for t, trl in enumerate(range(ntrials)):
            sets = get_sets(Beta_Day[t,:])
            for s in sets:
                ss = np.array(s)
                ed = np.min([10, len(s)])
                if t in ix:
                    mc[:ed, iu] += Spk_Day[t, ss[:ed], iu]
                    mc_n[:ed, iu] += 1
                    
                    if iu in J:
                        print iu
                        if ed == 10:
                            Jdict[iu, 'mc'].append(Spk_Day[t, ss[:ed], iu])
                        else:
                            additional = 10 - ed
                            Jdict[iu, 'mc'].append(np.hstack(( Spk_Day[t, ss[:ed], iu], np.zeros((additional, )) + np.nan)))


                elif t in ix2:
                    nf[:ed, iu] += Spk_Day[t, ss[:ed], iu]
                    nf_n[:ed, iu] += 1
                    if iu in J:
                        print iu
                        if ed == 10:
                            Jdict[iu, 'nf'].append(Spk_Day[t, ss[:ed], iu])
                        else:
                            additional = 10 - ed
                            Jdict[iu, 'nf'].append(np.hstack(( Spk_Day[t, ss[:ed], iu], np.zeros((additional, )) + np.nan)))


    # mn_mc = mc / np.tile(mc_n[:, np.newaxis], [1, nunits])
    # mn_nf = nf / np.tile(nf_n[:, np.newaxis], [1, nunits])


    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4))
    #f, ax = plt.subplots(nrows=8, ncols=8)

    plot_to = 5

    for i in range(4):
        j= J[i]
        yy = []
        #j = i
        mn_mc = mc[:, j]*20 / mc_n[:, j]
        ax[i/2, i%2].plot(mn_mc[:plot_to], 'b', linewidth=3)
        #sd = np.nanstd(np.vstack((Jdict[j, 'mc'])), axis=0)/np.sqrt(mc_n[:, iu])
        #ax[i/2, i%2].fill_between(range(10), mn_mc-sd, mn_mc+sd, color='b', alpha=0.5)


        mn_nf = nf[:, j]*20 / nf_n[:, j]
        ax[i/2, i%2].plot(mn_nf[:plot_to], 'r', linewidth=3)
        #sd = np.nanstd(np.vstack((Jdict[j, 'nf'])), axis=0)
        #ax[i/2, i%2].fill_between(range(10), mn_nf-sd, y2=mn_nf+sd, color='r', alpha=0.5)
        ax[i/2, i%2].set_title('sig '+Params[d]['sorted_un'][j])

        yy.append(mn_mc[:plot_to])
        yy.append(mn_nf[:plot_to])
        yy = np.hstack((yy))
        y_lims = np.linspace(np.nanmin(yy), np.nanmax(yy),3)
        y_lab = [str(int(round(10*y)/10.)) for y in y_lims]
        ax[i/2, i%2].set_yticks(y_lims)
        ax[i/2, i%2].set_yticklabels(y_lab)


    for i in range(2):
        for j in range(2):
            #ax[i, j].set_xlim([0., 5.])
            ax[i, j].set_xticks([0.0, 2.5, 5.0])
            ax[i, j].set_xticklabels([0., 125, 250])
            ax[i, j].set_xlabel('Time (ms)')
            ax[i, j].set_ylabel('FR (Hz)')
            
    plt.tight_layout()
    plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/NeuronPaper/JNeuroDraft/example_psths_by_beta_64only.eps', format='eps', dpi=300)

def get_sets(arr):
    sets = []
    subset= []
    flag = 'nogo'
    for j, i in enumerate(arr):
        if i == 1 and flag == 'nogo':
            subset.append(j)
            flag = 'go'
        elif i == 1 and flag == 'go':
            subset.append(j)
        elif i ==  0 and flag == 'go':
            sets.append(subset)
            subset = []
            flag = 'nogo'
        elif i == 0 and flag == 'nogo':
            pass
    return sets

def beta_psth(test, binsize, blocks, days, rt_dict, S_bin, B_bin, bin_kin_signal, Params, animal='grom'):
    if animal == 'grom':
        test = False
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, animal='grom', all_cells=False)
        binsize = 100
        import state_space_w_beta_bursts as ssbb
        B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=binsize, animal='grom')
        
        kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt = predict_kin_w_spk.get_kin(days, blocks, bef, aft, 
            go_times_dict, lfp_lab, binsize, smooth = 50, animal='grom')

    elif animal == 'cart':
        test = False
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, animal='cart', all_cells=True)
        binsize = 100
        import state_space_w_beta_bursts as ssbb
        B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=binsize, animal='cart')
        
        kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt = predict_kin_w_spk.get_kin(days, blocks, bef, aft, 
            go_times_dict, lfp_lab, binsize, smooth = 50, animal='cart')

    cmap = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]

    f, ax = plt.subplots(figsize=(5, 4))
    for i_d, d in enumerate(days):
        ntrials, nbins, nunits = S_bin[d].shape
        Spk_Day = S_bin[d]*binsize
        Beta_Day = B_bin[d]
        Kin_Day = []
        Spd_Day = []
        for i_b, blk in enumerate(blocks[i_d]):
            Kin_Day.append(bin_kin_signal[blk, d])
            spd = binned_kin_signal_dict[blk, d]
            Spd_Day.append(np.sqrt(spd[:, 2, :]**2 +spd[:,3,:]**2 ))
        Kin_Day = np.vstack((Kin_Day))
        Spd_Day = np.vstack((Spd_Day))
        perc_same_diff = np.zeros((2, ))

        for iu, un in enumerate(Params[d]['sorted_un']):
            lfp_lab2 = Params[d]['day_lfp_labs']

            ix = np.nonzero(np.logical_and(lfp_lab2 > 80, lfp_lab2<=87))[0]
            ix2 = np.nonzero(np.logical_or(lfp_lab2==88, lfp_lab2<80))[0] #RT PSTH changed all 64s to 88s. Oops

            spk = {}
            spk['mc'] = []
            spk['beta'] = []
 
            for t, trl in enumerate(range(ntrials)):
                if t in ix:
                    mc_hold = np.nonzero(np.logical_and(Beta_Day[t,:] == 1, Kin_Day[t, :]==0))[0]
                    if len(mc_hold) > 0:
                        spk['mc'].append(Spk_Day[t, mc_hold, iu])
                    else:
                        print 'skipping ', t
                elif t in ix2:
                    beta_hold = np.nonzero(np.logical_and(Beta_Day[t,:] == 1, Kin_Day[t, :]==0))[0]
                    if len(beta_hold) > 0:
                        spk['beta'].append(Spk_Day[t, beta_hold, iu])
                    else:
                        print 'skipping ', t


            mc = np.hstack((spk['mc']))
            bt = np.hstack((spk['beta']))
            if np.logical_and(np.mean(mc)/.1 > 5., np.mean(bt)/.1 > .5):
                print 'sizes: ', mc.shape, bt.shape

                u1, p1 = scipy.stats.mannwhitneyu(mc, bt)
                if p1 < 0.05:
                    perc_same_diff[1] += 1
                else:
                    perc_same_diff[0] += 1
        ax.bar(i_d+.1, perc_same_diff[1]/float(np.sum(perc_same_diff)), color=tuple(np.array(cmap[i_d])/255.))
        ax.text(i_d+.5, .1+perc_same_diff[1]/float(np.sum(perc_same_diff)), 'n='+str(iu+1), fontsize=16,horizontalalignment='center')
    ax.set_ylabel('Percent of Units')
    ax.set_xticks(np.arange(6)+0.5)
    ax.set_xticklabels(days, rotation='vertical')
    ax.set_ylim([0, 1.])
    plt.tight_layout()
    plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/NeuronPaper/JNeuroDraft/g_perc_sig_diff_FR_beta_vs_NF_hold_beta_events_on_to_targ64_100ms_bins.eps', format='eps', dpi=300)

def entropy_phase(train_ix, test_ix, day_bc, Params_day):
    f, ax = plt.subplots()
    f2, ax2 = plt.subplots(ncols=5)
    nunits = day_bc.shape[2]
    nbins = day_bc.shape[1]
    t = np.linspace(-1500, 1000, nbins)
    phz_bins = np.linspace(-np.pi, np.pi, 20)

    p0 = np.ones(len(phz_bins) -1)
    p0 = p0/float(np.sum(p0))
    e0 = np.sum(p*np.log(p0))
    
    mc = []
    mc_arr = []
    for i in range(nbins):
        tmp, _ = np.histogram(day_bc[train_ix, i, 1].reshape(-1), phz_bins)
        p = tmp/float(np.sum(tmp))
        logp = np.log(p)
        logp[logp==np.inf] = np.nan
        #Calculate entropy:
        e = np.nansum(p*logp) 
        mc.append(e)
        mc_arr.append(p)
    MC = np.vstack((mc_arr))
    ax2[0].pcolormesh(MC)

    ax.plot(t, mc, label='mc')
    ax.plot(t, [e0]*nbins, label='unif')

    lfp_lab = Params_day['day_lfp_labs'][test_ix]
    
    for j in [84, 87]:
        lfp = []
        lfp_arr = []
        for i in range(nbins):
            ix_ = np.nonzero(lfp_lab==j)
            tmp, _ = np.histogram(day_bc[test_ix[ix_], i, 1].reshape(-1), phz_bins)
            p = tmp/float(np.sum(tmp))
            logp = np.log(p)
            logp[logp==np.inf] = np.nan
            e = np.nansum(p*logp)
            lfp.append(e)
            lfp_arr.append(p)
        ax.plot(t, lfp, label=str(j))
        LFP = np.vstack((lfp_arr))
        ax2[j-84+1].pcolormesh(LFP)
    ax.legend()

def run_LDA_beta(method):
    binsizes = [5, 10, 25, 50, 100, 125]
    SC = {}
    CH = {}

    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = get_unbinned(test=True)
    
    for b in binsizes:
        B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, binsize=b)

        for i_d, d in enumerate(days):
            day_spks = S_bin[d]
            day_b = B_bin[d]
            day_trial_type = []
            day_lfp_lab = []

            for ib, bn in enumerate(blocks[i_d]):
                n_trials = len(lfp_lab[bn, d])

                if int(mc_indicator[i_d][ib]) == 1:
                    tt = ['mc']*n_trials
                elif int(mc_indicator[i_d][ib]) == 0:
                    tt = ['beta']*n_trials
                else:
                    raise Exception

                day_trial_type.append(tt)
                day_lfp_lab.append(lfp_lab[bn, d])

            day_trial_type = np.hstack(day_trial_type)
            day_lfp_lab = np.hstack((day_lfp_lab))

            #Training: 
            train_ix = np.nonzero(day_trial_type == 'mc')[0]
            test_ix = np.nonzero(day_trial_type == 'beta')[0]

            #Train and test on MC, then LFP targs individually: 
            day_spks_mc = day_spks[day_trial_type=='mc', :, :]
            day_b_mc = day_b[day_trial_type=='mc', :]
            score_mc, chance_mc = predict_beta_LDA([], np.arange(day_spks_mc.shape[0]), day_b_mc, day_spks_mc, method=method, train='xval_beta')
            SC[b, d, 'mc'] = score_mc
            CH[b, d, 'mc'] = chance_mc

            for l in range(84, 88):
                ix_lfp = np.nonzero(day_lfp_lab==l)[0]
                day_spks_beta = day_spks[ix_lfp, :, :]
                day_b_beta = day_b[ix_lfp, :]
                print 'day_spks / beta shapes: ', b, d, day_spks_beta.shape, day_b_beta.shape
                score, chance = predict_beta_LDA([], np.arange(len(ix_lfp)), day_b_beta, day_spks_beta, method=method, train='xval_beta')
                SC[b, d, l] = score
                CH[b, d, l] = chance

    T = dict(SC=SC, CH=CH)
    iso = datetime.datetime.now()
    iso_date = iso.isoformat()
    fname = ssbb.master_save_directory2 + 'beta_regressions/'+iso_date[:10]+'_'+method+'_LDA_predict_binary_beta.pkl'
    pickle.dump(T, open(fname, 'wb'))

def predict_beta_LDA(train_ix, test_ix, day_b, day_spks, method='LDA', train='mc', epochs=None):

    train_ix = np.array(train_ix)
    test_ix = np.array(test_ix)

    ntrials, nbins, nunts = day_spks.shape
    score = {}
    chance = {}
    if epochs is None:
        epochs = ['hold' , 'reach', 'full']
    else:
        print 'epochs: ', epochs

    if train == 'mc':
        train_ix_master = [train_ix]
        test_ix_master = [test_ix]
        reps = 1

    elif train == 'xval_beta':
        train_ix_master = []
        test_ix_master = []
        reps = 2
        for i in range(reps):
            ix = np.random.permutation(int(len(test_ix)*(1/float(reps))))
            train_ix_master.append(test_ix[ix])
            test_ix_i = np.array(list(set(test_ix).difference(set(test_ix[ix]))))
            test_ix_master.append(test_ix_i)

    for ie, e in enumerate(epochs):
        if e == 'hold':
            nbins_trunc = [0, 0.6*nbins]
        elif e == 'full':
            nbins_trunc = [0, nbins]
        elif e == 'reach':
            nbins_trunc = [0.6*nbins, nbins]

        for r in range(reps):
            train_ix = train_ix_master[r]
            test_ix = test_ix_master[r]

            spks_train_trunc = day_spks[train_ix, nbins_trunc[0]:nbins_trunc[1], :]
            spks_test_trunc = day_spks[test_ix, nbins_trunc[0]:nbins_trunc[1], :]

            b_train_trunc = day_b[train_ix, nbins_trunc[0]:nbins_trunc[1]]
            b_test_trunc = day_b[test_ix, nbins_trunc[0]:nbins_trunc[1]]

            assert spks_train_trunc.shape[1] == spks_test_trunc.shape[1]
            assert spks_train_trunc.shape[2] == spks_test_trunc.shape[2]

            s_trn = spks_train_trunc.reshape(len(train_ix)*(nbins_trunc[1]-nbins_trunc[0]), nunts)
            s_tst = spks_test_trunc.reshape(len(test_ix)*(nbins_trunc[1]-nbins_trunc[0]), nunts)

            b_trn = b_train_trunc.reshape(-1)
            b_tst = b_test_trunc.reshape(-1)

            #Duplicate beta events to make classes equal :
            non_beta_ix = np.nonzero(b_trn==0)[0]
            beta_ix = np.nonzero(b_trn == 1)[0]

            # fact = (np.floor(len(non_beta_ix)/float(len(beta_ix)))-1).astype(int)

            # b_trn_add = np.ones(len(beta_ix)*fact)
            # s_trn_add = np.tile(s_trn[beta_ix, :], [fact, 1])

            # s_trn_eq = np.vstack((s_trn, s_trn_add))
            # b_trn_eq = np.hstack((b_trn, b_trn_add))
            s_trn_eq = s_trn.copy()
            b_trn_eq = b_trn.copy()

            if method == 'LDA':
                clf = LDA(solver='lsqr')
            elif method == 'SVM':
                clf = svm.SVC()

            #X: samples x features, y: samples
            clf.fit(s_trn_eq, b_trn_eq)

            #feature_importance = parse_LL_by_predictor(clf)
            #Test above chance level: 

            chance_r_test = []
            chance_r_train = []

            for i in range(5):
                ix = np.random.permutation(len(b_tst))
                chance_r_test.append(clf.score(s_tst, b_tst[ix]))

                ix2 = np.random.permutation(len(b_trn))
                chance_r_train.append(clf.score(s_trn, b_trn[ix2]))
            try: 
                score[r, e, 'test'].append(clf.score(s_tst, b_tst))
                score[r, e, 'train'].append(clf.score(s_trn, b_trn))
                chance[r, e, 'test'].append(chance_r_train)
                chance[r, e, 'train'].append(chance_r_train)

            except:
                score[r, e, 'test'] = clf.score(s_tst, b_tst)
                score[r, e, 'train'] = clf.score(s_trn, b_trn)
                chance[r, e, 'test'] = chance_r_test
                chance[r, e, 'train'] = chance_r_train

    return score, chance

def parse_LL_by_predictor(clf, s_trn, b_trn):
    nunits = s_trn.shape[1]
    #On average during training, how do individual spikes contribute to +/- decisions? 
    f, ax = plt.subplots()
    #Beta on: 
    color = ['b', 'r']
    d_beta = np.zeros((2, nunits))
    for beta_status in [0, 1]:
        b_ix = np.nonzero(b_trn==beta_status)[0]
        s_sub_trn = s_trn[b_ix, :]
        coef_tile = np.tile(clf.coef_, [len(b_ix), 1])
        avg_contr = np.sum(s_sub_trn*coef_tile, axis=0)/float(len(b_ix))
        ax.bar(np.arange(nunits)+(0.2*beta_status), avg_contr, color=color[beta_status])
        d_beta[beta_status, :] = avg_contr
        #ax.errorbar(np.arange(nunits), np.sum(s_sub_trn*coef_tile, axis=0), yerr=np.var(s_sub_trn*coef_tile, axis=0))
    d = np.squeeze(np.diff(d_beta, axis=0))

def get_confusion_mat(y, y_pred):
    # actual --> predicted
    confusion = np.zeros((2, 2))

    ix0 = np.nonzero(y==0)[0]
    ix1 = np.nonzero(y==1)[0]

    confusion[0, 0] = len(np.nonzero(y_pred[ix0]==0)[0])
    confusion[0, 1] = len(np.nonzero(y_pred[ix0]==1)[0])
    confusion[1, 0] = len(np.nonzero(y_pred[ix1]==0)[0])
    confusion[1, 1] = len(np.nonzero(y_pred[ix1]==1)[0])
    return confusion

class filter_fcns(object):
    def __init__(self, *args, **kwargs):
        pass

    def format_data(self, kin_train, kin_test, spk_train, spk_test, binsize, n_units, predict, nbins):
        #Gaussian prior on kinematics 
        ssm = state_space_models.StateSpaceEndptVel2D()
        A, B, W = ssm.get_ssm_matrices(update_rate=1./binsize)

        if len(kin_train.shape) > 1:
            n_kin_ft = kin_train.shape[1]
            if predict == 'kin':
                w0 = np.cov(100*kin_train.T)
                fact = w0[2,2]/w0[3,3]
                W[3,3]  *= fact
                kin_train = kin_train.T
                kin_test = kin_test.T
                final_sub_ix = np.array([0, 2, 3, 5])

            elif predict == 'beta':
                w0 = np.cov(10*kin_train.T)
                W[3, 3] = w0
                kin_train = np.vstack((kin_train.T, np.zeros((1, kin_train.shape[0]))))
                kin_test = np.vstack((kin_test.T, np.zeros((1, kin_test.shape[0]))))
                final_sub_ix = np.array([3, 5])
        else:
            raise Exception

        if predict=='kin':
            kin_actual = 100*kin_train[2:,:]
        elif predict == 'beta':
            kin_actual = 10*kin_train

        #Convert to spike counts (instead of mean spk count)
        spk_train = spk_train*binsize
        spk_test = spk_test*binsize

        #Neural data: units x T:
        obs = spk_train.reshape(spk_train.shape[0]*spk_train.shape[1], n_units).T #Convert to spike counts (instead of mean spk count)

        #Kin data reshaped: 
        if predict == 'kin':
            k_tr = 100*kin_train.reshape(np.max(kin_train.shape)/nbins, nbins, n_kin_ft)
            tr, bn, _ = k_tr.shape
            z = np.zeros((tr, bn, 1))
            z1 = np.ones((tr, bn, 1))
            k_tr = np.dstack((k_tr[:, :, 0], z, k_tr[:, :, 1], k_tr[:, :, 2], z, k_tr[:, :, 3], z1))

            k_te = 100*kin_test.reshape(np.max(kin_test.shape)/nbins, nbins, n_kin_ft)
            tr, bn, _ = k_te.shape
            z = np.zeros((tr, bn, 1))
            z1 = np.ones((tr, bn, 1))
            k_te = np.dstack((k_te[:, :, 0], z, k_te[:, :, 1], k_te[:, :, 2], z, k_te[:, :, 3], z1))

        elif predict == 'beta':
            k_tr = 10*kin_train.T.reshape(np.max(kin_train.shape)/nbins, nbins, n_kin_ft+1)
            tr, bn, _ = k_tr.shape
            z = np.zeros((tr, bn, 1))
            z1 = np.ones((tr, bn, 1))
            k_tr = np.dstack((z, z, z, k_tr[:, :, 0], z, z, z1))

            k_te = 10*kin_test.T.reshape(np.max(kin_test.shape)/nbins, nbins, n_kin_ft+1)
            tr, bn, _ = k_te.shape
            z = np.zeros((tr, bn, 1))
            z1 = np.ones((tr, bn, 1))
            k_te = np.dstack((z, z, z, k_te[:, :, 0], z, z, z1))

        return obs, ssm, kin_actual, A, B, W, kin_train, kin_test, spk_train, spk_test, k_tr, k_te, n_kin_ft, final_sub_ix

    def ppf_make(self,kin_train, kin_test, spk_train, spk_test, binsize, predict, n_units, nbins):
        ### Preprocess data ### 
        obs, ssm, kin_actual, A, B, W, kin_train, kin_test, spk_train, spk_test, k_tr, k_te, n_kin_ft, final_sub_ix= self.format_data(kin_train, kin_test, 
            spk_train, spk_test, binsize, n_units, predict, nbins)

        #Squish PPF obs# : 
        obs[obs > 1] = 1

        C = np.zeros([n_units, ssm.n_states])
        C[:, ssm.drives_obs_inds], pvals = ppfdecoder.PointProcessFilter.MLE_obs_model(kin_actual, obs, include_offset=True)
        
        ppf = ppfdecoder.PointProcessFilter(A, W, C, B=B, dt=1./binsize, is_stochastic=ssm.is_stochastic)
        ppf._init_state()

        spk_tr = spk_train.copy()
        spk_te = spk_test.copy()

        #Use PPF to predict: 
        pred_train = ppf_predict(k_tr, spk_tr, ppf)
        pred_test = ppf_predict(k_te, spk_te, ppf)

        return pred_train[:, :, final_sub_ix], pred_test[:, :, final_sub_ix], n_kin_ft

    def kf_make(self, kin_train, kin_test, spk_train, spk_test, binsize, predict, n_units, nbins):
        obs, ssm, kin_actual, A, B, W, kin_train, kin_test, spk_train, spk_test, k_tr, k_te, n_kin_ft, final_sub_ix = self.format_data(kin_train, kin_test, 
            spk_train, spk_test, binsize, n_units, predict, nbins)

        C = np.zeros([n_units, ssm.n_states])
        C[:, ssm.drives_obs_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin_actual, obs, include_offset=True)
        
        # instantiate KFdecoder
        kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=ssm.is_stochastic)
        kf._init_state()

        spk_tr = spk_train.copy()
        spk_te = spk_test.copy()

        #Use PPF to predict: 
        pred_train = kf_predict(k_tr, spk_tr, kf)
        pred_test = kf_predict(k_te, spk_te, kf)
        return pred_train[:, :, final_sub_ix], pred_test[:, :, final_sub_ix], n_kin_ft

def ppf_predict(kin, spk, ppf, squish=True, plot=False):
    s = ppf.state
    ntrials, nbins, n_ft = kin.shape
    pred_kin = []

    for n in range(ntrials):
        #init state: 
        ppf._init_state()
        st = []

        #Cycle through bins
        for t in range(nbins):
            obs = np.array([spk[n, t, :]]).T

            #Squish observations
            if squish:
                obs[obs>1] = 1

            s = ppf._forward_infer(s, obs)
            st.append(s.mean)
        pred_kin.append(st)
    all_pred_kin = np.dstack((pred_kin)).T
    all_pred_kin = np.swapaxes(all_pred_kin[:, :-1, :], 1, 2)

    if plot:
        f, ax = plt.subplots(nrows=3, ncols=3)
        if decode == 'vel':
            offs = 0
        elif decode == 'pos':
            offs = 3
        for i in range(j, j+9):
        
            k = i-j
            ax[k/3, k%3].plot(kin[i, :, 3-offs], 'k-', label='xvel')
            ax[k/3, k%3].plot(kin[i, :, 5-offs], 'r-', label='yvel')
            ax[k/3, k%3].plot(all_pred_kin[i, :, 3-offs], 'k--', label='xvel_hat')
            ax[k/3, k%3].plot(all_pred_kin[i, :, 5-offs], 'r--', label='yvel_hat')
        plt.show()
    
    return all_pred_kin

def ppf_predict2(kin, spk, ppf_obj):
    pred_kin = []
    ntrials, nbins, nkin_ft = kin.shape
    for n in range(ntrials):
        st = []
        #Cycle through bins: 
        for t in range(nbins):
            obs = spk[n, t, :].T
            
            #Squish: 
            obs[obs>1] = 1
            unpred_spks = np.mat(obs - ppf_obj.lambda_pred*ppf_obj.dt).T
            X_ = ppf_obj.prior_mn + ppf_obj.P_pred*ppf_obj.C.T*unpred_spks
            st.append(np.squeeze(np.array(X_[:nkin_ft])))
        pred_kin.append(st)
    all_pred_kin = np.squeeze(np.dstack((pred_kin)).T)
    return all_pred_kin

def kf_predict(kin, spk, kf):

    return ppf_predict(kin, spk, kf, squish=False)

def plot_errs(kin_train, kin_train_pred, kin_test, kin_test_pred, B_train, B_test, 
    lfp_labs_train, lfp_labs_test, plot=True, metrics=None):

    if metrics is None:
        metrics = dict()

    for kin in ['signal_error', 'Y_pred', 'kin', 'R2']:
        if plot:
            f, ax = plt.subplots(nrows=3)
        else:
            ax = None

        ax, metrics = bar_plot_reg_accuracy(ax, kin_train, kin_train_pred, B_train, plot_by_beta = True, 
            lfp_label=None, metric=kin, metrics_dict=metrics, plot=plot)

        ax, metrics = bar_plot_reg_accuracy(ax, kin_test, kin_test_pred, B_test, plot_by_beta = True, 
            lfp_label=lfp_labs_test, metric=kin, metrics_dict = metrics, plot=plot)

        # for axi in ax:
        #     if kin == 'signal_error':
        #         axi.set_ylim([0., .003])
        #     else:
        #         axi.set_ylim([-.005, .09])
    return ax, metrics

def get_FR_diff(B_bin, S_bin, days, Params, b, method='hold', trial_types='mc'):
    ''' Test if FR is sig diff during beta or not, during MC trials

    method: 'hold' just looks at spks during beta events vs. spks
            during non-beta event (all only during hold def 0 - 1300ms) and
            uses kstest_2samp to compare

        'before_beta_vs_beta' looks at mean during beta events vs. 
            single bin right before beta events and uses ttest_rel (relationship 
            b/w observations, e.g. repeated samples)


    '''
    pv = dict()
    f, ax = plt.subplots(nrows=4)
    pbins = np.linspace(0, 1, 20)
    
    for i_d, d in enumerate(days):
        if trial_types == 'mc':
            ix = np.nonzero(Params[d]['day_lfp_labs']<= 80)[0]
        elif trial_types == 'beta_mod':
            ix = np.nonzero(Params[d]['day_lfp_labs']> 80)[0]
        if method=='hold':
            end = int(1300/b)
            spks = S_bin[d][ix, :end, :]
            spks = spks.reshape(spks.shape[0]*spks.shape[1], spks.shape[2])
            nbins, n_units = spks.shape
            beta = B_bin[d][ix, :end].reshape(-1)
        
        elif method=='before_beta_vs_beta':
            spks = S_bin[d][ix, :, :]
            ntrials, nbins, n_units = spks.shape
            beta = B_bin[d][ix, :]

        d_fr = dict()
        DFR = {}
        for iu in range(n_units):
            d_fr[iu] = []

        struct = np.zeros((3, 3))
        struct[1,:] = 1
        
        if method == 'before_beta_vs_beta':
            for nt in range(ntrials):
                    trl = np.vstack((np.zeros((1, nbins)), beta[nt, :][np.newaxis, :], np.zeros((1, nbins))))
                    id_regions, num_ids = ndimage.label(trl, structure=struct)
                    bef_beta = []
                    dur_beta = []
                    for iu in range(n_units):
                        for reg in np.unique(id_regions[id_regions>0]):
                            tmpix = np.nonzero(id_regions[1,:]==reg)[0]
                            if tmpix[0] > 0:
                                dur_beta.append(np.mean(spks[nt, tmpix, iu]))
                                bef_beta.append(spks[nt, tmpix[0] - 1, iu])
                        d_fr[iu].append(np.vstack((dur_beta, bef_beta)))


            for iu in range(n_units):
                DFR[d, iu]= np.hstack((d_fr[iu]))
                t, pv[d, iu] = scipy.stats.ttest_rel(DFR[d, iu][0, :], DFR[d, iu][1, :])
                pv[d, iu, 'fr'] = np.mean(DFR[d, iu], axis=1)


        elif method == 'hold':
            ix0 = np.nonzero(beta==0)[0]
            ix1 = np.nonzero(beta==1)[0]

            for iu in range(n_units):
                pv[d, iu, 'fr'] = np.zeros((2,))
                pv[d, iu, 'fr'][0] = np.mean(spks[ix1, iu])
                pv[d, iu, 'fr'][1] = np.mean(spks[ix0, iu])
                t, pv[d, iu] = scipy.stats.ks_2samp(spks[ix1, iu], spks[ix0, iu])



        for iu in range(n_units):
            ax[i_d].bar([iu], pv[d, iu, 'fr'][0], alpha=.5, color='r')
            ax[i_d].bar([iu+.3], pv[d, iu, 'fr'][1], alpha=.5, color='b')
            if pv[d, iu] < 0.05:
                ax[i_d].plot(iu+.15, 0, 'r*')
                print d, iu

def bar_plot_reg_accuracy(ax, Y, Y_pred, B_beta, plot_by_beta = True, 
    lfp_label=None, metric='signal_error', metrics_dict = None, plot=True):

    ''' 
    Assume: before = 1.5, aft = 1., so binsize = 2500/Y.shape[1]
    '''
    if metrics_dict is None:
        metrics_dict = dict()

    t = np.arange(Y.shape[1])*(2500/float(Y.shape[1]))
    t_step = 0.5*(t[1]-t[0])
    t = t + t_step - 1500

    epochs = ['hold', 'reach', 'full']
    tix = {}
    ntrials = Y.shape[0]

    if not plot_by_beta:
        B_beta = np.ones_like(Y_pred)
    if lfp_label is None:
        #Assume Manual Control: 
        lfp_label = np.zeros((ntrials)) + 88

    for epoch in epochs:
        if epoch=='hold':
            t_ix = np.nonzero(t<0)[0]
        elif epoch == 'reach':
            t_ix = np.nonzero(t>=0)[0]
        elif epoch=='full':
            t_ix = np.arange(len(t))
        tix[epoch] = t_ix

    if metric == 'signal_error':
        y_metric = (Y - Y_pred)**2
    elif metric == 'Y_pred':
        y_metric = Y_pred
    elif metric == 'kin':
        y_metric = Y
    elif metric == 'R2':
        y_bar = np.mean(Y)
        SST = (Y - y_bar)**2
        SSRes = (Y - Y_pred)**2
        y_metric = 1 - (SSRes/SST) # R2


    y_metric_by_epoch = {}

    for epoch in epochs:
        t_ix = tix[epoch]
        tmp_beta = np.zeros((ntrials, len(t_ix)))
        tmp_beta[:,:] = np.nan

        tmp_nonbeta = np.zeros((ntrials, len(t_ix)))
        tmp_nonbeta[:,:] = np.nan

        for i in range(ntrials):
            for j in t_ix:
                if B_beta[i, j]:
                    tmp_beta[i,j-t_ix[0]] = y_metric[i, j]
                else:
                    tmp_nonbeta[i,j-t_ix[0]] = y_metric[i, j]
        y_metric_by_epoch[epoch, 'beta_sum'] =tmp_beta
        y_metric_by_epoch[epoch, 'nonbeta_sum'] = tmp_nonbeta

    for ie, epoch in enumerate(epochs):
        if plot:
            axi = ax[ie]
        for il, l in enumerate(np.unique(lfp_label)):
            ix = np.nonzero(lfp_label==l)[0]
            if len(ix) > 0.:
                sub_beta = y_metric_by_epoch[epoch, 'beta_sum'][ix,:].reshape(-1)
                sub_non_beta = y_metric_by_epoch[epoch, 'nonbeta_sum'][ix,:].reshape(-1)

                mn = np.nanmean(sub_beta)
                mn_ = np.nanmean(sub_non_beta)

                std = np.nanstd(sub_beta)
                std_ = np.nanstd(sub_non_beta)

                n1 = len(np.nonzero(sub_beta!=np.nan)[0])
                n2 = len(np.nonzero(sub_non_beta!=np.nan)[0])

                sem = std/np.sqrt(n1)
                sem_ = std_/np.sqrt(n2)

                ### METRICS: MEAN, STD, N1, SEM ###
                metrics_dict[epoch, metric, l, 'beta_on'] = [mn, std, n1, sem]
                metrics_dict[epoch, metric, l, 'beta_off'] = [mn_, std_, n2, sem_]

                if l in range(84, 88):
                    color = cmap[il]
                else:
                    color = 'black'
                if plot:
                    axi.bar([l], mn, yerr=sem, width=.4, color=color)
                    axi.bar([l+.3], mn_, yerr=sem_, width=.4, color=color, alpha = .5)
                    axi.set_ylabel(epoch)
    if plot:
        ax[0].set_title(metric)
    return ax, metrics_dict

class ppf_pk(object):
    def __init__(self, C, w, mu, binsize):
        self.C = np.mat(C)
        self.prior_cov = np.mat(w)
        self.prior_cov_inv = np.mat(w).I
        self.prior_mn = np.mat(mu).T

        self.lambda_pred = np.squeeze(np.array(np.exp(self.C*self.prior_mn)))
        self.lambda_pred_mat = np.mat(np.diag(self.lambda_pred))
        self.P_pred_inv = (self.C.T*self.lambda_pred_mat*self.C + self.prior_cov_inv)
        self.P_pred = self.P_pred_inv.I
        self.dt = 1/float(binsize)

def test_hold_reach_thresh(binned_kin_signal_dict, S_bin, days, blocks, Params, t_ix=np.arange(25), nreps=10):
    test_thresh = [3.5]
    f, ax = plt.subplots(nrows=4)
    f2, ax2 = plt.subplots(nrows=4)
    f3, ax3 = plt.subplots(nrows=4, ncols=2)

    ax_master = [ax, ax2]
    conf_dict = {}
    for it, t in enumerate(test_thresh):
        for i_d, day in enumerate(days):
            ix_mc = np.nonzero(Params[day]['day_lfp_labs']<80)[0]
            ix_bet = np.nonzero(Params[day]['day_lfp_labs']>80)[0]
            trl_ix = [ix_mc, ix_bet]
            
            tmp_kin = []
            for i_b, blk in enumerate(blocks[i_d]):
                spd = np.sqrt(np.sum(binned_kin_signal_dict[blk, day][:, [2, 3], :]**2, axis=1))
                tmp_kin.append(spd)
            kin_ =np.vstack((tmp_kin))
            bin_kin_ = np.zeros_like(kin_)
            bin_kin_[np.abs(kin_)>t/100.] = 1
            
            for ii, ix in enumerate(trl_ix):
                ix_magic = np.ix_(ix, t_ix)
                ix_magic_spks = np.ix_(ix, t_ix, range(S_bin[day].shape[2]))
                ax3[i_d, ii].pcolormesh(bin_kin_[ix_magic])
                clf = sklearn.lda.LDA()
                Y = bin_kin_[ix_magic].reshape(-1)
                ntrls, nbins, nunits = S_bin[day][ix_magic_spks].shape
                X = S_bin[day][ix_magic_spks].reshape(len(ix)*len(t_ix), nunits)
                
                conf = np.zeros((2, 2))
                for n in range(nreps):
                    ix0 = np.random.permutation(len(Y))
                    ix0 = ix0[:len(Y)*5/6]
                    ix1 = np.array([i for i in range(len(Y)) if i not in ix0])
                    #ix0 = np.random.permutation(len(Y)/2)
                    #ix1 = np.array([i for i in range(len(Y)) if i not in ix])

                    #Equalize # of samples: 
                    fact = int(float(len(Y[ix0]))/np.sum(Y[ix0])) - 1
                    print ii,
                    move_ix= np.nonzero(Y[ix0]==1)[0]

                    concat_Y = np.tile(Y[ix0][move_ix], [fact, ])
                    concat_X = np.tile(X[ix0, :][move_ix, :], [fact, 1])

                    X_ad = np.vstack((X[ix0, :], concat_X))
                    Y_ad = np.hstack((Y[ix0], concat_Y))

                    clf.fit(X_ad, Y_ad)
                    Y_hat = clf.predict(X[ix1])
                    print t, 'test: ', clf.score(X[ix1], Y[ix1]), 'train: ', clf.score(X[ix0], Y[ix0])

                    for i in [0, 1]:
                        ix_ = np.nonzero(Y[ix1]==i)[0]
                        for j in [0, 1]:
                            ix1 = np.nonzero(Y_hat[ix_]==j)[0]
                            conf[i, j] += len(ix1)
                        if n==(nreps-1):
                            conf[i,:] = conf[i, :]/float(np.sum(conf[i, :]))
                            axi = ax_master[ii]

                            axi[i_d].pcolormesh(conf, vmin=0, vmax=1.)
                            if ii == 0:
                                axi[i_d].set_title(day+': mc')
                            elif ii == 1:
                                axi[i_d].set_title(day+': beta')
                conf_dict[t, ii, day] = conf

def plot_conf_dict(conf_dict, days, ii=0):
    ''' plot mc conf matrix from above fcn'''
    f, ax = plt.subplots(nrows=4)
    for i_d, day in enumerate(days):
        c = ax[i_d].pcolormesh(np.flipud(conf_dict[3.5, ii, day]), vmin=0, vmax=1., cmap=plt.get_cmap('Blues'))
        ax[i_d].set_ylabel('Actual Class, '+day)
        ax[i_d].set_yticks([0.5, 1.5])
        ax[i_d].set_yticklabels(['Reach', 'Hold'])
        ax[i_d].set_xticks([])
            
        if i_d == 3:
            ax[i_d].set_xlabel('Predicted Class')
            ax[i_d].set_xticks([0.5, 1.5])
            ax[i_d].set_xticklabels(['Hold', 'Reach'])

        #Colorbar: 
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(c, cax=cbar_ax)

        ax[0].set_title('Confusion Matrices for Manual Control Hold/Reach Classification')

def plot_perc_corr_conf(conf_dict, days):
    f, ax = plt.subplots()
    bar = dict()
    for i in range(4): bar[i]=[]
    for i_d, day in enumerate(days):
        nf_perf = conf_dict[day]['nf','nf', 'perc_corr']
        nf_ch = conf_dict[day]['nf','nf', 'chance_perc_corr']
        nf_c = nf_ch[0]/float(nf_ch[1])
        p = scipy.stats.binom_test(nf_perf[0], nf_perf[1], nf_c)
        bar[2].append(nf_perf[0]/float(nf_perf[1]))
        bar[3].append(nf_c)
        print day, ', nf, ', p
        mc_perf = conf_dict[day]['mc','mc', 'perc_corr']
        mc_ch = conf_dict[day]['mc','mc', 'chance_perc_corr']
        mc_c = mc_ch[0]/float(mc_ch[1])
        p = scipy.stats.binom_test(mc_perf[0], mc_perf[1], mc_c)
        bar[0].append(mc_perf[0]/float(mc_perf[1]))
        bar[1].append(mc_c)
        print day, ', mc, ', p

    c={}
    c[0]='b'
    c[1]='r'
    for i in range(4):
        ax.bar(i+.1, np.mean(bar[i]), color=c[i/2], width=0.8)
        ax.errorbar(i+0.5, np.mean(bar[i]), yerr=np.std(bar[i])/np.sqrt(len(bar[i])),
            ecolor=c[i/2])
    ax.set_xlim([-.5, 4.5])
    ax.set_xticks(0.5+np.arange(4))
    ax.set_xticklabels(['MC Perf.', 'MC Chance', 'NF Perf.', 'NF Chance'])
    ax.set_ylabel('Percent Correct')
    ax.set_ylim([0., .9])

    ax.plot([0.5, 1.5], [.75, .75], 'k-')
    ax.plot([0.5, 0.5], [.75, .73], 'k-')
    ax.plot([1.5, 1.5], [.75, .63], 'k-')

    ax.plot([2.5, 3.5], [.75, .75], 'k-')
    ax.plot([2.5, 2.5], [.75, .73], 'k-')
    ax.plot([3.5, 3.5], [.75, .5], 'k-')  
    ax.text(1, .8, 'all p < 0.001', horizontalalignment='center',fontsize=18 )    
    ax.text(3, .8, 'all p < 0.001', horizontalalignment='center',fontsize=18 )      
    plt.tight_layout()
    plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/grom_spk_v2_may2016/6class_within_task_bars.eps', format='eps', dpi=300)


def plot_6class_conf_dict(CONF, days, train='nf', test='nf'):
    ''' plot confusion matrices '''

    if np.logical_and(train=='nf', test=='nf'):
        for i_d, day in enumerate(days):
            f, ax = plt.subplots()
            c = ax.pcolormesh(np.flipud(CONF[day][train, test, 'xval']), vmin=0, vmax=1., cmap=plt.get_cmap('Blues'))
            ax.set_ylabel('Actual Class, '+day)
            ax.set_yticks(np.arange(6)+0.5)
            ax.set_yticklabels(['Reach', 'Hold', 'Beta4', 'Beta3', 'Beta2', 'Beta1'])
            ax.set_xticks([])
            ax.set_xlabel('Predicted Class')
            ax.set_xticks(np.arange(6)+0.5)
            ax.set_xticklabels(['Beta1', 'Beta2', 'Beta3', 'Beta4', 'Hold', 'Reach'])

            #Colorbar: 
            f.subplots_adjust(right=0.8)
            cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
            f.colorbar(c, cax=cbar_ax)

            ax.set_title('Confusion Matrices for Training on '+train+' , Test on '+test+': '+day)
            plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/grom_spk_v2_may2016/nfnf_conf_'+day+'.eps', format='eps', dpi=300)
    else:
        for i_d, day in enumerate(days):
            f, ax = plt.subplots()
            c = ax.pcolormesh(np.flipud(CONF[day][train, test, 'xval'][4:,4:]), vmin=0, vmax=1., cmap=plt.get_cmap('Blues'))
            ax.set_ylabel('Actual Class, '+day)
            ax.set_yticks(np.arange(2)+0.5)
            ax.set_yticklabels(['Reach', 'Hold'])
            ax.set_xticks([])
            ax.set_xlabel('Predicted Class')
            ax.set_xticks(np.arange(2)+0.5)
            ax.set_xticklabels(['Hold', 'Reach'])

            #Colorbar: 
            f.subplots_adjust(right=0.8)
            cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
            f.colorbar(c, cax=cbar_ax)

            ax.set_title('Confusion Matrices for Training on '+train+' , Test on '+test+': '+day)
            plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/grom_spk_v2_may2016/mcmc_conf_'+day+'.eps', format='eps', dpi=300)

def bar_plots_for_conf_dict(CONF, days):
    ''' figure: unconsolid and consolid'''
    f, ax = plt.subplots(nrows=4, ncols=2)
    f2, ax2 = plt.subplots()

    color_dict=dict(hold='royalblue', reach='navy')
    for i_d, day in enumerate(days):

        #First train on nf, test MC: 
        conf_mat = CONF[day]['nf', 'mc']
        conf_mat2 = CONF[day]['mc', 'nf']

        cnt = 0
        cnt2 = 0

        cnt3 = 0
        cnt4 = 0
        for cat in range(6):
            ax[i_d, 0].bar(cat, conf_mat[4, cat], color=color_dict['hold'], width=0.3)
            ax[i_d, 0].bar(cat+.4, conf_mat[5, cat], color=color_dict['reach'], width=0.3)
            if cat < 4:
                cnt += (cat+1)*conf_mat[4, cat]
                cnt2 += (cat+1)*conf_mat[5, cat]

            ax[i_d, 1].bar(cat, conf_mat2[cat, 4], color=color_dict['hold'], width=0.3)
            ax[i_d, 1].bar(cat+0.4, conf_mat2[cat, 5], color=color_dict['reach'], width=0.3)
            if cat < 4:
                cnt3 += (cat+1)*conf_mat2[cat, 4]
                cnt4 += (cat+1)*conf_mat2[cat, 5]
                
        for j in [0, 1]:
            ax[i_d, j].set_xticks(np.arange(6)+0.5)
            ax[i_d, j].set_xticklabels(['Beta1', 'Beta2', 'Beta3', 'Beta4', 'Hold', 'Reach'])

        ax[i_d, 0].set_title('Train NF, Pred MC: '+day)
        ax[i_d, 0].set_ylabel('Perc.')

        ax[i_d, 1].set_title('Train MC, Pred NF: '+day)
        ax[i_d, 1].set_ylabel('Perc.')

        #Average beta target for train NF, test MC: 
        ax2.plot([0, 1], [cnt, cnt2], 'b.-')
        ax2.plot([0, 1], [cnt3/np.sum(conf_mat2[:4, 4]), cnt4/np.sum(conf_mat2[:4, 5])], 'r.-')
    ax2.set_xlim([-0.5, 1.5])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Hold', 'Reach'])
    ax2.set_ylabel('Avg. NF Target (1-4)')
    ax2.set_title('Train NF & Predict MC (Blue), Train MC & Predict NF (Red)')
    f.tight_layout()
    f2.tight_layout()
    
def six_class_LDA(test=True, within_task_comparison=True, x_task_comparison=False, all_cells = False, animal='grom'):
    
    #6 class LDA: 
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = get_unbinned(test=test, all_cells=all_cells, animal=animal)
    
    # 100 ms bin size: 
    b = 100
    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_sig = get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, b, smooth = 50,  animal=animal)
    #New -- binary kin signal

    #Smoosh together binary kin signals
    K_bin_master = {}
    for i_d, day in enumerate(days):
        kbn = []
        for i_b, b in enumerate(blocks[i_d]):
            kbn.append(bin_kin_signal[b, day])
        K_bin_master[day] = np.vstack((kbn))

    b = 100
    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
        beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=b, animal=animal)
    

    CONF = dict()
    for i_d, day in enumerate(days):
        K_bin = K_bin_master[day]
        #K_bin_mc = K_bin_master[day]

        #Ix : [-500ms, +1000ms]
        #ix_ = np.ix_(np.arange(K_bin.shape[0]), np.arange(10, 25))
        #K_bin = K_bin[ix_]

        #Same time slice:
        X = S_bin[day]#[:, 10:25, :]
        #X_mc = S_bin[day].copy()

        classes = np.zeros_like(X[:, :, 0])

        #Label LFP indices: (1, 2, 3, 4)
        for i in [84, 85, 86, 87]:
            ix_ = np.nonzero(Params[day]['day_lfp_labs']==i)[0]
            print ix_.shape, i
            classes[ix_, :15] = i - 83


        for i in range(classes.shape[0]):
            for j in range(classes.shape[1]):
                #Unlabeled: 
                if classes[i, j] == 0:
                    #If abs(kin) > 0.5: 
                    if K_bin[i, j]:
                        classes[i, j] = 6 #REACH
                    else:
                        classes[i, j] = 5 #HOLD
        
        
        ix1=np.nonzero(Params[day]['day_lfp_labs']>80)[0]
        ix2 = np.nonzero(Params[day]['day_lfp_labs']<80)[0]

        #USE only -500 ms : 1000 ms for NF, use -1500 ms : 1000 ms for manual control (limited data issue)
        tix1 = np.arange(10, 25)
        tix2 = np.arange(25)

        master_ix = [ix1, ix2]
        master_t_ix = [tix1, tix2]
        master_nm = ['nf', 'mc']

        conf_day = {}
        for i, (ix_, tix_) in enumerate(zip(master_ix, master_t_ix)):
            
            #Train:      
            magic_ix = np.ix_(ix_, tix_, np.arange(X.shape[2]))    
            magic_ix2 = np.ix_(ix_, tix_)
            x1 = X[magic_ix].reshape(len(ix_)*len(tix_), X.shape[2])
            y1 = classes[magic_ix2].reshape(-1)

            #For training on MC-- make sure enough hold data. Add more if needed (replication): 
            if i == 1:
                hold_ix = np.nonzero(y1==5)[0]
                fact = int(len(y1)/float(len(hold_ix))) - 1
                cat_y = np.zeros((len(hold_ix)*fact, )) + 5
                cat_x = np.tile(x1[hold_ix, :], [fact, 1])
                x1 = np.vstack((x1, cat_x))
                y1 = np.hstack((y1, cat_y))

            clf = sklearn.lda.LDA()
            ch_clf = sklearn.lda.LDA()

            if within_task_comparison:
                #Xvalidate classifer by training on 80%, test on 20% of data: 

                ix_xval = np.random.permutation(len(y1))
                sub_tmp = np.zeros((6, 6))
                n_corr = 0
                n_tot = 0

                n_ch_corr = 0
                n_ch_tot = 0
                for xval in range(5):
                    ix_test = ix_xval[(xval*len(y1)/5):((xval+1)*len(y1)/5)]
                    ix_train = np.array([ii for ii in range(len(y1)) if ii not in ix_test])
                    
                    #Fit w/ training data: 
                    clf.fit(x1[ix_train, :], y1[ix_train])
                    ch_clf.fit(x1[np.random.permutation(ix_train), :], y1[ix_train])

                    #Predict held-out data: 
                    y_hat = clf.predict(x1[ix_test, :])
                    y_hat_ch = ch_clf.predict(x1[np.random.permutation(ix_test), :])

                    #Get actual held-out data: 
                    y2 = y1[ix_test]

                    #Add to confusion matrix: 
                    for iii, (yi, y_hat_i) in enumerate(zip(y2, y_hat)):
                        sub_tmp[yi-1, y_hat_i-1] += 1
                        if yi == y_hat_i:
                            n_corr += 1

                        if yi == y_hat_ch[iii]:
                            n_ch_corr += 1

                        n_tot += 1

                for iii in range(6):
                    sub_tmp[iii,:] = sub_tmp[iii, :]/float(np.sum(sub_tmp[iii,:]))
                
                conf_day[master_nm[i], master_nm[i], 'xval'] = sub_tmp
                conf_day[master_nm[i], master_nm[i], 'perc_corr'] = (n_corr, n_tot)
                conf_day[master_nm[i], master_nm[i], 'chance_perc_corr'] = (n_ch_corr, n_tot)

            elif x_task_comparison:
                # magic_ix_nf = np.ix_(ix1, np.arange(10, 15), np.arange(X.shape[2]))
                # magic_ix_nf2 = np.ix_(ix1, np.arange(10, 15))
                magic_ix_nf = np.ix_(ix1, np.arange(25), np.arange(X.shape[2]))
                magic_ix_nf2 = np.ix_(ix1, np.arange(25))

                magic_ix_mc = np.ix_(ix2, np.arange(25), np.arange(X.shape[2]))
                magic_ix_mc2 = np.ix_(ix2, np.arange(25),)

                x1_nf = X[magic_ix_nf].reshape(len(ix1)*len(np.arange(25)), X.shape[2])
                y1_nf = classes[magic_ix_nf2].reshape(-1)

                x1_mc = X[magic_ix_mc].reshape(len(ix2)*25, X.shape[2])
                y1_mc = (classes[magic_ix_mc2].reshape(-1)) #Make MC labels from 5, 6  (hold, reach) --> 0, 1 (hold, reach)

                clf.fit(x1_nf, y1_nf)
                y_train_nf_fit_mc = clf.predict(x1_mc)

                nf_trn = np.zeros((6, 6))
                for iii, (yi_mc, yi_nf_clf) in enumerate(zip(y1_mc, y_train_nf_fit_mc)):
                    nf_trn[yi_mc-1, yi_nf_clf-1] += 1

                clf.fit(x1_mc, y1_mc)
                y_train_mc_fit_nf = clf.predict(x1_nf)

                mc_trn = np.zeros((6, 6))
                for iii, (yi_nf, yi_mc_clf) in enumerate(zip(y1_nf, y_train_mc_fit_nf)):
                    mc_trn[yi_nf-1, yi_mc_clf-1] += 1

                conf_day['mc', 'nf', i] = mc_trn
                conf_day['nf', 'mc', i] = nf_trn

            else:

                for j, jx_ in enumerate(zip(master_ix, master_t_ix)):

                    #Test: 
                    magic_jx = np.ix_(jx_, tix_, np.arange(X.shape[2]))    
                    magic_jx2 = np.ix_(jx_, tix_)

                    x2 = X[magic_jx].reshape(len(jx_)*len(tix_), X.shape[2])
                    y2 = classes[magic_jx2].reshape(-1)
                    y_hat = clf.predict(x2)

                    tmp = np.zeros((6, 6))
                    for iii, (yi, y_hat_i) in enumerate(zip(y2, y_hat)):
                        tmp[yi-1, y_hat_i -1] += 1

                    for iii in range(6):
                        tmp[iii,:] = tmp[iii, :]/float(np.sum(tmp[iii,:]))
                    
                    conf_day[master_nm[i], master_nm[j]] = tmp
        CONF[day] = conf_day

    if all_cells:
        sufx = '_all_cells'
    else:
        sufx = '_select_cells'

    if within_task_comparison:
        import pickle
        pickle.dump(CONF, open('six_class_LDA_within_task_xval5'+sufx+'.pkl', 'wb'))
    elif x_task_comparison:
        import pickle
        pickle.dump(CONF, open('six_class_LDA_x_task_compare_all6classes'+sufx+'.pkl', 'wb'))
    else:
        raise Exception

def six_class_LDA_by_day(day, test=True, within_task_comparison=True, x_task_comparison=False, 
    all_cells = False, animal='grom'):
    
    #6 class LDA: 
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = get_unbinned(test=test, all_cells=all_cells, animal=animal, days=[day])
    
    # 100 ms bin size: 
    b = 100
    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_sig = get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, b, smooth = 50,  animal=animal)
    #New -- binary kin signal

    #Smoosh together binary kin signals
    K_bin_master = {}
    KC_bin_master = {}
    for i_d, day in enumerate(days):
        kbn = []
        kbn_c = []
        for i_b, b in enumerate(blocks[i_d]):
            kbn.append(bin_kin_signal[b, day])
            kbn_c.append(binned_kin_signal_dict[b, day])
        K_bin_master[day] = np.vstack((kbn))
        KC_bin_master[day] = np.vstack((kbn_c))

    b = 100
    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
        beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=b, animal=animal)
    

    CONF = dict()

    K_bin = K_bin_master[day]
    X = S_bin[day]
    classes = np.zeros_like(X[:, :, 0])

    #Label LFP indices: (1, 2, 3, 4)
    for i in [84, 85, 86, 87]:
        ix_ = np.nonzero(Params[day]['day_lfp_labs']==i)[0]
        print ix_.shape, i
        classes[ix_, :15] = i - 83


    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
            #Unlabeled: 
            if classes[i, j] == 0:
                #If abs(kin) > 0.5: 
                if K_bin[i, j]:
                    classes[i, j] = 6 #REACH
                else:
                    classes[i, j] = 5 #HOLD
    
        
    ix1=np.nonzero(Params[day]['day_lfp_labs']>80)[0]
    ix2 = np.nonzero(Params[day]['day_lfp_labs']<80)[0]

    #USE only -500 ms : 1000 ms for NF, use -1500 ms : 1000 ms for manual control (limited data issue)
    tix1 = np.arange(10, 25)
    tix2 = np.arange(25)

    master_ix = [ix1, ix2]
    master_t_ix = [tix1, tix2]
    master_nm = ['nf', 'mc']

    conf_day = {}
    for i, (ix_, tix_) in enumerate(zip(master_ix, master_t_ix)):
        
        #Train:      
        magic_ix = np.ix_(ix_, tix_, np.arange(X.shape[2]))    
        magic_ix2 = np.ix_(ix_, tix_)
        x1 = X[magic_ix].reshape(len(ix_)*len(tix_), X.shape[2])
        y1 = classes[magic_ix2].reshape(-1)

        #For training on MC-- make sure enough hold data. Add more if needed (replication): 
        if i == 1:
            hold_ix = np.nonzero(y1==5)[0]
            fact = int(len(y1)/float(len(hold_ix))) - 1
            cat_y = np.zeros((len(hold_ix)*fact, )) + 5
            cat_x = np.tile(x1[hold_ix, :], [fact, 1])
            x1 = np.vstack((x1, cat_x))
            y1 = np.hstack((y1, cat_y))

        clf = sklearn.lda.LDA()
        ch_clf = sklearn.lda.LDA()

        if within_task_comparison:
            #Xvalidate classifer by training on 80%, test on 20% of data: 

            ix_xval = np.random.permutation(len(y1))
            sub_tmp = np.zeros((6, 6))
            n_corr = 0
            n_tot = 0

            n_ch_corr = 0
            n_ch_tot = 0
            for xval in range(5):
                ix_test = ix_xval[(xval*len(y1)/5):((xval+1)*len(y1)/5)]
                ix_train = np.array([ii for ii in range(len(y1)) if ii not in ix_test])
                
                #Fit w/ training data: 
                clf.fit(x1[ix_train, :], y1[ix_train])
                ch_clf.fit(x1[np.random.permutation(ix_train), :], y1[ix_train])

                #Predict held-out data: 
                y_hat = clf.predict(x1[ix_test, :])
                y_hat_ch = ch_clf.predict(x1[np.random.permutation(ix_test), :])

                #Get actual held-out data: 
                y2 = y1[ix_test]

                #Add to confusion matrix: 
                for iii, (yi, y_hat_i) in enumerate(zip(y2, y_hat)):
                    sub_tmp[yi-1, y_hat_i-1] += 1
                    if yi == y_hat_i:
                        n_corr += 1

                    if yi == y_hat_ch[iii]:
                        n_ch_corr += 1

                    n_tot += 1

            for iii in range(6):
                sub_tmp[iii,:] = sub_tmp[iii, :]/float(np.sum(sub_tmp[iii,:]))
            
            conf_day[master_nm[i], master_nm[i], 'xval'] = sub_tmp
            conf_day[master_nm[i], master_nm[i], 'perc_corr'] = (n_corr, n_tot)
            conf_day[master_nm[i], master_nm[i], 'chance_perc_corr'] = (n_ch_corr, n_tot)

        elif x_task_comparison:
            # magic_ix_nf = np.ix_(ix1, np.arange(10, 15), np.arange(X.shape[2]))
            # magic_ix_nf2 = np.ix_(ix1, np.arange(10, 15))
            magic_ix_nf = np.ix_(ix1, np.arange(25), np.arange(X.shape[2]))
            magic_ix_nf2 = np.ix_(ix1, np.arange(25))

            magic_ix_mc = np.ix_(ix2, np.arange(25), np.arange(X.shape[2]))
            magic_ix_mc2 = np.ix_(ix2, np.arange(25),)

            x1_nf = X[magic_ix_nf].reshape(len(ix1)*len(np.arange(25)), X.shape[2])
            y1_nf = classes[magic_ix_nf2].reshape(-1)

            x1_mc = X[magic_ix_mc].reshape(len(ix2)*25, X.shape[2])
            y1_mc = (classes[magic_ix_mc2].reshape(-1)) #Make MC labels from 5, 6  (hold, reach) --> 0, 1 (hold, reach)

            clf.fit(x1_nf, y1_nf)
            y_train_nf_fit_mc = clf.predict(x1_mc)

            nf_trn = np.zeros((6, 6))
            for iii, (yi_mc, yi_nf_clf) in enumerate(zip(y1_mc, y_train_nf_fit_mc)):
                nf_trn[yi_mc-1, yi_nf_clf-1] += 1

            clf.fit(x1_mc, y1_mc)
            y_train_mc_fit_nf = clf.predict(x1_nf)

            mc_trn = np.zeros((6, 6))
            for iii, (yi_nf, yi_mc_clf) in enumerate(zip(y1_nf, y_train_mc_fit_nf)):
                mc_trn[yi_nf-1, yi_mc_clf-1] += 1

            conf_day['mc', 'nf', i] = mc_trn
            conf_day['nf', 'mc', i] = nf_trn

        else:

            for j, jx_ in enumerate(zip(master_ix, master_t_ix)):

                #Test: 
                magic_jx = np.ix_(jx_, tix_, np.arange(X.shape[2]))    
                magic_jx2 = np.ix_(jx_, tix_)

                x2 = X[magic_jx].reshape(len(jx_)*len(tix_), X.shape[2])
                y2 = classes[magic_jx2].reshape(-1)
                y_hat = clf.predict(x2)

                tmp = np.zeros((6, 6))
                for iii, (yi, y_hat_i) in enumerate(zip(y2, y_hat)):
                    tmp[yi-1, y_hat_i -1] += 1

                for iii in range(6):
                    tmp[iii,:] = tmp[iii, :]/float(np.sum(tmp[iii,:]))
                
                conf_day[master_nm[i], master_nm[j]] = tmp

    if all_cells:
        sufx = '_all_cells_'+day
    else:
        sufx = '_select_cells_'+day

    if within_task_comparison:
        import pickle
        pickle.dump(conf_day, open('six_class_LDA_within_task_xval5'+sufx+'.pkl', 'wb'))
    if x_task_comparison:
        import pickle
        pickle.dump(conf_day, open('six_class_LDA_x_task_compare_all6classes'+sufx+'.pkl', 'wb'))
    else:
        raise Exception
    return conf_day

def six_class_LDA_by_day_within(day):
    conf_day = six_class_LDA_by_day(day, test=True, within_task_comparison=True, 
        x_task_comparison=False, all_cells = False, animal='cart')
    return conf_day

def six_class_LDA_by_day_across(day):
    conf_day = six_class_LDA_by_day(day, test=True, within_task_comparison=False, 
        x_task_comparison=True, all_cells = False, animal='cart')
    return conf_day

if __name__ ==  "__main__":
    days = ['011315', '011415', '011515', '011615']
    pool1 = mp.Pool()
    results1 = pool1.map(six_class_LDA_by_day_within, days)
    CONF = {}
    for i_d, day in enumerate(days):
        CONF[day] = results1[i_d]
    pickle.dump(CONF, open('six_class_LDA_within_task_xval5_all_cells_mp_attempt.pkl', 'wb'))

    print '####### within --> across days ########'
    print '####### within --> across days ########'
    print '####### within --> across days ########'
    print '####### within --> across days ########'

    days = ['011315', '011415', '011515', '011615']
    pool2 = mp.Pool()
    results2 = pool2.map(six_class_LDA_by_day_across, days)
    CONF = {}
    for i_d, day in enumerate(days):
        CONF[day] = results2[i_d]
    pickle.dump(CONF, open('six_class_LDA_x_task_compare_all6classes_all_cells_mp_attempt.pkl', 'wb'))

    # lda = False
    # if lda: 
    #     print 'LDA beta: '
    #     run_LDA_beta('SVM')
    # else:
    #     import gc
    #     #binsizes = [5, 10, 20, 40, 50, 70, 100] #Numbers that 2500 is divisible by!
    #     binsizes = [5, 10, 25, 50, 75, 100]
    #     keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = get_unbinned(test=False)

    #     for b in binsizes:
    #         print 'starting bin: ', b
    #         # #Bin / smooth spike and beta dictionaries
    #         B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
    #             beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=b)

    # #           #Get kin signal
    #         kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt = get_kin(days, 
    #             blocks, bef, aft, go_times_dict, lfp_lab, b, smooth = 50)

    #         #kin_signal_dict = {}
    #         #binned_kin_signal_dict = {}

    #         for predict in ['beta']: #['kin', 'beta']:
    #             main_process( kin_signal_dict, binned_kin_signal_dict, B_bin, BC_bin, S_bin, Params, days, blocks, mc_indicator, lfp_lab,
    #                 b, train_trial_type='mc', test_trial_type='beta', spike_model='ssm_filters',predict=predict)

    #             gc.collect()

    #With multiunits:
    # keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = get_unbinned(test=False)
    # get_kf_trained_from_full_mc(keep_dict, days, blocks, mc_indicator)
    #six_class_LDA()

# B_unsmooth_mc = B_unsmooth[mc_trials,:]
# B_unsmooth_beta = B_unsmooth[beta_trials,:]

# i+= 10
# for i in range(i, i+10):
#     plt.plot(np.array(beta_kin[i,:]), 'b-')
#     plt.plot(Y_beta_mat[i,:],'k-')
#     for j in range(len(B_unsmooth_beta[i,:])):
#         if B_unsmooth_beta[i, j]:
#             plt.plot([j-.3, j+.3], [0, 0], 'r.-', markersize=20)
#     print lfp_lab_bcd[i]
#     plt.show()

# i=0

# i+= 10
# for i in range(i, i+10):
#     plt.plot(np.array(y_pred_resh[i,:]), 'b-')
#     plt.plot(Y_down[i,:],'k-')
#     for j in range(len(B_mc[i,:])):
#         if B_mc[i, j]:
#             plt.plot([j-.3, j+.3], [0, 0], 'r.-', markersize=20)
#     plt.show()


# i=0
# i+=10
# for i in range(i, i+10):
#     plt.plot(np.array(y_pred_resh_beta[i,:]), 'b-')
#     plt.plot(Y_beta_down[i,:],'k-')
#     for j in range(len(B_beta[i,:])):
#         if B_beta[i, j]:
#             plt.plot([j-.3, j+.3], [0, 0], 'r.-', markersize=20)
#     plt.show()

#     labs = np.hstack((lfp_lab['b'], lfp_lab['c'], lfp_lab['d']))

### Metrics: 

## First, which target is most accurate? 


