import predict_kin_w_spk
import state_space_w_beta_bursts as ssbb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import load_files
import seaborn
import scipy.signal
import scipy.stats
import sklearn
import pickle
import datetime
from scipy import ndimage
import glob
import state_space_spks as sss

def train_test_mc_decoders():
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=True)
    predict_kin_w_spk.get_kf_trained_from_full_mc(keep_dict, days, blocks, mc_indicator)

def train_mc_full_test_nf_full(test=False, plot=False, day='022815'):
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, days=[day])
    print 'decoder dict starting for ', day, ' using days: ', days
    decoder_dict = predict_kin_w_spk.get_kf_trained_from_full_mc(keep_dict, days, blocks, mc_indicator, decoder_only=True, kin_type='endpt')
    import pickle
    pickle.dump(decoder_dict, open(day+'_decoder_dict.pkl', 'wb'))
    print 'saving decoder dict'

    b = 25
    spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = predict_kin_w_spk.get_full_blocks(keep_dict, days, blocks, mc_indicator, kin_type='endpt', mc_only=False)

    #Smooth kinematics: 
    kin_smooth_len = 151
    kin_smooth_std = 51
    window = scipy.signal.gaussian(kin_smooth_len, std=kin_smooth_std)

    window = window/np.sum(window)
    smooth_kin_dict = {}
    for k in kin_dict.keys():
        smooth_kin_dict[k] = np.zeros_like(kin_dict[k])
        for j in range(4):
            smooth_kin_dict[k][:, j] = np.convolve(window, kin_dict[k][:,j], mode='same')

    #Only for single day now (test=True)

    for binsize in [25]: #[5, 10, 25, 50, 100]:
        #Bin stuff: 
        Bin_K = {}
        Bin_N = {}
        Bin_B = {}
        Bin_H = {}
        for k in spk_dict.keys():
            Bin_N[k] = {}
            for u in spk_dict[k].keys():
                Bin_N[k][u] = predict_kin_w_spk.bin_(spk_dict[k][u], binsize, mode='cnts')
            print 'done binning for neural :', k
            Bin_K[k] = predict_kin_w_spk.bin_(smooth_kin_dict[k], binsize, mode='mean')
            print 'done binning for kin :', k
            Bin_B[k] = predict_kin_w_spk.bin_(beta_dict[k][1,:], binsize, mode='mode')
            #Bin_H[k] = predict_kin_w_spk.bin_(hold_dict[k], binsize, mode='mode')
            print 'done binning for beta :', k

        fnames = []
        for i_d, day in enumerate(days):
            day_kin = {}
            day_pred_kin = {}
            day_pred_kin_lpf = {}

            for i_b, blk in enumerate(blocks[i_d]):
                #Predict each day and block!

                k = (blk, day)
                mat_Bin_N_tmp = []

                for u in Bin_N[k]:
                    mat_Bin_N_tmp.append(Bin_N[k][u])

                obs = np.hstack((mat_Bin_N_tmp))
                kin = Bin_K[k]

                if obs.shape[0] != kin.shape[0]:
                    mx = np.min([obs.shape[0], kin.shape[0]])
                    print 'obs, kin mismatch: ', obs.shape[0], kin.shape[0]
                    obs = obs[:mx, :]
                    kin = kin[:mx, :]
                else:
                    mx = obs.shape[0]


                z = np.zeros((len(kin), ))
                kin_vel = kin[:, [2,3]]
                kin_full = np.vstack((kin[:, 0], z, kin[:, 1], kin[:, 2], z, kin[:,3])).T

                try:
                    k_new = (str(binsize), k[1], 'a', 'kf')
                    filter_ = decoder_dict[k_new]
                except:
                    k_new = (str(binsize), k[1], 'b', 'kf')
                    filter_ = decoder_dict[k_new]

                filter_._init_state()
                s = filter_.state
                nbins, n_ft = kin.shape

                st = []

                #Cycle through bins
                for t in range(nbins):
                    obs_ = obs[t, :][:, np.newaxis]
                    s = filter_._forward_infer(s, obs_)
                    st.append(s.mean)
                pred_kin = np.array(np.hstack((st))).T
                day_pred_kin[day, blk] = pred_kin

                if plot:
                    f, ax = plt.subplots(nrows = 2, ncols=2)
                    ax[0,0].plot(pred_kin[:, 3], label='pred x vel')
                    ax[0,0].plot(kin_full[:, 3], label='act x vel')
                    ax[0,0].legend()
                    ax[0,0].set_title('Binsize: '+str(binsize)+' Model: '+model+' Key: '+k[1]+k[0])
                    ax[1,0].plot(pred_kin[:, 5], label='pred y vel')
                    ax[1,0].plot(kin_full[:, 5], label='act x vel')
                    ax[1,0].legend()

                #Calc R2 after 1 minute in to -1 min out: 
                n_samp_buffer = int(60*1000./binsize)
                #y_hat = pred_kin[n_samp_buffer:-1*(n_samp_buffer+1250), [3, 5]]            
                #y = kin_full[n_samp_buffer:-1*(n_samp_buffer+1250), [3, 5]]
                y_hat = pred_kin[:, [3, 5]]
                y = kin_full[:, [3, 5]]
                day_kin[day, blk] = y

                beta_sig = Bin_B[k][0, :mx]
                day_kin[day, blk, 'beta_sig'] = beta_sig

                kin_sig = Bin_K[k][:mx, [2, 3]]
                #hold_sig = Bin_H[k][0, :mx]

                #SLOW vs. FAST cutoff
                spd = np.sqrt(kin_sig[:, 0]**2 + kin_sig[:, 1]**2)        
                binary_kin_sig = np.array([int(spd[i] > 3.5) for i in range(len(spd))])
                day_kin[day, blk, 'binary_kin_sig'] = binary_kin_sig

                y_mean = np.tile(np.mean(y, axis=0)[np.newaxis, :], [y.shape[0], 1])
                R2_ = 1 - (np.sum((y - y_hat)**2)/np.sum((y - y_mean)**2))
                day_pred_kin[day, blk, 'r2'] = R2_

                ### LPF STUFF ### 
                lpf_window = scipy.signal.gaussian(kin_smooth_len, std=kin_smooth_std)
                t_ix = np.arange(0, kin_smooth_len, binsize)
                lpf_window = lpf_window[t_ix]
                lpf_window = lpf_window/np.sum(lpf_window)

                lpf_x_vel = np.convolve(lpf_window, y_hat[:, 0], mode='same')
                lpf_y_vel = np.convolve(lpf_window, y_hat[:, 1], mode='same')
                y_hat_lpf = np.vstack((lpf_x_vel, lpf_y_vel)).T

                R2_lpf_ = 1 - (np.sum((y - y_hat_lpf)**2, axis=0)/np.sum((y - y_mean)**2, axis=0))
                R2_lpf_sum = 1 - (np.sum((y - y_hat_lpf)**2)/np.sum((y - y_mean)**2))
                
                day_pred_kin_lpf[day, blk] = y_hat_lpf
                day_pred_kin_lpf[day, blk, 'r2'] = R2_lpf_sum
                

            # n, wn = scipy.signal.buttord(.75/40., .5/40, 3, 20)
            # b, a  = scipy.signal.butter(n, wn, 'highpass')
            # dat_filt = scipy.signal.filtfilt(b, a, y_hat_lpf, axis=0)
            # R2_hpf = 1 - (np.sum((y - dat_filt)**2)/np.sum((y - y_mean)**2))

                if plot:
                    ax[0, 1].plot(y_hat_lpf[:, 0], label='pred x vel')
                    ax[0, 1].plot(y[:, 0], label='act x vel')
                    # ax[0, 1].plot(dat_filt[:, 0], label='lpf_hpf_xvel')
                    ax[0, 1].plot(20+ (2*beta_sig), 'r-')
                    ax[0, 1].legend()
                    ax[0, 1].set_title('Binsize: '+str(binsize)+' Model: '+model+' Key: '+k[1]+k[0])
                    ax[1, 1].plot(y_hat_lpf[:, 1], label='pred y vel')
                    ax[1, 1].plot(y[:, 1], label='act y vel')
                    # ax[1, 1].plot(dat_filt[:, 1], label='lpf_hpf_yvel')
                    ax[1, 1].plot(20+ (2*beta_sig), 'r-')
                    ax[1, 1].legend()

            ### Flag hold periods, kin < 3.0, kin > 3.0 ###
                if plot:
                    f2, ax2 = plt.subplots(nrows=2, ncols=4)
                    f3, ax3 = plt.subplots()
                    hat = ['', '/']
                    col = ['b', 'g']

                stat_dict = {}
                #hold_ix = np.nonzero(hold_sig)[0]
                slow_ix = np.nonzero(binary_kin_sig==0)[0]
                fast_ix = np.nonzero(binary_kin_sig==1)[0]
                all_ix = np.arange(len(binary_kin_sig))

                master_ix = [slow_ix, fast_ix]
                master_nm =['slow', 'fast']
                for ii, (ix_, nm_) in enumerate(zip(master_ix, master_nm)):
                    y_ = [y[ix_, 0], y[ix_, 1]]
                    y_hat_ = [y_hat_lpf[ix_, 0], y_hat_lpf[ix_, 1]]
                    y_mn_ = [np.mean(y_hat_lpf[:, 0]), np.mean(y_hat_lpf[:, 1])]
                    beta_sub_sig = beta_sig[ix_]

                    r2 = 1 - ((np.sum((y_[0] - y_hat_[0])**2) + (np.sum((y_[1] - y_hat_[1])**2)))/(np.sum((y_[0] - y_mn_[0])**2)+np.sum((y_[1] - y_mn_[1])**2)))
                    mse = np.sum((y_[0]-y_hat_[0])**2) + np.sum((y_[1]-y_hat_[1])**2)/float(len(y_[1]))
                    spd = np.sqrt(y_[0]**2 + y_[1]**2)
                    spd_pred = np.sqrt(y_hat_[0]**2 + y_hat_[1]**2)


                    for iv in [0, 1]:
                        k, p = scipy.stats.ks_2samp(y_[iv], y_hat_[iv])
                        t, p2 = scipy.stats.ttest_ind(y_[iv], y_hat_[iv])

                        print nm_, 'vel: ', iv, ' ttest: ', p2, ' kstest: ', p, ' R2: ', r2, 'MSE: ', mse

                        b0 = np.nonzero(beta_sub_sig==0)[0]
                        b1 = np.nonzero(beta_sub_sig==1)[0]
                        bix = [b0, b1]

                        for ib, bt in enumerate(bix):
                            r2_ = 1 - (np.sum((y_[iv][bt] - y_hat_[iv][bt])**2)/np.sum((y_[iv][bt] - np.mean(y_[iv][bt]))**2))
                            mse_ = (np.sum((y_[iv][bt] - y_hat_[iv][bt])**2))/float(len(y_[iv][bt]))
                            print 'beta: ', ib, ' r2: ', r2_, ' length: ', len(y_[iv][bt]), ' mse: ', mse_
                            if plot:
                                ax2[iv, (2*ii)+ib].bar(0, np.mean(np.abs(y_[iv][bt])), color='b')
                                ax2[iv, (2*ii)+ib].bar(1, np.mean(np.abs(y_hat_[iv][bt])), color='g')
                                ax2[iv, (2*ii)+ib].set_title('Beta:'+str(ib)+nm_+' vel:'+str(iv))
                                ax2[iv, (2*ii)+ib].set_ylim([0, 7])

                            if ii == 0:
                                stat_dict[ib, 'act']= spd[bt]
                                stat_dict[ib, 'pred']= spd_pred[bt]
                                if plot:
                                    ax3.bar(0+ib, np.mean(spd[bt]), color='b', yerr=np.std(spd[bt])/np.sqrt(len(bt)), hatch=hat[ib])
                                    ax3.bar(2+ib, np.mean(spd_pred[bt]), color='g', yerr=np.std(spd_pred[bt])/np.sqrt(len(bt)), hatch=hat[ib])
                                    ax3.set_ylabel('Speed (cm/s)')
                                    ax3.set_xlim([-.5, 4.5])
                                    ax3.set_xticks(np.arange(4)+.5)
                                    ax3.set_xticklabels(['Actual, Beta Off', 'Actual, Beta On', 'Pred, Beta Off', 'Pred, Beta On'])
            import pickle
            master_dict = dict(act=day_kin, pred=day_pred_kin, pred_lpf=day_pred_kin_lpf)        
            dt = datetime.datetime.now()
            dt_ = dt.isoformat()[:16]
            pickle.dump(master_dict, open(dt_+day+'_'+str(binsize)+'_cts_KF.pkl', 'wb'))
            fnames.append(dt_+day+'_'+str(binsize)+'_cts_KF.pkl')
    return fnames, days, mc_indicator

    # lpf_spd = np.sqrt(lpf_x_vel**2 + lpf_y_vel**2)
    # act_spd = np.sqrt(y[:,0]**2 + y[:,1]**2)
    # hold_periods = Bin_H[k][0, n_samp_buffer:-n_samp_buffer]

    # hold_ix = np.nonzero(binary_kin_sig==0)[0]
    # beta_hold = beta_sig[hold_ix]

    # #y_hold = y.copy()
    # #y_pred = y_hat_lpf.copy()

    # y_hold = y[hold_ix, :]
    # y_pred = y_hat_lpf[hold_ix, :]

    # for beta_ix_ in [0, 1]:
    #     ix = np.nonzero(beta_hold==beta_ix_)[0]
    #     print 'beta ix: ', len(ix)
        
    #     for vel in [0, 1]:
    #         r2_ = 1 - (np.sum((np.abs(y_pred[ix, vel]) - np.abs(y_hold[ix, vel]))**2)/(np.sum((np.abs(y_hold[ix, vel])- np.abs(np.mean(y_hold[ix, vel])))**2)))
    #         R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel] = r2_

    #         MN = np.mean(np.abs(y_pred[ix, vel]) - np.abs(y_hold[ix, vel]))
    #         print 'beta ix: ', beta_ix_, ' vel: ', vel, MN,

    #         R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel, 'mean_abs_err'] = MN
    #         R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel, 'y_pred'] = y_pred[ix, vel]
    #         R2_lpf_beta[binsize, k[1], k[0], model, beta_ix_, vel, 'y_act'] = y_hold[ix, vel]
    # R2_lpf_beta[binsize, k[1], k[0], model, 'full_R2_lpf'] = R2_lpf_
    # R2_lpf_beta[binsize, k[1], k[0], model, 'y_hat_lpf'] = y_hat_lpf
    # R2_lpf_beta[binsize, k[1], k[0], model, 'y'] = y

def open_plot_train_mc_full_test_nf_full(fname, mc_indicator_day, blocks_day, ax1=None, ax2=None, save=True, day=None):
    if fname is None:
        srch = '2016-08-27*'+day+'*'
        fnm = glob.glob(srch)
        if len(fnm)==1:
            fname = fnm[0]
        else:
            raise Exception

    if ax1 is None and ax2 is None:
        f, ax = plt.subplots(nrows=3)
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]

    dat = pickle.load(open(fname))

    #Get block names from file: 
    k = []
    for ik in dat['pred'].keys():
        if len(ik) == 2:
            k.append(ik)
    k = np.vstack((k))
    blocks = np.sort(np.unique(k[:, 1]))
    day = k[0,0]

    #For each block, find the hold periods from the NF cue: 
    tms_dict = dict()
    beta_dict = dict()


    try:
        fdict = pickle.load(open(fname[:-4]+'_tms_and_beta.pkl'))
        tms_dict = fdict['tms_dict']
        beta_dict = fdict['beta_dict']
        make_beta_file = False
    except:
        make_beta_file = True
    
    for i_b, blk in enumerate(blocks):
        if not make_beta_file: 
            b = get_beta(np.squeeze(beta_dict[blk, 'raw_ad74']), bp_filt=[13, 25])
            beta_dict[blk] = b
        else:
            f = load_files.load(blk, day)
            Strobed=f['Strobed']

            #Key offset: 
            ts_key = 'AD33_ts'
            offset = f[ts_key]

            t = np.arange(len(np.squeeze(f['AD33'])))
            t_sub = np.arange(0, len(t), 25)

            tms = []
            if np.any(np.array(offset.shape) > 1): #Any size is nonzero
                offset = np.squeeze(offset)
                lfp_offset = offset[1] #Second one is the LFP offset
            else:
                lfp_offset = offset[0, 0]

            rew_ix = np.nonzero(Strobed[:, 1]==9)[0]
            for ir in rew_ix:
                rev_strb = Strobed[:ir, 1]
                try:
                    ix = np.max(np.nonzero(rev_strb==15)[0])
                    tm1 = (Strobed[ix, 0]-lfp_offset)*1000
                    ix1 = np.nonzero(t_sub <= tm1)[0][-1]

                    tm2 = (Strobed[ir-3, 0]-lfp_offset)*1000
                    ix2 = np.nonzero(t_sub <= tm2)[0][-1]

                    tms.append([ix1, ix2])
                except:
                    print 'skipping reward: ', ir

            #Times: 
            tms_dict[blk] = tms
            beta_dict[blk] = get_beta(np.squeeze(f['AD74']))
            beta_dict[blk, 'raw_ad74'] = np.squeeze(f['AD74'])

    if make_beta_file:
        fnm = fname[:-4]+'_tms_and_beta.pkl'
        d = dict(tms_dict=tms_dict, beta_dict = beta_dict)
        pickle.dump(d, open(fnm, 'wb'))
        print 'done saving beta and times'

        print victory_moose

    mc_kin = []
    mc_kin_hat = []
    mc_beta = []
    mc_binary_kin = []

    nf_kin = []
    nf_kin_hat = []
    nf_beta = []
    nf_binary_kin = []
    err_mn = []
    nf_kin_hat_mn = []
    nf_kin_mn = []

    b_cnt = 0
    for i_b, blk in enumerate(blocks):
        tms = tms_dict[blk]
        for i, (strt, stp) in enumerate(tms):
            for s in range(strt, stp):
                if mc_indicator_day[i_b] == '1':
                    spd_k = np.sqrt(np.sum(dat['act'][day, blk][s, :]**2))
                    mc_kin.append(spd_k)

                    spd_k2 = np.sqrt(np.sum(dat['pred_lpf'][day, blk][s, :]**2))
                    mc_kin_hat.append(spd_k2)

                    mc_beta.append(beta_dict[blk][1, s])
                    #mc_beta.append(dat['act'][day, blk, 'beta_sig'][s])                    
                    #mc_binary_kin.append(dat['act'][day, blk, 'binary_kin_sig'][s])
                    mc_binary_kin.append(int(spd_k2>3.5))
                
                elif mc_indicator_day[i_b] == '0':
                    spd_k = np.sqrt(np.sum(dat['act'][day, blk][s, :]**2))
                    nf_kin.append(spd_k)

                    spd_k2 = np.sqrt(np.sum(dat['pred_lpf'][day, blk][s, :]**2))
                    nf_kin_hat.append(spd_k2)

                    nf_beta.append(beta_dict[blk][1, s])
                    #nf_beta.append(dat['act'][day, blk, 'beta_sig'][s])

                    #nf_binary_kin.append(dat['act'][day, blk, 'binary_kin_sig'][s])
                    nf_binary_kin.append(int(spd_k2>3.5))
        prev_b_cnt = b_cnt
        b_cnt = len(nf_kin)

        if b_cnt > 0:
            ix_pk = np.arange(prev_b_cnt, b_cnt)
            b0_ = np.nonzero(np.array(nf_beta)[ix_pk]==0)[0]
            b1_ = np.nonzero(np.array(nf_beta)[ix_pk]==1)[0]
            
            err = ((np.array(nf_kin) - np.array(nf_kin_hat))**2)
            err_mn.append([np.mean(err[ix_pk][b0_]), np.mean(err[ix_pk][b1_])])

            nf_kin_hat_mn.append([np.mean(np.array(nf_kin_hat)[ix_pk][b0_]), np.mean(np.array(nf_kin_hat)[ix_pk][b1_])])
            nf_kin_mn.append([np.mean(np.array(nf_kin)[ix_pk][b0_]), np.mean(np.array(nf_kin)[ix_pk][b1_])])

    # k_act = []
    # k_hat = []
    # z = []

    # beta_err_lt = 0
    # tot_err = 0

    # beta_lower_act = 0
    # beta_lower_hat = 0

    # for i in range(np.array(err_mn).shape[0]):
    #     z.append(np.array(err_mn)[i, :])
    #     if err_mn[i][0] > err_mn[i][1]:
    #         beta_err_lt+=1
    #     tot_err += 1

    #     k_act.append([nf_kin_mn[i][0], nf_kin_mn[i][1]])
    #     if nf_kin_mn[i][0] > nf_kin_mn[i][1]:
    #         beta_lower_act += 1

    #     k_hat.append([nf_kin_hat_mn[i][0], nf_kin_hat_mn[i][1]])
    #     if nf_kin_hat_mn[i][0] > nf_kin_hat_mn[i][1]:
    #         beta_lower_hat += 1

    #Error is lower w/ beta? 
    #perc of pairs w/ beta lower; 

    # ff, p = scipy.stats.friedmanchisquare(*z)
    # print 'perc of blocks w/ lower beta error: ', beta_err_lt/np.float(tot_err)
    # if beta_err_lt/np.float(tot_err) > 0.5:
    #     ns0=True
    # else:
    #     ns0 = False
    # print 'p val of error comparison: ', p


    mc_kin = np.hstack((mc_kin))
    mc_kin_hat = np.hstack((mc_kin_hat))
    mc_binary_kin = np.hstack((mc_binary_kin))
    mc_beta = np.hstack((mc_beta))

    nf_kin = np.hstack((nf_kin))
    nf_kin_hat = np.hstack((nf_kin_hat))
    nf_binary_kin = np.hstack((nf_binary_kin))
    nf_beta = np.hstack((nf_beta))

    k_ix = np.nonzero(mc_binary_kin==0)[0]
    mc_ix0 = np.nonzero(mc_beta[k_ix]==0)[0]
    mc_ix1 = np.nonzero(mc_beta[k_ix]==1)[0]
    ax1.bar(0, np.mean(mc_kin[k_ix][mc_ix0]), yerr=np.std(mc_kin[k_ix][mc_ix0])/np.sqrt(len(mc_ix0)), color='b', hatch='')
    print 'mc, ', np.mean(mc_kin[k_ix][mc_ix0])
    ax1.bar(1, np.mean(mc_kin[k_ix][mc_ix1]), yerr=np.std(mc_kin[k_ix][mc_ix1])/np.sqrt(len(mc_ix1)), color='b', hatch='/')
    ax1.bar(2, np.mean(mc_kin_hat[k_ix][mc_ix0]), yerr=np.std(mc_kin_hat[k_ix][mc_ix0])/np.sqrt(len(mc_ix0)), color='g', hatch='')
    ax1.bar(3, np.mean(mc_kin_hat[k_ix][mc_ix1]), yerr=np.std(mc_kin_hat[k_ix][mc_ix1])/np.sqrt(len(mc_ix1)), color='g', hatch='/')

    ax1.set_ylabel('Speed (cm/s)')
    ax1.set_xlim([-.5, 4.5])
    ax1.set_ylim([0., 4.5])
    ax1.set_xticks(np.arange(4)+.5)
    ax1.set_xticklabels([''])
    ax1.set_title('MC Slow Spd Epochs, '+day)

    k_ix = np.nonzero(nf_binary_kin==0)[0]
    mc_ix0 = np.nonzero(nf_beta[k_ix]==0)[0]
    mc_ix1 = np.nonzero(nf_beta[k_ix]==1)[0]

    k_act = []
    k_act.append(nf_kin[k_ix][mc_ix0])
    k_act.append(nf_kin[k_ix][mc_ix1])
    tt, p1 = scipy.stats.ttest_ind(*k_act)
    print 'avg diff in beta0 - beta1', np.mean(nf_kin[k_ix][mc_ix0]) - np.mean(nf_kin[k_ix][mc_ix1])
    if np.mean(nf_kin[k_ix][mc_ix0]) - np.mean(nf_kin[k_ix][mc_ix1]) > 0:
        ns1 = True
    else:
        ns1 = False
    print 'p val of k_actual_nf', p1

    k_pred = []
    k_pred.append(nf_kin_hat[k_ix][mc_ix0])
    k_pred.append(nf_kin_hat[k_ix][mc_ix1])
    tt, p2 = scipy.stats.ttest_ind(*k_pred)
    print 'avg diff in pred beta0 - beta1', np.mean(nf_kin_hat[k_ix][mc_ix0]) - np.mean(nf_kin_hat[k_ix][mc_ix1])
    if np.mean(nf_kin_hat[k_ix][mc_ix0]) - np.mean(nf_kin_hat[k_ix][mc_ix1]) > 0:
        ns2 = True
    else:
        ns2 = False
    print 'p val of k_pred_nf', p2

    k_err = []
    k_err.append((nf_kin_hat[k_ix][mc_ix0] - nf_kin[k_ix][mc_ix0])**2)
    k_err.append((nf_kin_hat[k_ix][mc_ix1] - nf_kin[k_ix][mc_ix1])**2)
    tt, p0 = scipy.stats.ttest_ind(*k_err)
    print 'avg diff in err beta0 - beta1', np.mean(k_err[0]) - np.mean(k_err[1])
    if np.mean(k_err[0]) - np.mean(k_err[1]) > 0:
        ns0 = True
    else:
        ns0 = False
    print 'p val of k_pred_nf', p0


    ax2.bar(0, np.mean(nf_kin[k_ix][mc_ix0]), yerr=np.std(nf_kin[k_ix][mc_ix0])/np.sqrt(len(mc_ix0)), color='b', hatch='')
    ax2.bar(1, np.mean(nf_kin[k_ix][mc_ix1]), yerr=np.std(nf_kin[k_ix][mc_ix1])/np.sqrt(len(mc_ix1)), color='b', hatch='/')
    ax2.bar(2, np.mean(nf_kin_hat[k_ix][mc_ix0]), yerr=np.std(nf_kin_hat[k_ix][mc_ix0])/np.sqrt(len(mc_ix0)), color='g', hatch='')
    ax2.bar(3, np.mean(nf_kin_hat[k_ix][mc_ix1]), yerr=np.std(nf_kin_hat[k_ix][mc_ix1])/np.sqrt(len(mc_ix1)), color='g', hatch='/')

    ax2.set_ylabel('Speed (cm/s)')
    ax2.set_xlim([-.5, 4.5])
    ax2.set_ylim([0., 4.5])
    ax2.set_xticks(np.arange(4)+.5)
    ax2.set_xticklabels(['Actual, Beta Off', 'Actual, Beta On', 'Pred, Beta Off', 'Pred, Beta On'])
    add_title = ''
    add_title2 = ''
    if ns1:
        add_title = add_title+' p val, act_kin_B_off > act_kin_B_on: '+str(round(p1*1000)/1000.)+', '
    else:
        add_title = add_title+' n.s. act_kin_B_off > act_kin_B_on, '
    if ns2:
        add_title = add_title+' p val, pred_kin_B_off > pred_kin_B_on: '+str(round(p2*1000)/1000.)
    else:
        add_title = add_title+' n.s. p val, pred_kin_B_off > pred_kin_B_on'
    if ns0:
        add_title2 = add_title2 +' p val, err_B_off > err_B_on: '+str(round(p0*1000)/1000.)+', '
    else:
        add_title2 = add_title2 + ' n.s. p val, err_B_off > err_B_on'
    ax2.set_title('NF Epochs, '+day+': '+add_title)


    E = np.array(err_mn)
    for i in range(E.shape[0]):
        ax3.plot([0, 2], [E[i, 0], E[i, 1]], 'r.-')

    ax3.set_ylabel('Mean Sq. Err')
    ax3.set_xlim([-.5, 4.5])
    ax3.set_xticks([0.5, 2.5])
    ax3.set_xticklabels(['Err Beta Off', 'Err Beta On'])
    ax3.set_title('Err B/w Act-Pred in NF, '+day+ ': '+add_title2)

    plt.tight_layout()
    if save:

        plt.savefig(fname[:-4]+'_13_25.png')
    
def get_beta(lfp_dat, min_beta_burst_len=100, perc_beta = 60, bp_filt=[20, 45]):
    #Filter LFPs between 20 - 45 Hz: 
    nyq = 0.5* 1000
    bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
    data_filt = scipy.signal.filtfilt(bw_b, bw_a, lfp_dat)

    #Get amplitude of filtered signal: 
    try:
        sig = np.abs(scipy.signal.hilbert(data_filt, N=None, ))
    except:
        print 'error in preidct_w_mc_decoders, get_beta_bursts function'
    sig_bin = np.zeros_like(sig)
    sig_bin[sig > np.percentile(sig.reshape(-1), perc_beta)] = 1

    #Get only blobs >= 50 ms: 
    #see http://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array
    sig_bin_filt = sig_bin.copy()
    sig_bin_filt = np.vstack((np.zeros((len(sig_bin_filt))), sig_bin_filt, np.zeros((len(sig_bin_filt)))))
    struct = np.zeros((3,3))
    struct[1,:] = 1 #Get patterns that only are horizontal
    id_regions, num_ids = ndimage.label(sig_bin_filt, structure=struct)
    id_sizes = np.array(ndimage.sum(sig_bin, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes <= min_beta_burst_len )
    sig_bin_filt[area_mask[id_regions]] = 0
    return sig_bin_filt

def main_full_file_plt():
    for i_d, d in enumerate(sss.master_days):
    #for d in ['022415', '022515']:

        print 'starting day: ', d
        fnames, days, mc_indicator = train_mc_full_test_nf_full(day=d)
        try:
            #open_plot_train_mc_full_test_nf_full(fnames[0], mc_indicator[0], sss.master_blocks[i_d])
            open_plot_train_mc_full_test_nf_full(None, sss.master_mc_indicator[i_d], sss.master_blocks[i_d], day=d)
        except:
            print 'skipped plotting bc of error, relevant outputs:'
            #print fnames, days, mc_indicator
            print d
        print 'done w/ ', d

def train_mc_test_nf_decoders(test=False):
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test)
    decoder_dict = predict_kin_w_spk.get_kf_trained_from_full_mc(keep_dict, days, blocks, mc_indicator, decoder_only=True)
    b = 25
    spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = predict_kin_w_spk.get_full_blocks(keep_dict, days, blocks, mc_indicator, kin_type='endpt')

    ##Bin / smooth spike and beta dictionaries
    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict,
        beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=b)

    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_bin = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, b, smooth = 50)

    pred_kin_dict = {}

    for i_d, d in enumerate(days):
        day_spks = S_bin[d]*b
        day_b = B_bin[d]

        day_kin = []
        day_trial_type = []
        day_lfp_lab = []

        for ib, blk in enumerate(blocks[i_d]):
            day_kin.append(np.swapaxes(binned_kin_signal_dict[blk, d], 1, 2))

            n_trials = len(lfp_lab[blk, d])

            if int(mc_indicator[i_d][ib]) == 1:
                tt = ['mc']*n_trials
            elif int(mc_indicator[i_d][ib]) == 0:
                tt = ['beta']*n_trials
            else:
                raise NameError

            day_trial_type.append(tt)
            day_lfp_lab.append(lfp_lab[blk, d])

        day_trial_type = np.hstack(day_trial_type)
        day_kin = np.vstack((day_kin))
        day_lfp_lab = np.hstack((day_lfp_lab))


        #All stuff consolidated now: 
        ntrials, nbins, nunits = day_spks.shape

        #Training: 
        test_ix = np.nonzero(day_trial_type == 'beta')[0]
        day_resh_spks_test = day_spks[test_ix, :, :].reshape(len(test_ix)*nbins, nunits)
        day_beta_binary_test = day_b[test_ix, :].reshape(len(test_ix)*nbins)

        day_resh_kin_test = day_kin[test_ix, :].reshape(len(test_ix)*nbins, day_kin.shape[2])
        day_resh_beta_test = day_b[test_ix, :].reshape(len(test_ix)*nbins)

        pred_kin = np.zeros((ntrials, nbins, 7))
        ky = []
        for k in decoder_dict.keys():
            if k[1] == d:
                ky.append(k)
        if len(ky) > 1:
            ky = np.vstack((ky))
            ix = np.nonzero(ky[:, 2]== 'a')[0]
            k = ky[ix, :]
        else:
            k = ky[0]
        print 'Using Decoder: ', k
        filter_ = decoder_dict[tuple(np.squeeze(k))]

        for n in range(ntrials):
            filter_._init_state()
            s = filter_.state
            st = []
            for nb in range(nbins):
                obs = day_spks[n, nb, :][:, np.newaxis]
                s = filter_._forward_infer(s, obs)
                st.append(s.mean)
            pred_kin[n, :, :] = np.array(np.hstack((st))).T
        pred_kin_dict[d, 'pred_kin'] = pred_kin
        pred_kin_dict[d, 'filter'] = filter_
        pred_kin_dict[d, 'act_kin'] = day_kin
        pred_kin_dict[d, 'binary_beta'] = day_b
        pred_kin_dict[d, 'lfp_lab'] = day_lfp_lab
    import pickle 
    pickle.dump(pred_kin_dict, open('mc_train_nf_test_25msKF.pkl', 'wb'))

    if __name__ ==  "__main__":
        train_mc_test_nf_decoders()

def plot_train_mc_test_nf(fname):

    dat = pickle.load(open(fname))

    ############################# 
    ## Plot 1: Pred vs. Actual ## 
    ############################# 

    f, ax = plt.subplots(nrows = 4, ncols = 4)
    keys_arr = np.vstack((dat.keys()))
    days = np.sort(np.unique(keys_arr[:, 0]))

    for i_d, day in enumerate(days):
        day_lfp_lab = dat[day, 'lfp_lab']
        pred_kin = dat[day, 'pred_kin'][:, :, [3, 5]]
        act_kin = dat[day, 'act_kin'][:, :, [2, 3]]
        if np.mean(dat[day, 'act_kin'][:, :, [0, 1]].reshape(-1)) <1.:
            act_kin *= 100
            print 'mult. kin x 100: ', day
        ntrials, nbins, __ = pred_kin.shape

        for vel in [0, 1]:
            for im, trl_type in enumerate(['mc', 'beta']):
                if trl_type == 'mc':
                    ix = np.nonzero(day_lfp_lab<80)[0]
                elif trl_type == 'beta':
                    ix = np.nonzero(day_lfp_lab>80)[0]
                axi = ax[i_d, vel + (2*im)]
                offs = 0
                for n in ix:
                    axi.plot(offs+np.arange(nbins), act_kin[n, :, vel], 'b')
                    axi.plot(offs+np.arange(nbins), pred_kin[n, :, vel], 'g')
                    offs += nbins
                    axi.plot([offs, offs], [-5, 5], 'r-')
                axi.set_title('Vel: '+str(vel)+' Date: '+day+' Trials: '+trl_type)

    ############################## 
    #### Plot 2: MSE by Epoch #### 
    ############################## 
    f2, ax2_mc = plt.subplots(nrows=2)
    f3, ax3_nf = plt.subplots(nrows=2)
    f4, ax4_nftarg = plt.subplots(nrows=2, ncols=4)
    master_ax = [ax2_mc, ax3_nf, ax4_nftarg]

    color_dict = dict(c0='dodgerblue', c1='orangered', c2='teal')
    hatch_dict = dict(h0='/', h1='*', h2='')
    for i_d, day in enumerate(days):
        day_lfp_lab = dat[day, 'lfp_lab']
        pred_kin = dat[day, 'pred_kin'][:, :, [3, 5]]
        act_kin = dat[day, 'act_kin'][:, :, [2, 3]]
        if np.mean(dat[day, 'act_kin'][:, :, [0, 1]].reshape(-1)) <1.:
            act_kin *= 100
            print 'mult. kin x 100 pt 2: ', day

        binary_kin = np.zeros_like(act_kin[:, :, 0])
        for i in range(binary_kin.shape[0]):
            for j in range(binary_kin.shape[1]):
                if np.logical_or(np.abs(act_kin[i, j, 0]) > 0.5, np.abs(act_kin[i, j, 1]) > 0.5):
                    binary_kin[i, j] = 1

        binary_beta = dat[day, 'binary_beta']

        for vel in [0, 1]:

            #MC data: 
            mc_ix = np.nonzero(day_lfp_lab<80)[0]
            beta_ix = np.nonzero(day_lfp_lab>80)[0]
            beta_spec_ix = []
            for t in range(84, 88):
                tmp_ix = np.nonzero(day_lfp_lab==t)[0]
                beta_spec_ix.append(tmp_ix)

            trl_ix = [[mc_ix], [beta_ix], beta_spec_ix]
            num_tgs = [1, 1, 4]

            for ti, (trl_ix, n_tgs) in enumerate(zip(trl_ix, num_tgs)):

                for nt in range(n_tgs):
                    ix =  trl_ix[nt]
                    sub_pred_kin = pred_kin[ix, :, vel].reshape(-1)
                    sub_act_kin = act_kin[ix, :, vel].reshape(-1)
                    sub_beta = binary_beta[ix, :].reshape(-1)
                    sub_bin_kin = binary_kin[ix, :].reshape(-1)

                    k0 = np.nonzero(sub_bin_kin==0)[0]
                    k1 = np.nonzero(sub_bin_kin==1)[0]
                    k_all = np.arange(len(sub_bin_kin))
                    k_ix = [k0, k1, k_all]

                    for ki, kix in enumerate(k_ix):
                        ss_pk = sub_pred_kin[kix]
                        ss_ak = sub_act_kin[kix]
                        ss_bb = sub_beta[kix]
                        ss_bk = sub_bin_kin[kix]

                        b0 = np.nonzero(ss_bb==0)[0]
                        b1 = np.nonzero(ss_bb==1)[0]
                        b_all = np.arange(len(ss_bb))
                        b_ix = [b0, b1, b_all]

                        for bix, beta_ix in enumerate(b_ix):
                            mse = np.mean((ss_pk[beta_ix] - ss_ak[beta_ix])**2)
                            mse_sem = np.std((ss_pk[beta_ix] - ss_ak[beta_ix])**2)/np.sqrt(len(beta_ix))

                            if ti < 2:
                                axi = master_ax[ti][vel]
                            else:
                                axi = master_ax[ti][vel, nt]

                            xpos = (10*i_d)+(3*ki)+bix
                            color = color_dict['c'+str(ki)]
                            hatch = hatch_dict['h'+str(bix)]
                            alpha = 0.75

                            axi.bar(xpos, mse, yerr=mse_sem, color=color, ecolor=None, hatch=hatch, alpha=alpha)

if __name__ == '__main__':
    print 'starting!'
    main_full_file_plt()
    #train_mc_test_nf_decoders()
