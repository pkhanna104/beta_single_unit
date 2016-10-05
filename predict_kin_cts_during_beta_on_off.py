import pickle
import state_space_spks as sss
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5,style='whitegrid')

def plot_binary_beta_events(bp_filt, perc_beta=60, min_beta_burst_len=125):
    d = '022315'
    blk = 'b'
    fdict = pickle.load(open('/Users/preeyakhanna/Dropbox/carmena_lab/lfp_multitask/analysis/grom_spk_v2_may2016/2016-08-27T23:46022315_25_cts_KF_tms_and_beta.pkl'))
    beta_dict = fdict['beta_dict']

    for blk in ['b', 'g']:
        raw = np.squeeze(beta_dict[blk, 'raw_ad74'])
        nyq = 0.5* 1000
        bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
        data_filt = scipy.signal.filtfilt(bw_b, bw_a, raw)
        data_filt = data_filt[:25000]

        sig = np.abs(scipy.signal.hilbert(data_filt, N=None))
        sig_bin = np.zeros_like(sig)
        sig_bin[sig > np.percentile(sig, perc_beta)] = 1

        #Get only blobs >= 125 ms: 
        #see http://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array

        sig_bin_filt = np.vstack(( np.zeros((1, len(sig_bin))), sig_bin.copy(), np.zeros((1, len(sig_bin)))))
        struct = np.zeros((3,3))
        struct[1,:] = 1 #Get patterns that only are horizontal
        id_regions, num_ids = ndimage.label(sig_bin_filt, structure=struct)

        id_sizes = np.array(ndimage.sum(sig_bin, id_regions, range(num_ids + 1)))
        area_mask = (id_sizes <= min_beta_burst_len )
        sig_bin_filt[area_mask[id_regions]] = 0

        f, ax = plt.subplots(figsize=(10, 5))
        ax.plot(raw[:60000], label='raw')
        ax.plot(data_filt, label='bp filt')
        #ax.plot(sig, label='amp')
        ax.plot(sig_bin_filt[1,:]*.1, label='binary')
        if blk == 'b':
            ax.set_xlim([8140, 9360])
            ax.set_xticks([8140, 8640, 9140])
            ax.set_xticklabels([0., .5, 1.0])
            ax.set_xlabel('Time (sec)')
            ax.set_ylim([-.1, .12])
            ax.set_title('Manual Control')
        elif blk == 'g':
            ax.set_xlim([23600, 24820])
            ax.set_xticks([23600, 24100, 24600])
            ax.set_xticklabels([0., .5, 1.0])
            ax.set_xlabel('Time (sec)')
            ax.set_ylim([-.1, .12])
            ax.set_title('Beta Neurofeedback')
            
        ax.legend()
        plt.tight_layout()
        plt.savefig(d+blk+'_ex_beta.eps', format='eps', dpi=300)

def get_binary_events(bp_filt, perc_beta = 60, min_beta_burst_len = 125):
    blocks = sss.master_blocks
    mc_indicator = sss.master_mc_indicator
    days = sss.master_days

    beta_dict_binary = {}
    tms_dict_all = {}
    kin_dict_all = {}

    for i_d, d in enumerate(sss.master_days):
        srch1 = '2016-08-27*'+d+'_25_cts_KF.pkl'
        srch2 = '2016-08-28*'+d+'_25_cts_KF.pkl'
        
        fnm1 = glob.glob(srch1)
        fnm2 = glob.glob(srch2)
        if len(fnm1)==1:
            fname = fnm1[0]
        elif len(fnm2) == 1:
            fname = fnm2[0]
        else:
            raise Exception

        fdict = pickle.load(open(fname[:-4]+'_tms_and_beta.pkl'))
        tms_dict = fdict['tms_dict']
        tms_dict_all[d] = tms_dict
        beta_dict = fdict['beta_dict']

        fdict2 = pickle.load(open(fname))
        kin_dict_all[d, 'act']=fdict2['act']
        kin_dict_all[d, 'pred_lpf']=fdict2['pred_lpf']


        for ib, blk in enumerate(sss.master_blocks[i_d]):
            raw = np.squeeze(beta_dict[blk, 'raw_ad74'])

            nyq = 0.5* 1000
            bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
            data_filt = scipy.signal.filtfilt(bw_b, bw_a, raw)
            data_filt = data_filt

            #Get amplitude of filtered signal: 
            try:
                sig = np.abs(scipy.signal.hilbert(data_filt, N=None))
            except:
                print b, d, 'error in state_space_w_beta_bursts, get_beta_bursts function'
            sig_bin = np.zeros_like(sig)
            sig_bin[sig > np.percentile(sig, perc_beta)] = 1

            #Get only blobs >= 125 ms: 
            #see http://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array

            sig_bin_filt = np.vstack(( np.zeros((1, len(sig_bin))), sig_bin.copy(), np.zeros((1, len(sig_bin)))))
            struct = np.zeros((3,3))
            struct[1,:] = 1 #Get patterns that only are horizontal
            id_regions, num_ids = ndimage.label(sig_bin_filt, structure=struct)

            id_sizes = np.array(ndimage.sum(sig_bin, id_regions, range(num_ids + 1)))
            area_mask = (id_sizes <= min_beta_burst_len )
            sig_bin_filt[area_mask[id_regions]] = 0

            beta_dict_binary[blk, d] = sig_bin_filt

    return beta_dict_binary, tms_dict_all, kin_dict_all

def analyze_kin_in_binary_events(beta_dict_binary, tms_dict_all, kin_dict_all):

    for i_d, d in enumerate(sss.master_days):

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
        for i_b, blk in enumerate(sss.master_blocks[i_d]):

            tms = tms_dict_all[d][blk]
            for i, (strt, stp) in enumerate(tms):
                ix = np.arange(strt, stp)

                if sss.master_mc_indicator[i_d][i_b] == '1':
                    spd_k = np.sqrt(np.sum(kin_dict_all[d, 'act'][d, blk][ix, :]**2, axis=1))
                    mc_kin.extend(list(spd_k))

                    spd_k2 = np.sqrt(np.sum(kin_dict_all[d, 'pred_lpf'][d, blk][ix, :]**2, axis=1))
                    mc_kin_hat.extend(list(spd_k2))

                    mc_beta.extend(list(beta_dict_binary[blk, d][1, ix]))
                    #mc_beta.append(dat['act'][day, blk, 'beta_sig'][s])                    
                    #mc_binary_kin.append(dat['act'][day, blk, 'binary_kin_sig'][s])
                    tmp = [ss > 3.5 for i_, ss in enumerate(spd_k)]
                    mc_binary_kin.extend(tmp)
                
                elif sss.master_mc_indicator[i_d][i_b] == '0':
                    spd_k = np.sqrt(np.sum(kin_dict_all[d, 'act'][d, blk][ix, :]**2, axis=1))
                    nf_kin.extend(list(spd_k))

                    spd_k2 = np.sqrt(np.sum(kin_dict_all[d, 'pred_lpf'][d, blk][ix, :]**2, axis=1))
                    nf_kin_hat.extend(list(spd_k2))

                    nf_beta.extend(list(beta_dict_binary[blk, d][1, ix]))
                    #nf_beta.append(dat['act'][day, blk, 'beta_sig'][s])

                    #nf_binary_kin.append(dat['act'][day, blk, 'binary_kin_sig'][s])
                    tmp = [ss > 3.5 for i_, ss in enumerate(spd_k)]
                    nf_binary_kin.extend(tmp)
            
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


        # Stack everything!
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
        

