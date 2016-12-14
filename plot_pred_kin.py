import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import state_space_w_beta_bursts as ssbb
import scipy.stats
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5,style='whitegrid')

save_directory = '/Volumes/TimeMachineBackups/v2_Beta_pap_rev_data/beta_regressions/'
binsizes = [5, 10, 25, 50, 75, 100]

color_dict = dict()
color_dict['mc'] = 'k'
color_dict[84] = 'lightseagreen'
color_dict[85] = 'dodgerblue'
color_dict[86] = 'gold'
color_dict[87] = 'orangered'
color_dict[88] = 'k'
color_dict[0] = 'k'
color_dict[1] = 'r'

def agg_X_preds(binsizes=binsizes, date='030215', savedate='today', X='kin'):
    ### METRICS: MEAN, STD, N1, SEM ###
    # 3 figures (kin, pred_kin,  error)
    # Color = BMI target B, G, Y, R, K (mc)
    # Subplot cols = Beta on vs. Beta off for Gaussian vs. Poisson
    # Subplot rows = Reach, Hold, Full
    # X axis = Bin size
    f1, ax1 = plt.subplots(nrows = 3, ncols = 4)
    f2, ax2 = plt.subplots(nrows = 3, ncols = 4)
    f3, ax3 = plt.subplots(nrows = 3, ncols = 4)
    f4, ax4 = plt.subplots(nrows = 3, ncols = 4)
    master_ax = [ax1, ax2, ax3, ax4]

    if savedate is 'today':
        now = datetime.datetime.now()
        dt = now.isoformat()
        datesaved = dt[:10]
        regx = datesaved +'_'
    else:
        regx = savedate+'_'

    for im, model in enumerate(['poisson']):
        add = im*2
        for bin in binsizes:
            fname = save_directory+regx+date+'_'+str(bin)+'_' + model + '_predicting_'+X+'.pkl'
            found_fnames = glob.glob(fname)
            if len(found_fnames) == 1:
                f = pickle.load(open(found_fnames[0]))
            else:
                print 'No file found: ', fname
                raise NameError

            try:
                f_date = f[(date, bin, model)]
            except:
                f_date = f[(date, bin, model, 0)]

            for im, metric in enumerate(['Y_pred','kin','signal_error', 'R2']):
                axi = master_ax[im]

                for ib, beta in enumerate(['beta_on', 'beta_off']):

                    for ie, epoch in enumerate(['hold','reach','full']):

                        for it, targ in enumerate([84, 85, 86, 87, 88]):
                            color = color_dict[targ]
                            info = f_date[epoch, metric, targ, beta]

                            sub_axi = axi[ie, ib+add]

                            if info[3] != np.nan:
                                sub_axi.errorbar(bin, info[0], yerr=info[3], marker='.', color=color)
                            else:
                                sub_axi.plot(bin, info[0], marker='.', color=color)

                        sub_axi.set_title(epoch + ','+ beta + ','+ metric + ','+ model)
                        sub_axi.set_xlim([binsizes[0]-5, binsizes[-1]+5])
                        if X == 'kin':
                            if metric is 'signal_error':
                                sub_axi.set_ylim([0., 0.01])
                            elif metric is 'R2':
                                pass
                            else:
                                sub_axi.set_ylim([0., 0.08])

def agg_beta_preds(binsizes = binsizes, dates=['022315', '022715', '022815','030215'], savedate='today'):
    # 1 figure
    # Color: Beta On = B, Beta Off = K
    # Subplot cols = Gaussian vs. Poisson, mc vs. beta
    # Subplot rows = Dates
    # X axis = Bin size
    f1, ax1 = plt.subplots(nrows = 4, ncols = 4)
    if savedate is 'today':
        now = datetime.datetime.now()
        dt = now.isoformat()
        datesaved = dt[:10]
        regx = datesaved + (16*'?')+'_'
    else:
        regx = savedate+(16*'?')+'_'

    for i_d, date in enumerate(dates):
        for im, model in enumerate(['gaussian', 'poisson']):
            
            for bin in binsizes:
                fname = save_directory+regx+date+'_'+str(bin)+'_' + model + '_predicting_kin.pkl'
                found_fnames = glob.glob(fname)
                if len(found_fnames) == 1:
                #fname = save_directory+date+'_'+str(bin)+'_' + model + '_predicting_beta.pkl'
                    f = pickle.load(open(found_fnames[0]))
                
                kz = np.sort(np.array([k[0] for k in f.keys()]))

                for ik, key in enumerate(kz):

                    for it, task in enumerate(['mc', 'beta']):
                        axi = ax1[ik, im+(2*it)]
                        info = f[key, bin, model][it]
                        axi.bar([bin], info[0,0]/float(np.sum(info[0,:])), color='gray',width=4)
                        axi.bar([bin+4], info[1,1]/float(np.sum(info[1,:])), color='black', width=4)
                        axi.set_title(model+','+task+','+key)

def plot_LDA_beta(fname, dates=['022315', '022715', '022815','030215']):
    dat = pickle.load(open(fname))
    ch = dat['CH']
    sc = dat['SC']
    f2, ax2 = plt.subplots()
    for i_d, d in enumerate(dates): 
        f, ax = plt.subplots(nrows=3)

        for ie, epoch in enumerate(['hold', 'reach', 'full']):
            axi = ax[ie]
            for ib, b in enumerate(binsizes):
                score = sc[b, d][epoch, 'test']
                chance = ch[b, d][epoch, 'test']
                
                tr_score = sc[b, d][epoch, 'train']
                tr_chance = ch[b, d][epoch, 'train']

                l1 = axi.bar(b, score, width = 1.5, color='b')
                l2 = axi.bar(b+1.5, np.mean(chance), width = 1.5, color='k')
                l3 = axi.bar(b+3, tr_score, width = 1.5, color='b', alpha=0.5)
                l4 = axi.bar(b+4.5, np.mean(tr_chance), width = 1.5, color='k', alpha=0.5)
                ax2.plot(b, 100*(score-np.mean(chance)), '.', color=color_dict[ie+85])
                print d, epoch, b, score-np.mean(chance)
            axi.set_xlabel('Binsize (ms)')
            axi.set_ylabel('Classific. Accuracy')
            axi.set_title(d)
            plt.legend([l1, l2, l3, l4], ['Sc Test', 'Sc Chance Tst', 'Sc Train', 'Sc Chance Tr'], 
                loc='upper center',fontsize='x-small')

    ax2.set_xlabel('Binsize (ms)')
    ax2.set_ylabel('Score - Chance (perc)')

def plot_LDA_beta_v2(fname, dates=['022315', '022715', '022815','030215']):
    class_names = np.hstack(('mc', range(84, 88)))
    dat = pickle.load(open(fname))
    ch = dat['CH']
    sc = dat['SC']
    f2, ax2 = plt.subplots(nrows=3)
    for i_d, d in enumerate(dates): 
        f, ax = plt.subplots(nrows=3, ncols=5)

        for ie, epoch in enumerate(['hold', 'reach', 'full']):
            for ib, b in enumerate(binsizes):
                for ic, c in enumerate(class_names):
                    axi = ax[ie, ic]
                    if c=='mc':
                        key = 'mc'
                    else:
                        key = int(c)
                    if b == 5:
                        b_adj = 0
                    else:
                        b_adj = b
                    score = np.array([sc[b, d, key][0, epoch, 'test'], sc[b, d, key][1, epoch, 'test']])
                    chance = ch[b, d, key][0, epoch, 'test']+ ch[b, d, key][1, epoch, 'test']
                    tr_score = np.array([sc[b, d, key][0, epoch, 'train'], sc[b, d, key][1, epoch, 'train']])
                    tr_chance = ch[b, d, key][0, epoch, 'train'] + ch[b, d, key][1, epoch, 'train']

                    l1 = axi.bar(b_adj, np.mean(score), width = 1.5, color='b')
                    l2 = axi.bar(b_adj+1.5, np.mean(chance), width = 1.5, color='k')
                    l3 = axi.bar(b_adj+3, np.mean(tr_score), width = 1.5, color='b', alpha=0.5)
                    l4 = axi.bar(b_adj+4.5, np.mean(tr_chance), width = 1.5, color='k', alpha=0.5)


                    ax2[ie].plot(b, 100*(np.mean(score)-np.mean(chance)), '.', color=color_dict[key])
                    print d, epoch, b, 100*(score-np.mean(chance))
            axi.set_xlabel('Binsize (ms)')
            axi.set_ylabel('Classific. Accuracy')
            axi.set_title(d)
            plt.legend([l1, l2, l3, l4], ['Sc Test', 'Sc Chance Tst', 'Sc Train', 'Sc Chance Tr'], 
                loc='upper center',fontsize='x-small')

    ax2.set_xlabel('Binsize (ms)')
    ax2.set_ylabel('Score - Chance (perc)')

def plot_filter_pred(binsizes, date, savedate='today', predict='kin'):
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False, days=[date])
    f1, ax1 = plt.subplots(nrows=3)
    f2, ax2 = plt.subplots(nrows=3)
    master_ax = [ax1, ax2]

    for ib2, bin in enumerate(binsizes):
        
        B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=bin)
        nbins = 2500/bin
        epochs = dict(hold=[0., 0.6*nbins], reach=[0.6*nbins, nbins], full=[0., nbins])

        if savedate is 'today':
            now = datetime.datetime.now()
            dt = now.isoformat()
            datesaved = dt[:10]
            regx = datesaved +'_'
        else:
            regx = savedate+'_'

        for im, model in enumerate(['kf', 'ppf']):
            ax = master_ax[im]

            fname = save_directory+regx+date+'_'+str(bin)+'_ssm_filters' + '_predicting_'+predict+'_'+model+'.pkl'
            found_fnames = glob.glob(fname)
            if len(found_fnames) == 1:
                f = pickle.load(open(found_fnames[0]))
            else:
                print 'No file found: ', fname
                raise NameError

            key = (date, bin, 'ssm_filters')
            #plot_rand(f, key, type='spd')

            #Now take the diff b/w Xvel and Yvel, demean them, and plot: 
            err_tr, err_ts = get_err2(f, date, bin)
            err = np.vstack((err_tr, err_ts))

            #Get binary beta for this: 
            train_ix = np.nonzero(Params[date]['day_lfp_labs']<80)[0]
            test_ix = np.nonzero(Params[date]['day_lfp_labs']>80)[0]
            bbin_trn = B_bin[date][train_ix, :]
            bbin_tst = B_bin[date][test_ix, :]

            for ib, beta in enumerate(['beta_on', 'beta_off']):

                for ie, epoch in enumerate(['hold','reach','full']):
                    ep_bins = epochs[epoch]
                    for it, targ in enumerate([84, 85, 86, 87, 88]):
                        color = color_dict[targ]

                        #Compute error: 
                        #trials: 
                        if targ < 88:
                            ix = np.nonzero(Params[date]['day_lfp_labs'] == targ)[0]
                        else:
                            ix = np.nonzero(Params[date]['day_lfp_labs'] <= 80)[0]

                        sub_err = err[ix, ep_bins[0]:ep_bins[1]].reshape(-1)
                        #sub_err = 
                        sub_beta = B_bin[date][ix, ep_bins[0]:ep_bins[1]].reshape(-1)
                        ix0 = np.nonzero(sub_beta==0)[0]
                        ix1 = np.nonzero(sub_beta==1)[0]

                        ax[ie].plot(bin+(it*.5), np.mean(sub_err[ix0]), 's', color=color_dict[targ])
                        ax[ie].plot(bin+(it*.5), np.mean(sub_err[ix1]), '.', color=color_dict[targ])
                        ax[ie].plot([bin+(it*.5), bin+(it*.5)], [np.mean(sub_err[ix0]), np.mean(sub_err[ix1])], '-', color=color_dict[targ])
                        ax[ie].set_ylabel('Beta Est.')
                        ax[ie].set_title(epoch)
            #plt.tight_layout()

                        t = np.linspace(-1.5, 1.0, nbins)
                        ht = np.nonzero(np.logical_and(t<0, t>=-0.5))[0]
                        rt = np.nonzero(np.logical_and(t<.5, t>=0))[0]
                        ix_ht = np.ix_(ix, ht)
                        ix_rt = np.ix_(ix, rt)
                        #ax[0].plot(bin+(0.5*it), np.mean(err[ix_ht].reshape(-1)), 's', color=color_dict[targ])
                        #ax[1].plot(bin+(0.5*it), np.mean(err[ix_rt].reshape(-1)), 's', color=color_dict[targ])
        #     ax[0].set_ylabel('Vel Squ Err')
        #     ax[1].set_xlabel('Bin Size')
        #     ax[0].set_title('Model: '+model+' Date: '+date)
        # plt.tight_layout()

                        # ax[ib2].plot(t, np.mean(err[ix, :], axis=0), '.-', color=color_dict[targ])
                        # #ax[ib2].set_xlabel('Trial Time')
                        # ax[ib2].set_ylabel('Vel-Squ Error')
                        # ax[0].set_title('Model: '+model)





                        #     sub_axi = axi[ie, ib+add]

                        #     if info[3] != np.nan:
                        #         sub_axi.errorbar(bin, info[0], yerr=info[3], marker='.', color=color)
                        #     else:
                        #         sub_axi.plot(bin, info[0], marker='.', color=color)

                        # sub_axi.set_title(epoch + ','+ beta + ','+ metric + ','+ model)
                        # sub_axi.set_xlim([binsizes[0]-5, binsizes[-1]+5])
                        # if X == 'kin':
                        #     if metric is 'signal_error':
                        #         sub_axi.set_ylim([0., 0.01])
                        #     elif metric is 'R2':
                        #         pass
                        #     else:
                        #         sub_axi.set_ylim([0., 0.08])

def get_beta_binary(lfp_sig, bp_filt=[20, 45]):

    #Filter LFPs between 20 - 45 Hz: 
    nyq = 0.5* 1000
    bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
    data_filt = scipy.signal.filtfilt(bw_b, bw_a, lfp_sig)

    #Get amplitude of filtered signal: 
    try:
        sig = np.abs(scipy.signal.hilbert(data_filt, N=None))
    except:
        print 'error in plot_pred_dkin, get_beta_bursts function'
    sig_bin = np.zeros_like(sig)
    sig_bin[sig > np.percentile(sig.reshape(-1), perc_beta)] = 1

    #Get only blobs >= 50 ms: 
    #see http://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array

    sig_bin_filt = sig_bin.copy()
    struct = np.zeros((3,3))
    struct[1,:] = 1 #Get patterns that only are horizontal
    id_regions, num_ids = ndimage.label(sig_bin_filt, structure=struct)
    id_sizes = np.array(ndimage.sum(sig_bin, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes <= min_beta_burst_len )
    sig_bin_filt[area_mask[id_regions]] = 0

    beta_dict[b, d] = sig_bin_filt
    beta_dict_cont[b, d] = sig
    beta_dict_cont[b, d, 'phz'] = np.angle(scipy.signal.hilbert(data_filt, N=None, axis=1))
    beta_dict_cont[b, d, 'filt'] = data_filt

def plot_filter_pred2(bin, date, savedate='2016-07-12', predict='kin'):
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False, days=[date])
    f1, ax1 = plt.subplots(nrows=5)
    f2, ax2 = plt.subplots(nrows=5)
    master_ax = [ax1, ax2]

    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=bin)
    nbins = 2500/bin
    epochs = dict(hold=[0., 0.6*nbins], reach=[0.6*nbins, nbins], full=[0., nbins])

    if savedate is 'today':
        now = datetime.datetime.now()
        dt = now.isoformat()
        datesaved = dt[:10]
        regx = datesaved +'_'
    else:
        regx = savedate+'_'

    for im, model in enumerate(['ppf']):
        ax = master_ax[im]

        fname = save_directory+regx+date+'_'+str(bin)+'_ssm_filters' + '_predicting_'+predict+'_'+model+'.pkl'
        found_fnames = glob.glob(fname)
        if len(found_fnames) == 1:
            f = pickle.load(open(found_fnames[0]))
        else:
            print 'No file found: ', fname
            raise NameError

        key = (date, bin, 'ssm_filters')
        #plot_rand(f, key, type='spd')

        #Now take the diff b/w Xvel and Yvel, demean them, and plot: 
        spd = get_disc_kin(f, date, bin)

        #Get binary beta for this: 
        train_ix = np.nonzero(Params[date]['day_lfp_labs']<80)[0]
        test_ix = np.nonzero(Params[date]['day_lfp_labs']>80)[0]
        bbin = B_bin[date]
        R = {}
        for it, targ in enumerate([84, 85, 86, 87, 88]):
            axi = ax[it]

            if targ == 88:
                ix = np.nonzero(Params[date]['day_lfp_labs']<80)[0]
            else:
                ix = np.nonzero(Params[date]['day_lfp_labs']==targ)[0]

            for ie, epoch in enumerate(['reach']):
                ep_bins = epochs[epoch]


                for vel in [2]:
                    act = spd[vel, 'act'][ix, ep_bins[0]:ep_bins[1]].reshape(-1)
                    pred = spd[vel, 'pred'][ix, ep_bins[0]:ep_bins[1]].reshape(-1)
                    beta_arr = bbin[ix, ep_bins[0]:ep_bins[1]].reshape(-1)

                    # rts = []
                    # for iii in range(len(ix)):
                    #     try:
                    #         _ix = np.nonzero(pred[iii, :]> 40)[0]
                    #         _ix2 = _ix > 75
                    #         rts.append(_ix[_ix2][0])
                    #     except:
                    #         print iii
            #R[targ] = rts


                    for ib, beta in enumerate(['beta_off', 'beta_on']):
                        #axi = ax[ie, ib]
                        bins = np.linspace(0, 10, 20)
                        ix2 = np.nonzero(beta_arr==ib)[0]
                        _n, _x = np.histogram(pred[ix2], bins)
                        axi.plot(_x[1:], np.cumsum(_n)/float(np.sum(_n)), '-', color=color_dict[ib])
            axi.set_title('Target: '+str(targ-83))
            axi.set_ylabel('CDF')
            axi.set_xlabel('Pred Kin w/ PPF, Binsize: '+str(bin))
            plt.tight_layout()

                        # act_ = act[ix2]
                        # ix_act1 = np.arange((len(act_)/n_b)*n_b)
                        # ix_act2 = np.argsort(act_[ix_act1])
                        # sub_act = act_[ix_act1[ix_act2]].reshape(n_b, len(ix_act1)/n_b)
                        # mn_sub_act = np.sqrt(np.mean(sub_act, axis=1))

                        # pred_ = pred[ix2]
                        # sub_pred = pred_[ix_act1[ix_act2]].reshape(n_b, len(ix_act1)/n_b)
                        # mn_sub_pred = np.sqrt(np.mean(sub_pred, axis=1))

                        # axi.plot(mn_sub_act, mn_sub_pred, '.', color=color_dict[targ], markersize=10)
                        # slp, int_, rv_, pv_, se_ = scipy.stats.linregress(mn_sub_act, mn_sub_pred)
                        # axi.plot(mn_sub_act, int_+(slp*mn_sub_act), '-', color=color_dict[targ])
                        # print beta, epoch, targ, model, 'R**2: ', rv_**2, 'Slope: ', slp
                



                        #Compute error:   

def plot_rand(f, key, type_='spd'):
    f1, ax = plt.subplots(nrows=3, ncols=3)
    f2, ax2 = plt.subplots(nrows=3, ncols=3)

    master_ax = [ax, ax2]
    master_key = [['test_pred', 'test_act'], ['train_pred', 'train_act']]
    color_order = [['r--', 'r-'], ['k--','k-']]

    for i in range(2):
        axi = master_ax[i]
        keys = master_key[i]

        ky = list(key)
        ky = ky + [type_, keys[0]]
        ky2 = tuple(ky)

        max_tr = f[ky2].shape[0]

        ix = np.random.permutation(max_tr)
        for j in range(2):
            ky = list(key)
            ky = ky + [type_, keys[j]]
            ky2 = tuple(ky)
            if keys[j].find('act') > 0:
                fact = 10
            else:
                fact = 1

            for k in range(9):
                axi[k/3, k%3].plot(f[ky2][ix[k], :]*fact, color_order[i][j])
    plt.show()

def get_err(f, date, bin):
    err = {}
    spd = {}
    for i in [2, 3]:
        key1 = f[date, bin, 'ssm_filters', i, 'train_act']*100
        key2 = f[date, bin, 'ssm_filters', i, 'train_pred']
        key3 = f[date, bin, 'ssm_filters', i, 'test_act']*100
        key4 = f[date, bin, 'ssm_filters', i, 'test_pred']

        k = [key1, key2, key3, key4]
        k_dmn = []
        for ik in k:
            mn = np.tile(np.mean(ik, axis=1)[:, np.newaxis], [1, 2500/bin])
            k_dmn.append(ik-mn)

        err['train', i] = (k_dmn[0] - k_dmn[1])**2
        err['test', i] = (k_dmn[2] - k_dmn[3])**2
        spd['train', i] = key2**2
        spd['test', i] = key4**2


    #Actual Error
    err_tr = np.sqrt(err['train', 2]+err['train', 3])
    err_ts = np.sqrt(err['test', 2]+err['test', 3])
    return err_tr, err_ts

def get_disc_kin(f, date, bin):
    spd = {}
    for i in [2, 3]:
        key1 = f[date, bin, 'ssm_filters', i, 'train_act']*100
        key2 = f[date, bin, 'ssm_filters', i, 'train_pred']
        key3 = f[date, bin, 'ssm_filters', i, 'test_act']*100
        key4 = f[date, bin, 'ssm_filters', i, 'test_pred']

        k = [key1, key2, key3, key4]
        k_dmn = []
        for ik in k:
            mn = np.tile(np.mean(ik, axis=1)[:, np.newaxis], [1, 2500/bin])
            k_dmn.append(ik-mn)

        spd[i, 'pred'] = np.vstack((k_dmn[1]**2, k_dmn[3]**2))
        spd[i, 'act'] = np.vstack((k_dmn[0]**2, k_dmn[2]**2))

    return spd

def get_err2(f, date, bin):
    err = {}
    key1 = f[date, bin, 'ssm_filters', 0, 'train_act']*10
    key2 = f[date, bin, 'ssm_filters', 0, 'train_pred']
    key3 = f[date, bin, 'ssm_filters', 0, 'test_act']*10
    key4 = f[date, bin, 'ssm_filters', 0, 'test_pred']

    k = [key1, key2, key3, key4]
    k_dmn = []
    for ik in k:
        mn = np.tile(np.mean(ik, axis=1)[:, np.newaxis], [1, 2500/bin])
        k_dmn.append(ik-mn)

    #err_tr = np.sqrt((k_dmn[0] - k_dmn[1])**2)
    #err_ts = np.sqrt(k_dmn[2] - k_dmn[3])**2)

    err_tr = key2
    err_ts = key4
    return err_tr, err_ts

def plot_lpf_beta_25ms_v2(fname):
    dat = pickle.load(open(fname))
    f, ax =plt.subplots(nrows = 4, ncols = 2)
    ky = []
    days = np.array(['022315', '022715', '022815', '030215'])
    for k in dat.keys():
        if len(k) >= 7:
            if k[-1] == 'mean_abs_err':
                if k[2]=='a':
                    ky.append(k)
        elif len(k) >=5:
            if k[-1] == 'full_R2_lpf':
                print k, dat[k]
            elif k[-1] == 'y':
                if k[2]=='a':
                    i_d = np.nonzero(days==k[1])[0][0]
                    ax[i_d, 0].plot(dat[k][:, 0], color='b')
                    ax[i_d, 1].plot(dat[k][:, 1], color='b')
            elif k[-1] == 'y_hat_lpf':
                if k[2]=='a':
                    i_d = np.nonzero(days==k[1])[0][0]
                    ax[i_d, 0].plot(dat[k][:, 0], color='g')
                    ax[i_d, 1].plot(dat[k][:, 1], color='g')
    
    ky = np.vstack((ky))
    days = np.unique(ky[:,1])
    blocks = []
    for i_d, d in enumerate(days):
        ix = np.nonzero(ky[:,1]==d)[0]
        blks = np.unique(ky[ix, 2])
        blocks.append(blks)

    beta = [0, 1]
    vel = [0, 1]

    f, ax = plt.subplots(nrows=2)
    f2, ax2 = plt.subplots(nrows=2)

    for iv, v in enumerate(vel):
        for i_d, d in enumerate(days):
            for ib, blk in enumerate(blocks[i_d]):
                ax[iv].plot((2*i_d)+ib, np.mean(dat[25, d, blk, 'kf', 0, v, 'y_pred']), 'b.')
                ax[iv].plot((2*i_d)+ib+.1, np.mean(dat[25, d, blk, 'kf', 0, v, 'y_act']), 'bs')
                
                ax[iv].bar((2*i_d)+ib, dat[25, d, blk, 'kf', 0, v], color='blue', alpha=0.5, width=.25)
                ax[iv].bar((2*i_d)+ib+0.5, dat[25, d, blk, 'kf', 1, v], color='red', alpha=0.5, width=.25)

                ax[iv].plot((2*i_d)+ib+.5, np.mean(dat[25, d, blk, 'kf', 1, v, 'y_pred']), 'r.')
                ax[iv].plot((2*i_d)+ib+.5+.1, np.mean(dat[25, d, blk, 'kf', 1, v, 'y_act']), 'rs')

                n_pred = dat[25, d, blk, 'kf', 0, v, 'y_pred']
                n_act = dat[25, d, blk, 'kf', 0, v, 'y_act']
                y_pred = dat[25, d, blk, 'kf', 1, v, 'y_pred']
                y_act = dat[25, d, blk, 'kf', 1, v, 'y_act']

                ax2[iv].bar((2*i_d)+ib, np.mean((n_pred - n_act)**2), yerr=np.std((n_pred - n_act)**2)/np.sqrt(len(n_pred)), color='blue', alpha=0.5, width=.25)
                ax2[iv].bar((2*i_d)+ib+.5, np.mean((y_pred - y_act)**2), yerr=np.std((y_pred - y_act)**2)/np.sqrt(len(y_pred)), color='red', alpha=0.5, width=.25)
                

    ax[0].set_ylabel('|Predicted X Vel| - |Act X Vel|')
    ax[1].set_ylabel('|Predicted Y Vel| - |Act Y Vel|')
    ax[1].set_xlabel('Days')
    ax[0].set_title('Red: During Beta, Blue: Off Beta, All: During Kin < 0.5 cm/sec')

    ax2[0].set_ylabel('Mean Sq. Err (S.E) X Vel')
    ax2[1].set_ylabel('Mean Sq. Err (S.E) Y Vel')
    ax2[1].set_xlabel('Days')
    ax2[0].set_title('Red: During Beta, Blue: Off Beta, All: During Kin < 0.5 cm/sec')

def plot_6class_lda(fname='six_class_LDA_train_test.pkl'):
    eps = 10**-12
    dat = pickle.load(open(fname))
    f, ax = plt.subplots(nrows = 6, ncols = 2)
    f2, ax2 = plt.subplots(nrows=6, )
    f3, ax3 = plt.subplots()

    for i_d, d in enumerate(np.sort(dat.keys())): #days
        try:
            x = dat[d]['train_nf', 'test_mc']
        except:
            x = dat[d]['nf', 'mc']

        for i in range(x.shape[0]):
            x[i, :] = x[i, :]/float(np.sum(x[i, :]))+eps
        ax[i_d, 0].pcolormesh(x, vmin=0., vmax=1.)

        try:
            x2 = dat[d]['train_mc', 'test_nf']
        except:
            x2 = dat[d]['mc', 'nf']

        for i in range(x.shape[0]):
            x2[i, :] = x2[i, :]/float(np.sum(x2[i, :]))+eps
        c = ax[i_d, 1].pcolormesh(x2,vmin=0., vmax=1.)   

        #Bar plot: 
        mv_mn = 0
        hld_mn = 0
        for i in range(4):
            ax2[i_d].bar(i+1, x[4, i], color='b', alpha=0.5, width=0.4)
            ax2[i_d].bar(i+1.5, x[5, i], color='r', alpha=0.5, width=0.4)
            mv_mn += (i+1)*x[5, i]
            hld_mn += (i+1)*x[4, i]
        ax2[i_d].set_title(d+' avg move: '+str(mv_mn)+' avg hold: '+str(hld_mn))
        ax2[i_d].set_ylim([0., .4])
    ax2[3].set_xlabel('LFP Target Number')
    ax2[0].set_ylabel('Perc. Classification')
    #ax2[0].set_title('Red : Move, Blue: Hold (from MC)')
    #plt.tight_layout()

    f, ax = plt.subplots(nrows = 6, ncols = 2)
    for i_d, d in enumerate(np.sort(dat.keys())):
        try:
            x = dat[d]['train_nf', 'test_nf']
        except:
            x = dat[d]['nf', 'nf']

        corr = np.trace(x)/np.sum(x)
        for i in range(x.shape[0]):
            x[i, :] = x[i, :]/float(np.sum(x[i, :]))+eps
        ax[i_d, 0].pcolormesh(x, vmin=0., vmax=1.)
        ax[i_d, 0].set_title('Perc. Corr: '+str(corr))    

        try:
            x2 = dat[d]['train_mc', 'test_mc'][4:, 4:]
        except:
            x2 = dat[d]['mc', 'mc']
        corr = np.trace(x2)/np.sum(x2)
        for i in range(x2.shape[0]):
            x2[i, :] = x2[i, :]/float(np.sum(x2[i, :]))+eps
        ax[i_d, 1].pcolormesh(x2, vmin=0., vmax=1.)
        ax[i_d, 1].set_title('Perc. Corr: '+str(corr)) 
    #plt.tight_layout()

def plot_6class_lda_xval_within_task(fname='six_class_LDA_within_task_xval5.pkl', within_class=True):
    eps = 10**-12
    dat = pickle.load(open(fname))

    mc_ch = []
    mc_perf = []
    nf_ch = []
    nf_perf = []
    f2, ax2 = plt.subplots()

    binom = {}
    binom['mc_corr'] = 0
    binom['mc_ch'] = 0
    binom['mc_n'] = 0

    binom['nf_corr'] = 0
    binom['nf_ch'] = 0
    binom['nf_n'] = 0


    x = dict()
    for i in [0, 1, 3, 4]:
        x[i] = []

    for i_d, d in enumerate(np.sort(dat.keys())): #days
        f, ax = plt.subplots(ncols=3, figsize=(15, 5))

        xx = dat[d]['nf', 'nf', 'xval']
        axi = ax[0]
        axi.pcolormesh(np.flipud(xx), vmin=0, vmax = 1)
        ax[0].set_xticks([.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        ax[0].set_yticks([.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        lab = np.array(['NF 1', 'NF 2', 'NF 3', 'NF 4', 'Hold', 'Reach'])
        
        ax[0].set_xticklabels(lab)
        ax[0].set_yticklabels(lab[-1:0:-1])


        xx = dat[d]['mc', 'mc', 'xval'][4:, 4:]
        ax[1].pcolormesh(np.flipud(xx), vmin=0, vmax = 1)
        ax[1].set_xticks([.5, 1.5,])
        ax[1].set_yticks([.5, 1.5, ])
        lab = np.array(['Hold', 'Reach'])
        
        ax[1].set_xticklabels(lab)
        ax[1].set_yticklabels(lab[-1:0:-1])

        perc_corr_mc = dat[d]['mc','mc', 'perc_corr']
        perc_corr_mc2 = float(perc_corr_mc[0])/perc_corr_mc[1]
        mc_perf.append(perc_corr_mc2)
        binom['mc_corr'] += perc_corr_mc[0]

        perc_corr_mc_ch = dat[d]['mc','mc', 'chance_perc_corr']
        perc_corr_mc_ch2 = float(perc_corr_mc_ch[0])/perc_corr_mc_ch[1]
        mc_ch.append(perc_corr_mc_ch2)
        binom['mc_ch'] += perc_corr_mc_ch[0]

        assert perc_corr_mc_ch[1]==perc_corr_mc[1]
        binom['mc_n'] += perc_corr_mc_ch[1]

        perc_corr_nf = dat[d]['nf','nf', 'perc_corr']
        perc_corr_nf2 = float(perc_corr_nf[0])/perc_corr_nf[1]
        nf_perf.append(perc_corr_nf2)
        binom['nf_corr'] += perc_corr_nf[0]

        perc_corr_nf_ch = dat[d]['nf','nf', 'chance_perc_corr']
        perc_corr_nf_ch2 = float(perc_corr_nf_ch[0])/perc_corr_nf_ch[1]
        nf_ch.append(perc_corr_nf_ch2)
        binom['nf_ch'] += perc_corr_nf_ch[0]
        assert perc_corr_nf_ch[1]==perc_corr_nf[1]
        binom['nf_n'] += perc_corr_nf_ch[1]

        ax[2].bar(0, perc_corr_mc2, color='red')
        x[0].append(perc_corr_mc2)

        ax[2].bar(1, perc_corr_mc_ch2, color='red', alpha=.5)
        x[1].append(perc_corr_mc_ch2)

        ax[2].bar(3, perc_corr_nf2, color='blue')
        x[3].append(perc_corr_nf2)

        ax[2].bar(4, perc_corr_nf_ch2, color='blue', alpha=.5)
        x[4].append(perc_corr_nf_ch2)

        ax[2].set_xticks([.5, 1.5, 3.5, 4.5])
        ax[2].set_xticklabels(['MC Perc. Corr.', 'MC Chance', 'NF Perc. Corr', 'NF Chance'])
        ax[2].set_ylabel('LDA Percent Correct')
        #plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/NeuronPaper/JNeuroDraft/lda_within_task_vs_chance_'+d+'.eps', format='eps', dpi=300)

    t1, p1 = scipy.stats.ttest_rel(mc_perf, mc_ch)
    t2, p2 = scipy.stats.ttest_rel(nf_perf, nf_ch)
    print 'mc repated measures ttest: ', p1, t1
    print 'nf repated measures ttest: ', p2, t2

    # Binomial Test #
    pv = scipy.stats.binom_test(binom['mc_ch'], binom['mc_n'], binom['mc_corr']/float(binom['mc_n']))    
    pv2 = scipy.stats.binom_test(binom['nf_ch'], binom['nf_n'], binom['nf_corr']/float(binom['nf_n']))  

    print 'binomial test: ', ' manual control: ', pv, binom['mc_n'], 'nf: ', pv2, binom['nf_n']

    ax2.bar(0, np.mean(x[0]), color='red')
    ax2.bar(1, np.mean(x[1]), color='red', alpha=0.5)
    ax2.bar(3, np.mean(x[3]), color='blue')
    ax2.bar(4, np.mean(x[4]), color='blue', alpha=0.5)
    ax2.set_xticks([.5, 1.5, 3.5, 4.5])
    ax2.set_xticklabels(['MC Perc. Corr.', 'MC Chance', 'NF Perc. Corr', 'NF Chance'])
    ax2.set_ylabel('LDA Percent Correct')
    plt.tight_layout()

def plot_6class_lda_x_tasks(fname, save=False):
    #cmap = ['maroon', 'firebrick', 'orangered', 'darksalmon', 'powderblue', 'lightslategrey']
    cmap = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]
    dat = pickle.load(open(fname))
    f, ax = plt.subplots(figsize=(7, 7))

    hld_mn = []
    rch_mn = []

    heeld = []
    reech = []

    for i_d, d in enumerate(np.sort(dat.keys())):
        data_mats = dat[d]
        ix = 0

        data_trn_mc = data_mats['mc', 'nf', ix]
        sm_arr = []
        mn_arr = []
        err_arr = []
        for j in [4, 5]:
            sm = 0
            for i, n in enumerate(data_trn_mc[:4, j]):
                sm += (i+1)*n 
                mn_arr.extend([i+1]*n)
            sm_arr.append(sm/np.sum(data_trn_mc[:4, j]))
            err_arr.append(np.std(mn_arr)/np.sqrt(len(mn_arr)))
            
        hold = np.hstack(([[i]*data_trn_mc[i, 4] for i in range(4)]))
        rch = np.hstack(([[i]*data_trn_mc[i, 5] for i in range(4)]))
        u, p = scipy.stats.mannwhitneyu(hold, rch)
        # if i_d <2:
        #     lab = '$p = $'+str(int(1000*p)/1000.)
        # else:
        #     lab = '$p < 0.001$'
        lab = d
        if i_d in [0, 1]:
            rnd = 0
        elif i_d ==2:
            rnd = -.05
        else:
            rnd = np.random.rand()
        ax.errorbar([0+(0.1*rnd), 1+(0.1*rnd)], sm_arr, yerr=err_arr, color=tuple(np.array(cmap[i_d])/255.), label=lab, linewidth=5)
        print np.mean(rch)-np.mean(hold), np.median(rch)-np.median(hold), 'p: ', p
        hld_mn.append(np.mean(hold))
        heeld.append(hold)

        rch_mn.append(np.mean(rch))
        reech.append(rch)

        data_trn_nf = data_mats['nf', 'mc', ix].T
        sm_arr = []
        for j in [4, 5]:
            sm = 0
            for i, n in enumerate(data_trn_nf[:4, j]):
                sm += (i+1)*n 
            sm_arr.append(sm/np.sum(data_trn_nf[:4, j]))

        #ax.plot([0, 1], sm_arr, 'r.-')
        #hold = np.hstack(([[i]*data_trn_nf[i, 4] for i in range(4)]))
        #rch = np.hstack(([[i]*data_trn_nf[i, 5] for i in range(4)]))
        #u, p = scipy.stats.mannwhitneyu(hold, rch)
        #print np.mean(rch)-np.mean(hold), np.median(rch)-np.median(hold), 'p: ', p

    #print repeated measures ttest: 
    t, p = scipy.stats.ttest_rel(hld_mn, rch_mn)
    print 'repeated ttest: ', p, t
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Hold', 'Reach'])
    ax.set_xlim([-.5, 1.5, ])
    ax.set_ylabel('Average NF Target in Class')
    #ax.set_title('Train with MC, Test with NF')
    ax.legend(loc=3,fontsize=14)
    plt.tight_layout()
 
    p = []
    p2 = []
    reech = np.hstack((reech))
    heeld = np.hstack((heeld))

    for i in np.unique(reech):
        ix = np.nonzero(reech==i)[0]
        p.append(len(ix))

        ix2 = np.nonzero(heeld==i)[0]
        p2.append(len(ix2))

    p = np.array(p)
    n_p = float(np.sum(p))

    p2 = np.array(p2)
    n_p2 = float(np.sum(p2))

    if n_p < n_p2:
        p2_adj = p2/n_p2*n_p
        p1_adj = p
    elif n_p >= n_p2:
        p2_adj = p2
        p1_adj = p/n_p*n_p2

    chai, pv = scipy.stats.chisquare(p1_adj.astype(int), p2_adj.astype(int))
    print 'multinomial: ', chai, pv, sum(p), '<-- fast', sum(p2), '<-- slow'



    if save:
        plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/NeuronPaper/JNeuroDraft/6classes_lda_trn_mc_test_nf_rel_ttest_xoffs.eps', format='eps', dpi=300)






