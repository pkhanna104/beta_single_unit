import predict_kin_w_spk
import sklearn.linear_model
import state_space_spks as sss
import state_space_cart as ssc
import numpy as np
import matplotlib.pyplot as plt
import state_space_w_beta_bursts as ssbb
import pickle
import scipy.stats
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.2,style='white')

cmap1 = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]
cmap2  = ['lightseagreen','dodgerblue','gold','orangered','k']

def pre_process(day, test, all_cells, animal, bin, lags):
    if test:
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=None)
    else:
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=[day])

    binsize = bin
    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_sig = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize, smooth = 50,  animal=animal)

    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
        beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=binsize, animal=animal)

    #Smoosh together binary kin signals
    KC_bin_master = {}
    RT_bin_master = {}
    LAB_master = {}

    for i_d, day in enumerate(days):
        kbn_c = []
        rt = []
        lab = []
        spd = []

        for i_b, b in enumerate(blocks[i_d]):
            kbn_c.append(binned_kin_signal_dict[b, day])
            rt.append(rt_sig[b, day])
            lab.append(lfp_lab[b, day])

        if animal == 'grom':
            KC_bin_master[day] = 100*np.vstack((kbn_c))
        else:
            KC_bin_master[day] = np.vstack((kbn_c))
        
        RT_bin_master[day] = np.hstack((rt))
        LAB_master[day] = np.hstack((lab))


    # Go through and train on CO RTs: 
    # Get CO trials:
    co_ix = np.nonzero(LAB_master[day] < 80)[0]
    nf_ix = np.nonzero(LAB_master[day] > 80)[0]

    X = []
    X2 = []

    dist_from_onset = []
    dist_from_onset2 = []

    tm_trl_key = []
    tm_trl_key2 = []

    Y = []
    Y2 = []

    trl2 = []

    binary_beta = []
    binary_beta2 = []
    kc_bin_master = []
    kc_bin_master2 = []

    #Use points before reach that are less than 3.5 cm/sec and after reach that are greater than 3.5 cm/sec
    for it, trl in enumerate(LAB_master[day]):
        rt_bin = RT_bin_master[day][it]
        spd = np.sqrt(np.sum(KC_bin_master[day][it, [2, 3], :]**2, 0))
        
        for tm in range(lags, int(2500/binsize)):
            ad = 0
            skip = False

            if it in co_ix:

                if np.logical_and(np.mean(spd[tm-lags:tm]) < 100, tm < rt_bin):
                    X.append(S_bin[day][it, tm-lags:tm, :].reshape(-1))
                    Y.append(0)
                    dist_from_onset.append(rt_bin - tm)
                    tm_trl_key.append([tm, it])
                    L = list(B_bin[day][it, tm-lags:tm])
                    binary_beta.append(max(set(L), key=L.count))
                    ad = 1


                elif np.logical_and(np.mean(spd[tm-lags:tm]) > 0, tm >= rt_bin):
                    X.append(S_bin[day][it, tm-lags:tm, :].reshape(-1))
                    Y.append(1)
                    dist_from_onset.append(rt_bin - tm)
                    tm_trl_key.append([tm, it])
                    L = list(B_bin[day][it, tm-lags:tm])
                    binary_beta.append(max(set(L), key=L.count))
                    ad = 1
                
                else:
                    skip = True

                if not skip:
                    kc_bin_master.append(KC_bin_master[day][it, [2, 3]][:, tm])

            elif it in nf_ix:
                if np.logical_and(np.mean(spd[tm-lags:tm]) < 100, tm < rt_bin):
                    X2.append(S_bin[day][it, tm-lags:tm, :].reshape(-1))
                    Y2.append(0)
                    dist_from_onset2.append(rt_bin - tm)
                    trl2.append(LAB_master[day][it])
                    tm_trl_key2.append([tm, it])
                    L = list(B_bin[day][it, tm-lags:tm])
                    binary_beta2.append(max(set(L), key=L.count)) 
                    ad = 1
                    
            
                elif np.logical_and(np.mean(spd[tm-lags:tm]) > 0, tm >= rt_bin):
                    X2.append(S_bin[day][it, tm-lags:tm, :].reshape(-1))
                    Y2.append(1)
                    dist_from_onset2.append(rt_bin - tm)
                    trl2.append(LAB_master[day][it])
                    tm_trl_key2.append([tm, it])
                    L = list(B_bin[day][it, tm-lags:tm])
                    binary_beta2.append(max(set(L), key=L.count))
                    ad = 1
                else:
                    skip = True

                if not skip:
                    kc_bin_master2.append(KC_bin_master[day][it, [2, 3]][:, tm])
            
     
    X = np.vstack((X))
    T0 = X.shape[0]
    mnX = np.mean(X, axis=0)
    mnX_tile = np.tile(mnX[np.newaxis, :], [T0, 1])
    stdX = np.std(X, axis=0)
    stdX_tile = np.tile(stdX[np.newaxis, :], [T0, 1])
    demean_X = (X - mnX_tile)/stdX_tile

    ix = np.random.permutation(T0)
    ix0 = ix[:int(T0/1.5)]
    ix1 = ix[int(T0/1.5):]
    demean_X_train = demean_X[ix0, :]
    demean_X_test = demean_X[ix1, :]

    Y = np.hstack((Y))
    Y_train = Y[ix0]
    Y_test = Y[ix1]

    X2 = np.vstack((X2))
    T = X2.shape[0]
    demean_X2 = (X2 - np.tile(mnX[np.newaxis, :], [T, 1]))/np.tile(stdX[np.newaxis, :], [T, 1])
    Y2 = np.hstack((Y2))        
    trl2 = np.hstack((trl2))

    tm_trl_key = np.vstack((tm_trl_key))
    tm_trl_key_test = tm_trl_key[ix1, :]

    tm_trl_key2 = np.vstack((tm_trl_key2))

    binary_beta = np.hstack((binary_beta))
    binary_beta_test = binary_beta[ix1]
    binary_beta2 = np.hstack((binary_beta2))

    kc_bin_master = np.vstack((kc_bin_master))
    kc_bin_master2 = np.vstack((kc_bin_master2))
    
    #Incorporate lags: 
    # Train Logistic Regression: 
    lg = sklearn.linear_model.LogisticRegression(fit_intercept=True)
    lg.fit(demean_X_train, Y_train)
    y0 = lg.predict(demean_X_test)

    ##########################################
    ###### Train individual models too: ######
    ##########################################
    score_ind_units = []
    nunits = demean_X.shape[1]/lags
    for i in range(nunits):
        iix = [i, nunits+i, (2*nunits) + i]
        lgi = sklearn.linear_model.LogisticRegression(fit_intercept=True)
        lgi.fit(demean_X[:, iix], Y)

        # score: 
        ftii = -1*np.squeeze(np.array((np.mat(lgi.coef_)*demean_X[:, iix].T + lg.intercept_)))
        scii = 1/(1+np.exp(ftii))

        ftiix = -1*np.squeeze(np.array((np.mat(lgi.coef_)*demean_X2[:, iix].T + lg.intercept_)))
        sciix = 1/(1+np.exp(ftiix))

        score_ind_units.append([lgi.coef_, lgi.intercept_, scii, sciix])


    # Predict
    ft = -1*np.squeeze(np.array((np.mat(lg.coef_)*demean_X_test.T + lg.intercept_)))
    sc = 1/(1+np.exp(ft))

    ft_all = -1*np.squeeze(np.array((np.mat(lg.coef_)*demean_X.T + lg.intercept_)))
    sc_all = 1/(1+np.exp(ft_all))

    assert sc.shape[0] != sc_all.shape[0]

    distance_from_hyperplane = sc - 0.5
    distance_from_onset_test = np.hstack((dist_from_onset))[ix1]
    plt.plot(distance_from_hyperplane, distance_from_onset_test, '.')
    plt.xlabel('Distance from Hyperplane')
    plt.ylabel('Dist from Move Onset')

    co_chance = []
    for i in range(10):
        ix = np.random.permutation(Y_test.shape[0])
        co_chance.append(lg.score(demean_X_test, Y_test[ix]))

    plt.title(day+', perc corr: '+str(lg.score(demean_X_test, Y_test))+ '\n'+', perc_chance: '+str(np.mean(co_chance)))

    # Test on NF: 
    y1 = lg.predict(demean_X2)

    nf_chance = []
    for i in range(10):
        ix = np.random.permutation(Y2.shape[0])
        nf_chance.append(lg.score(demean_X2, Y2[ix]))

    ft2 = -1*np.squeeze(np.array((np.mat(lg.coef_)*demean_X2.T + lg.intercept_)))
    sc2 = 1/(1+np.exp(ft2))
    distance_from_hyperplane = sc2 - 0.5
    dist_from_onset2 = np.array(dist_from_onset2)

    cmap =['g','b','y','r']

    for it, t in enumerate(range(84, 88)):
        ix = np.nonzero(trl2 == t)[0]
        plt.plot(distance_from_hyperplane[ix], dist_from_onset2[ix], cmap[it]+'.')

    save_dict = dict()

    assert demean_X.shape[0] == kc_bin_master.shape[0]

    save_dict['co_spks_all'] = demean_X
    save_dict['co_label_all'] = Y
    save_dict['co_spks'] = demean_X_test
    save_dict['co_label'] = Y_test
    save_dict['co_t2rt_all'] = dist_from_onset
    save_dict['co_t2rt'] = distance_from_onset_test
    save_dict['co_log_R'] = lg
    save_dict['score_co_all'] = sc_all - 0.5
    save_dict['score_co'] = sc - 0.5
    save_dict['co_y0'] = y0
    save_dict['co_tm_trl_key_all'] = tm_trl_key
    save_dict['co_tm_trl_key'] = tm_trl_key_test
    save_dict['co_binary_beta_all'] = binary_beta
    save_dict['co_binary_beta'] = binary_beta_test
    save_dict['co_chance_perf'] = co_chance
    save_dict['co_kin'] = kc_bin_master
    save_dict['co_ind_log_R'] = score_ind_units
    save_dict['co_unit_mns'] = mnX
    save_dict['co_unit_stds'] = stdX

    save_dict['nf_spks'] = demean_X2
    save_dict['nf_label'] = Y2
    save_dict['nf_t2rt'] = dist_from_onset2
    save_dict['score_nf'] = sc2 - 0.5
    save_dict['nf_y0'] = y1
    save_dict['nf_targ_num'] = trl2
    save_dict['nf_tm_trl_key'] = tm_trl_key2
    save_dict['nf_binary_beta'] = binary_beta2
    save_dict['nf_chance_perf'] = nf_chance
    save_dict['nf_kin'] = kc_bin_master2

    save_dict['S_bin'] = S_bin
    save_dict['lfp_lab'] = lfp_lab

    return save_dict, day

def get_autocorrelations(day, test, all_cells, animal):
    if test:
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=None)
    else:
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=[day])

    binsize = 1
    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_sig = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize, smooth = 50,  animal=animal)
    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
        beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=binsize, animal=animal)

    kbn_c = []
    for i_b, b in enumerate(blocks[0]):
        kbn_c.append(bin_kin_signal[b, day])
    kin_bin = np.vstack((kbn_c))


    BB = B_bin[day]
    dBB = np.diff(B_bin[day], axis=1)
    nunits = S_bin[day].shape[2]
    autoc = np.zeros((nunits, 400))
    autoc_off = np.zeros((nunits, 400))
    cnts = np.zeros((nunits, 400))
    cnts_off = np.zeros((nunits, 400))

    for i in range(nunits):
        for trl in range(S_bin[day].shape[0]):
            dbb = dBB[trl, :]
            for j, (st, nd, ac, ct) in enumerate(zip([1, -1], [-1, 1], [autoc, autoc_off], [cnts, cnts_off])):
                nsegs = np.nonzero(dbb==st)[0]
                for n in nsegs:
                    ix = np.nonzero(dbb==nd)[0]
                    ix = ix[ix > n]
                    if len(ix)< 1:
                        ix = np.array([dBB.shape[1] - 1])

                    end = np.argmin(np.abs(ix-n))
                    seg = S_bin[day][trl, n:ix[end], i]
                    kseg = kin_bin[trl, n:ix[end]]

                    if np.sum(kseg) == 0:
                        spk = np.nonzero(seg)[0]
                        for s in spk:
                            if s < 200:
                                ac[i, 200-s:np.min([400, 200+len(seg)-s])] += seg[:np.min([s+200, len(seg)])]
                                ct[i, 200-s:np.min([400, 200+len(seg)-s])] += 1
                            else:
                                ac[i, :np.min([400, 200+len(seg)-s])] += seg[s-200:np.min([s+200, len(seg)])]
                                ct[i, :np.min([400, 200+len(seg)-s])] += 1

    return autoc, cnts, autoc_off, cnts_off

def get_AC_all_days(animal):
    if animal == 'grom':
        master_days = sss.master_days
        all_cells = False

    elif animal == 'cart':
        master_days = ssc.master_days
        all_cells = True

    master_dict = {}
    for i_d, day in enumerate(master_days):
        autoc, cnts, autoc_off, cnts_off = get_autocorrelations(day, False, all_cells, animal)
        master_dict[day] = autoc
        master_dict[day, 'cnts'] = cnts
        master_dict[day, 'off'] = autoc_off
        master_dict[day, 'cnts_off'] = cnts_off
        
        pickle.dump(master_dict, open('/home/lab/code/beta_single_unit/c_data/'+animal+'_'+day+'_auto_correlations.pkl', 'wb'))
    pickle.dump(master_dict, open('/home/lab/code/beta_single_unit/c_data/'+animal+'_master_auto_correlations_pls_off.pkl', 'wb'))

def try_dpca(X, labels):

    day = '022715'
    test = False
    all_cells = False
    animal = 'grom'
    bin = 10
    lags = 0
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=[day])
    binsize = bin
    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_sig = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize, smooth = 50,  animal=animal)
    #New -- binary kin signal

    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
        beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=binsize, animal=animal)

    KC_bin_master = {}
    RT_bin_master = {}
    LAB_master = {}

    for i_d, day in enumerate(days):
        kbn_c = []
        rt = []
        lab = []
        spd = []
        S_bin_smooth = {}

        for i_b, b in enumerate(blocks[i_d]):
            kbn_c.append(binned_kin_signal_dict[b, day])
            rt.append(rt_sig[b, day])
            lab.append(lfp_lab[b, day])

            S_bin_smooth[day].append


        if animal == 'grom':
            KC_bin_master[day] = 100*np.vstack((kbn_c))
        else:
            KC_bin_master[day] = np.vstack((kbn_c))
        
        RT_bin_master[day] = np.hstack((rt))
        LAB_master[day] = np.hstack((lab))
        LAB_master[day][LAB_master[day] < 80] = 88

    for i in np.unique(LAB_master[day]):
        ix = np.nonzero(LAB_master[day]==i)[0]
        print len(np.nonzero(RT_bin_master[day][ix] > 73)[0]), len(np.nonzero(RT_bin_master[day][ix] <= 73)[0]) 
    

    X = np.zeros((6, 31, 4, 2, 100))
    rt_cutoff = 72

    cnts = np.zeros((5, 2))
    for i in range(S_bin[day].shape[0]):
        ix = LAB_master[day][i] - 84
        if ix < 4:
            if RT_bin_master[day][i] > rt_cutoff:
                rtix = 1
            else:
                rtix = 0

            if cnts[ix, rtix] < 6:
                X[int(cnts[ix, rtix]), :, ix, rtix, :] = S_bin[day][i, :, :].T
                cnts[ix, rtix]+= 1

    # number of neurons, time-points and stimuli
    trl, N, S, R, T = X.shape

    trialR = X.copy()

    # trial-average data
    R = mean(trialR,0)

    # center data
    R -= mean(R.reshape((N,-1)),1)[:,None,None,None]

    from dPCA import dPCA
    dpca = dPCA.dPCA(labels='srt',regularizer='auto')
    dpca.protect = ['t']
    Z = dpca.fit_transform(R,trialR)

    time = arange(T)

    figure(figsize=(16,7))
    subplot(131)

    for s in range(S):
        for r in range(2):
            if r ==0:
                plot(time,Z['r'][0, s, r, :],'k-')
            elif r == 1:
                plot(time,Z['r'][0, s, r, :], 'r-')
    title('1st rate component')
        
    subplot(132)

    for s in range(S):
        for r in range(2):
            plot(time,Z['s'][0,s, r, :])
        
    title('1st beta targ component')
        
    subplot(133)

    colors = ['k', 'b', 'g','r']
    dot = ['--', '-']
    for s in range(S):
        col = colors[s]
        for r in range(2):
            colrn = col+dot[r]
            plot(time,Z['sr'][0,s, r, :], colrn)
        
    title('1st rate_x_beta targ component')
    show()

def get_all_days(animal, bin, lags):
    if animal == 'grom':
        master_days = sss.master_days
        all_cells = False

    elif animal == 'cart':
        master_days = ssc.master_days
        all_cells = True

    master_dict = {}
    for i_d, day in enumerate(master_days):
        d_dict = pre_process(day, False, all_cells, animal, bin, lags)
        master_dict[day] = d_dict
        pickle.dump(d_dict, open('/home/lab/code/beta_single_unit/c_data/'+animal+'_'+day+'_'+str(bin)+'_lag_'+str(lags)+'_logistic_regression_zsc_traintest_dict.pkl', 'wb'))
    pickle.dump(master_dict, open('/home/lab/code/beta_single_unit/c_data/'+animal+'_'+str(bin)+'_lag_'+str(lags)+'_master_logistic_regression_zsc_traintest_dict.pkl', 'wb'))

def plot_co_traj(animal, bin, lags):

    srch_str = '/home/lab/code/beta_single_unit/c_data/'+animal+'_'+str(bin)+'_lag_'+str(lags)+'_master*'
    import glob
    fnm = glob.glob(srch_str)

    if len(fnm) == 1:
        fnm = fnm[0]
    else:
        raise Exception

    dat = pickle.load(open(fnm))
    days = dat.keys()

    for i_d, dy in enumerate(days):
        d = dat[dy][0]

        # Figure time
        f, ax = plt.subplots(nrows=2, ncols=2)

        # Trajectory plot
        axi = ax[0, 0]
        axi = traj_plot(axi, d, cmap2)

        # Bar plots: 
        axi = ax[0, 1]
        axi = bar_plots(axi, d, cmap2, tsk='co')
        axi.set_title('CO by Beta On/Off')

        axi = ax[1, 0]
        axi = bar_plots(axi, d, cmap2, tsk='nf')
        axi.set_title('NF by Beta On/Off')

        axi = ax[1, 1]
        axi = bar_plots(axi, d, cmap2, tsk='nf', sort_by_targ=True)
        axi.set_title('NF by Beta target')

        f2, ax2 = plt.subplots()
        ax2 = perc_chance(ax2, d)

    plot_chance_all(dat)

def perc_chance(axi, d):
    co_perc_corr = len(np.nonzero(d['co_y0'] + d['co_label'] != 1)[0])/float(len(d['co_y0']))
    co_perc_ch = np.mean(d['co_chance_perf'])

    nf_perc_corr = len(np.nonzero(d['nf_y0'] + d['nf_label'] != 1)[0])/float(len(d['nf_y0']))
    nf_perc_ch = np.mean(d['nf_chance_perf'])

    axi.bar(0, co_perc_corr)
    axi.bar(1, np.mean(co_perc_ch))
    axi.bar(2, nf_perc_corr)
    axi.bar(3, np.mean(nf_perc_ch))

def plot_chance_all(dat):
    f, ax = plt.subplots()

    co_perf = []
    co_ch = []

    nf_perf = []
    nf_ch = []

    days = dat.keys()
    for i_d, dy in enumerate(np.sort(days)):
        d = dat[dy][0]

        co_add = d['co_y0'] + d['co_label']
        co_perf.append(len(np.nonzero(co_add != 1)[0])/float(len(d['co_y0'])))
        co_ch.append(np.mean(d['co_chance_perf']))

        nf_add = d['nf_y0'] + d['nf_label']
        nf_perf.append(len(np.nonzero(nf_add != 1)[0])/float(len(d['nf_y0'])))
        nf_ch.append(np.mean(d['nf_chance_perf']))

    ax.bar(0, np.mean(co_perf), color='blue', width = .6)
    ax.bar(1, np.mean(co_ch), color='blue', width = .6)
    ax.bar(2, np.mean(nf_perf), color='red', width = .6)
    ax.bar(3, np.mean(nf_ch), color='red', width = .6)
    
    ax.errorbar(0+.3, np.mean(co_perf), yerr=np.std(co_perf)/np.sqrt(len(co_perf)), color='k')
    ax.errorbar(1+.3, np.mean(co_ch), yerr=np.std(co_ch)/np.sqrt(len(co_ch)), color='k')
    ax.errorbar(2+.3, np.mean(nf_perf), yerr=np.std(nf_perf)/np.sqrt(len(nf_perf)), color='k')
    ax.errorbar(3+.3, np.mean(nf_ch),  yerr=np.std(nf_ch)/np.sqrt(len(nf_ch)), color='k')

def plot_perc_all():
    d2 = pickle.load(open('grom_25_lag_3_master_logistic_regression_zsc_traintest_dict.pkl'))
    d  = pickle.load(open('cart_25_lag_3_master_logistic_regression_zsc_traintest_dict.pkl'))
    offs = [0, .315]
    subj = ['grom', 'cart']

    co_0 = []
    co_1 = []

    nf_0 = []
    nf_1 = []

    for i, dat in enumerate([d2, d]):
        days = dat.keys()

        co_0day = []
        co_1day = []
        nf_0day = []
        nf_1day = []

        for i_d, dy in enumerate(np.sort(days)):
            d = dat[dy][0]
            co_ = d['co_label']

            ix0 = np.nonzero(co_==0)[0]
            ix1 = np.nonzero(co_ ==1)[0]

            co_0.append(d['score_co'][ix0])
            co_1.append(d['score_co'][ix1])
            co_0day.append(d['score_co'][ix0])
            co_1day.append(d['score_co'][ix1])

            nf_ = d['nf_label']
            ix0 = np.nonzero(nf_==0)[0]
            ix1 = np.nonzero(nf_==1)[0]

            nf_0.append(d['score_nf'][ix0])
            nf_1.append(d['score_nf'][ix1])
            nf_0day.append(d['score_nf'][ix0])
            nf_1day.append(d['score_nf'][ix1])

            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            
            # fpr, tpr, _ = sklearn.metrics.roc_curve(d['co_label'], d['score_co']) 
            # roc_auc =  sklearn.metrics.auc(fpr, tpr)
            # #print 'co auc: ', roc_auc

            # fpr, tpr, _ = sklearn.metrics.roc_curve(d['nf_label'], d['score_nf']) 
            # roc_auc =  sklearn.metrics.auc(fpr, tpr)
            # #print 'nf auc: ', roc_auc
        f, ax = plt.subplots()

        co0 = np.hstack((co_0day))
        co1 = np.hstack((co_1day))
        nf0 = np.hstack((nf_0day))
        nf1 = np.hstack((nf_1day))

        ax.bar(0, np.mean(co0)-offs[i], color='blue', width = .6)
        ax.bar(1, np.mean(co1)-offs[i], color='blue', width = .6)
        ax.bar(2, np.mean(nf0)-offs[i], color='red', width = .6)
        ax.bar(3, np.mean(nf1)-offs[i], color='red', width = .6)
        
        ax.errorbar(0+.3, np.mean(co0)-offs[i], yerr=np.std(co0)/np.sqrt(len(co0)), color='k')
        ax.errorbar(1+.3, np.mean(co1)-offs[i], yerr=np.std(co1)/np.sqrt(len(co1)), color='k')
        ax.errorbar(2+.3, np.mean(nf0)-offs[i], yerr=np.std(nf0)/np.sqrt(len(nf0)), color='k')
        ax.errorbar(3+.3, np.mean(nf1)-offs[i],  yerr=np.std(nf1)/np.sqrt(len(nf1)), color='k')

        plt.savefig('/home/lab/code/beta_single_unit/c_data/'+subj[i]+'_fig7bar.eps', format='eps', dpi=300)

        co00 = np.array([np.mean(co) for co in co_0day])
        co11 = np.array([np.mean(co) for co in co_1day])
        nf00 = np.array([np.mean(co) for co in nf_0day])
        nf11 = np.array([np.mean(co) for co in nf_1day])
        print ''
        print ''
        print ' SUBJ: ', subj[i]
        print 'CO ttest:', scipy.stats.ttest_rel(co00, co11) , 'n = ', len(co00), len(co11)
        print 'NF ttest:', scipy.stats.ttest_rel(nf00, nf11) , 'n= ', len(nf00), len(nf11)

    
    co00 = np.array([np.mean(co) for co in co_0])
    co11 = np.array([np.mean(co) for co in co_1])
    nf00 = np.array([np.mean(co) for co in nf_0])
    nf11 = np.array([np.mean(co) for co in nf_1])
    print ''
    print ''
    print ' ALL: '
    print 'CO ttest:', scipy.stats.ttest_rel(co00, co11) , 'n = ', len(co00), len(co11)
    print 'NF ttest:', scipy.stats.ttest_rel(nf00, nf11) , 'n= ', len(nf00), len(nf11)

def plot_pretty_traj(dat, day, binsize):
    
    # GO aligned
    d = dat[day][0]

    co_trls = np.unique(d['co_tm_trl_key'][:, 1])
    CO_DICT = np.zeros((int(2500/binsize), len(co_trls)))
    CO_DICT[:, :] = np.nan

    co_t2rt = -1*np.array(d['co_t2rt'])
    co_score = d['score_co']
        
    for it, trl in enumerate(co_trls):
        ix = np.nonzero(d['co_tm_trl_key'][:, 1]==trl)[0]
        ix_sort = np.argsort(d['co_tm_trl_key'][ix, 0])
        CO_DICT[d['co_tm_trl_key'][ix[ix_sort], 0], it] = co_score[ix[ix_sort]]

    
    nf_trls = np.unique(d['nf_tm_trl_key'][:, 1])
    NF_DICT = np.zeros((int(2500/binsize), len(nf_trls), 4))
    NF_DICT[:, :, :] = np.nan
    nf_t2rt = -1*np.array(d['nf_t2rt'])
    nf_score = d['score_nf']
    nf_targ_num = d['nf_targ_num']
        
    for it, trl in enumerate(nf_trls):
        ix = np.nonzero(d['nf_tm_trl_key'][:, 1]==trl)[0]
        if len(ix) > 0:
            ix_sort = np.argsort(d['nf_tm_trl_key'][ix, 0])
            nf_targ = int(nf_targ_num[ix[0]]) - 84

            NF_DICT[d['nf_tm_trl_key'][ix[ix_sort], 0], it, nf_targ] = nf_score[ix[ix_sort]]

    f, ax = plt.subplots()

    t = np.arange(-1.5, 1.0, binsize/1000.)
    plot_mean_and_sem(t, CO_DICT, ax, color='k')

    for n in range(4):
        nf_dict = NF_DICT[:, :, n]
        plot_mean_and_sem(t, nf_dict, ax, color=cmap2[n])


    # RT aligned
    d = dat[day][0]

    co_trls = np.unique(d['co_tm_trl_key'][:, 1])
    co_t2rt = -1*np.array(d['co_t2rt'])
    co_t2rt_adj = co_t2rt + np.abs(np.min(co_t2rt))
    CO_DICT = np.zeros((np.abs(np.min(co_t2rt))+np.max(co_t2rt)+1, len(co_trls)))
    CO_DICT[:, :] = np.nan

    t_co = np.arange(np.min(co_t2rt)*binsize/1000., (np.max(co_t2rt)+1)*binsize/1000., binsize/1000.)
    co_score = d['score_co']
        
    for it, trl in enumerate(co_trls):
        ix = np.nonzero(d['co_tm_trl_key'][:, 1]==trl)[0]
        ix_sort = np.argsort(co_t2rt_adj[ix])
        CO_DICT[co_t2rt_adj[ix[ix_sort]], it] = co_score[ix[ix_sort]]


    nf_trls = np.unique(d['nf_tm_trl_key'][:, 1])
    nf_t2rt = -1*np.array(d['nf_t2rt'])
    nf_t2rt_adj = nf_t2rt + np.abs(np.min(nf_t2rt))
    NF_DICT = np.zeros((np.abs(np.min(nf_t2rt))+np.max(nf_t2rt)+1, len(nf_trls), 4))
    NF_DICT[:, :] = np.nan

    t_nf = np.arange(np.min(nf_t2rt)*binsize/1000., (np.max(nf_t2rt)+1)*binsize/1000., binsize/1000.)
    nf_score = d['score_nf']
    nf_targ_num = d['nf_targ_num']
        
    for it, trl in enumerate(nf_trls):
        ix = np.nonzero(d['nf_tm_trl_key'][:, 1]==trl)[0]
        if len(ix) > 0:
            ix_sort = np.argsort(nf_t2rt_adj[ix])
            nf_targ = int(nf_targ_num[ix[0]]) - 84
            NF_DICT[nf_t2rt_adj[ix[ix_sort]], it, nf_targ] = nf_score[ix[ix_sort]]

    f, ax = plt.subplots()

    plot_mean_and_sem(t_co, CO_DICT, ax, color='k')

    for n in range(4):
        nf_dict = NF_DICT[:, :, n]
        plot_mean_and_sem(t_nf, nf_dict, ax, color=cmap2[n])

def plot_mean_and_sem(t ,array, ax, color='b', array_axis=1, label='0',
    log_y=False, make_min_zero=[False,False]):
    eps = 10**-10;

    mean = np.nanmean(array, axis=array_axis)
    n = np.array([ np.sqrt(len(np.nonzero(~np.isnan(array[i, :]))[0])) for i in range(array.shape[0]) ])
    sem = np.nanstd(array, axis=array_axis) / (n + eps) 

    sem_plus = mean + sem
    sem_minus = mean - sem
     
    if make_min_zero[0] is not False:
        bi, bv = get_in_range(x,make_min_zero[1])
        add = np.min(mean[bi])
    else:
        add = 0
 
    ax.fill_between(t, sem_plus-add, sem_minus-add, color=color, alpha=0.5)
    ax.plot(t,mean-add, '-',color=color,label=label)
    if log_y:
        ax.set_yscale('log')
    return ax

def bar_plots(axi, d, cmap2, tsk='nf', sort_by_targ=False):
    # Beta on vs. beta off plots 

    tsk_trls = np.max(d[tsk+'_tm_trl_key'][:, 1])
    tsk_t2rt = -1*np.array(d[tsk+'_t2rt'])
    tsk_score = np.array(d['score_'+tsk])

    if sort_by_targ:
        beta_tg = np.zeros((4, ))
        beta_cnt = np.zeros((4, ))
        label = d['nf_targ_num'] - 84
    else:
        beta_tg = np.zeros((2, ))
        beta_cnt = np.zeros((2, ))
        label = d[tsk+'_binary_beta']

    for i, L in enumerate(label):
        if tsk_t2rt[i] < 0:
            beta_tg[L] += tsk_score[i]
            beta_cnt[L] += 1

    for L in np.unique(label):
        axi.bar(L, beta_tg[L]/float(beta_cnt[L]))
    return axi

def line_plots_all():
    d2 = pickle.load(open('grom_25_lag_3_master_logistic_regression_zsc_traintest_dict.pkl'))
    d  = pickle.load(open('cart_25_lag_3_master_logistic_regression_zsc_traintest_dict.pkl'))
    tskd = dict(co={}, nf={})
    tskd['co'][0] = []
    tskd['co'][1] = []
    tskd['nf'][0] = []
    tskd['nf'][1] = []

    for i, (dat, offs) in enumerate(zip([d2, d], [0, .315])):
        days = dat.keys()
        f, axi = plt.subplots()

        for it, tsk in enumerate(['co', 'nf']):
            tskd[0] = []
            tskd[1] = []

            for i_d, dy in enumerate(np.sort(days)):
                d = dat[dy][0]

                if tsk == 'co':
                    key = tsk+'_tm_trl_key_all'
                    key1 = tsk+'_t2rt_all'
                    key2 = 'score_'+tsk+'_all'
                    key3 = tsk+'_binary_beta_all'
                else:
                    key = tsk+'_tm_trl_key'
                    key1 = tsk+'_t2rt'
                    key2 = 'score_'+tsk
                    key3 = tsk+'_binary_beta'

                tsk_trls = np.max(d[key][:, 1])
                tsk_t2rt = -1*np.array(d[key1])
                tsk_score = np.array(d[key2])

                beta_tg = dict()
                beta_tg[0] = []
                beta_tg[1] = []
                beta_cnt = np.zeros((2, ))
                label = d[key3]

                for i, L in enumerate(label):
                    #Only take the portion from before the onset:
                    if tsk_t2rt[i] < 0:
                        beta_tg[L].append(tsk_score[i])

                B0 = beta_tg[0]
                B1 = beta_tg[1]

                axi.plot(np.array([0, 1])+(2*it), [np.mean(B0)-offs, np.mean(B1)-offs], '.-',
                    color=np.array(cmap1[i_d])/255.)


                axi.errorbar(np.array([0, 1])+(2*it), [np.mean(B0)-offs, np.mean(B1)-offs], 
                    yerr=np.array([np.std(B0)/np.sqrt(len(B0)), np.std(B1)/np.sqrt(len(B1))]),
                    color=np.array(cmap1[i_d])/255.)

                tskd[tsk][0].append(np.mean(B0))
                tskd[tsk][1].append(np.mean(B1))
                tskd[0].append(np.mean(B0))
                tskd[1].append(np.mean(B1))

            print 'TASK: ', tsk
            print tsk+ ' ttest:', scipy.stats.ttest_rel(tskd[0], tskd[1]) , 'n = ', len(tskd[0]), len(tskd[1])

        axi.set_xlabel('CO, NF')
        axi.set_ylabel('Distance from Thresh')

    for it, tsk in enumerate(['co', 'nf']):
        print 'TASK: ', tsk
        print tsk+ ' ttest:', scipy.stats.ttest_rel(tskd[tsk][0], tskd[tsk][1]) , 'n = ', len(tskd[tsk][0]), len(tskd[tsk][1])
            
def traj_plot(axi, d, cmap2):
    co_trls = np.unique(d['co_tm_trl_key'][:, 1])
    co_t2rt = -1*np.array(d['co_t2rt'])
    co_score = d['score_co']
        
    for trl in co_trls:
        ix = np.nonzero(d['co_tm_trl_key'][:, 1]==trl)[0]
        ix_sort = np.argsort(d['co_tm_trl_key'][ix, 0])

        axi.plot(co_t2rt[ix[ix_sort]], co_score[ix[ix_sort]], 'k.-',
            linewidth=.2, markersize=3)

    nf_trls = np.unique(d['nf_tm_trl_key'][:, 1])
    nf_t2rt = -1*np.array(d['nf_t2rt'])
    nf_score = d['score_nf']
    nf_targ_num = d['nf_targ_num']
        
    for trl in nf_trls:
        ix = np.nonzero(d['nf_tm_trl_key'][:, 1]==trl)[0]
        if len(ix) > 0:
            ix_sort = np.argsort(d['nf_tm_trl_key'][ix, 0])
            nf_targ = int(nf_targ_num[ix[0]])

            axi.plot(nf_t2rt[ix[ix_sort]], nf_score[ix[ix_sort]], '.-',
                linewidth=.2, markersize=3,
                color=cmap2[nf_targ-84])

        else:
            print 'skipping trial: ', trl
    return axi

def analyze_wts(dat):
    units = dict()

    for i_d, dy in enumerate(np.sort(dat.keys())):
        d = dat[dy][0]
        lg = d['co_log_R']
        c = lg.coef_.T
        ccoll = c.reshape(3, len(c)/3)
        c_max = np.argmax(np.abs(ccoll), axis=0)
        CM = np.array([ccoll[cc, i] for i, cc in enumerate(c_max)])

        # Try to re-do w/ only top 25% of wts:
        R2 = []
        cm = dict()
        for ip, prc in enumerate(np.arange(0, 100, 100/(len(c)/3.))):
            thresh = np.percentile(np.abs(CM), prc)
            CM2 = np.zeros((3, len(CM) ))
            CM2[:, np.abs(CM) > thresh] = ccoll[:, np.abs(CM) > thresh]
            CM2 = CM2.reshape(-1)
            cm[ip] = CM2
            cm[ip, 'all'] = c

            # Score: 
            demean_X = d['co_spks_all'] # time x neurons
            ft = -1*np.squeeze(np.array((np.mat(CM2)*demean_X.T + lg.intercept_)))
            sc = (1/(1+np.exp(ft))) - 0.5
            Y_est = sc > 0

            demean_X = d['nf_spks'] # time x neurons
            demean_X2 = demean_X - np.mean(demean_X, axis=0)[None, :]
            demean_X3 = demean_X2/np.std(demean_X2, axis=0)[None, :]

            ft2 = -1*np.squeeze(np.array((np.mat(CM2)*demean_X3.T + lg.intercept_)))
            sc2 = (1/(1+np.exp(ft2))) - 0.5
            Y_est2 = sc2 > 0

            # Compare to Score: 
            sc_actual = d['score_co_all']
            sc_nf_all = d['score_nf']

            Y = d['co_label_all']
            Y2 = d['nf_label']

            s, i, rv, pv, se = scipy.stats.linregress(sc, sc_actual)
            s, i, rv2, pv, se = scipy.stats.linregress(sc2, sc_nf_all)

            print 'prc: ', prc, ' mc: ', rv**2, ', nf: ', rv2**2
            R2.append(np.min([rv**2, rv2**2]))

        ix = np.nonzero(np.array(R2) > 0.8)[0]
        IX = np.max(ix)

        # Units: 
        units[dy] = cm[IX]
        units[dy, 'nunits'] = len(cm[IX])
        units[dy, 'chosen'] =  np.nonzero(cm[IX])[0]
        units[dy, 'nonchosen'] =  np.nonzero(cm[IX] == 0)[0]
        units[dy, 'all'] = cm[IX, 'all']
    days = np.sort(dat.keys())
    return units, days, dat

def analyze_ind_log_R(dat):
    '''
    An alternative to analyze_wts that selects units based on 
    successful classification (on individual basis) instead of by wts 
    '''

    units = dict()
    days = np.sort(dat.keys())

    for i_d, day in enumerate(days):
        d = dat[day][0]
        nunits = d['S_bin'][day].shape[2]
        co_test_y = d['co_label_all']
        nf_test_y = d['nf_label']

        # KIN + BETA
        co_kin = np.array(d['co_t2rt_all'])
        co_binary_beta = d['co_binary_beta_all']
        nf_kin = np.array(d['nf_t2rt'])
        nf_binary_beta = d['nf_binary_beta']

        lg_score = d['co_ind_log_R']

        co_0 = np.nonzero(co_test_y==0)[0]
        co_1 = np.nonzero(co_test_y==1)[0]
        co_premo = np.nonzero(co_kin<0)[0]

        nf_0 = np.nonzero(nf_test_y==0)[0]
        nf_1 = np.nonzero(nf_test_y==1)[0]
        nf_premo = np.nonzero(nf_kin<0)[0]

        mns = d['co_unit_mns']
        stds = d['co_unit_stds']

        good_ix = []
        wts = []
        bad_ix = []
        wts_bad = []
        mn_std = []

        for n in range(nunits):
            sc_co = lg_score[n][2]
            sc_nf = lg_score[n][3]
            u, p = scipy.stats.mannwhitneyu(sc_co[co_0], sc_co[co_1])
            u2, p2 = scipy.stats.mannwhitneyu(sc_nf[nf_0], sc_nf[nf_1])

            bd = True
            if np.logical_and(p < 0.05, p2 < 0.05):
                if np.logical_and(np.mean(sc_co[co_0]) < np.mean(sc_co[co_1]), 
                    np.mean(sc_nf[nf_0]) < np.mean(sc_nf[nf_1])):
                    co_premo_0 = np.nonzero(co_binary_beta[co_premo]==0)[0]
                    co_premo_1 = np.nonzero(co_binary_beta[co_premo]==1)[0]
                    nf_premo_0 = np.nonzero(nf_binary_beta[nf_premo]==0)[0]
                    nf_premo_1 = np.nonzero(nf_binary_beta[nf_premo]==1)[0]
                    
                    co_premo_0_sc = np.mean(sc_co[co_premo[co_premo_0]])
                    co_premo_1_sc = np.mean(sc_co[co_premo[co_premo_1]])
                    nf_premo_0_sc = np.mean(sc_nf[nf_premo[nf_premo_0]])
                    nf_premo_1_sc = np.mean(sc_nf[nf_premo[nf_premo_1]])

                    if np.logical_and(co_premo_0_sc > co_premo_1_sc, nf_premo_0_sc > nf_premo_1_sc):
                        good_ix.append(n)
                        wts.append([lg_score[n][0], lg_score[n][1]])
                        bd = False
                    else:
                        print co_premo_0_sc - co_premo_1_sc, nf_premo_0_sc - nf_premo_1_sc
            if bd:
                bad_ix.append(n)
                wts_bad.append([lg_score[n][0], lg_score[n][1]])

            mn_std.append([mns[n], stds[n]])

        mod_idx = {}
        mod_idx['chosen'] = np.zeros((len(good_ix), 2, 2)) # unit, task, min/max
        mod_idx['unchosen'] = np.zeros((len(bad_ix), 2, 2))  


        ky = d['lfp_lab'].keys()
        blks = np.vstack((ky))
        ib = np.argsort(blks[:, 0])

        lfp_labs = []
        for i, b in enumerate(ib):
            lfp_labs.append(d['lfp_lab'][ky[b]])
        lfp_labs = np.hstack((lfp_labs))

        co_ix = np.nonzero(lfp_labs < 80)[0]
        nf_ix = np.nonzero(lfp_labs > 80)[0]

        for i, ix in enumerate([co_ix, nf_ix]):
            X = np.mean(d['S_bin'][day][ix, :, :], 0)
                
            for j, key in enumerate(zip(['chosen', 'unchosen'], [good_ix, bad_ix])):

                    for ii, ic in enumerate(key[1]):
                        mod_idx[key[0]][ii, i, 0] = np.min(X[:, ii])
                        mod_idx[key[0]][ii, i, 1] = np.max(X[:, ii])
                        

        units[day, 'chosen'] = good_ix
        units[day, 'nonchosen'] = bad_ix
        units[day, 'nunits'] = nunits
        units[day, 'wts'] = wts
        units[day, 'wts_bad'] = wts_bad
        units[day, 'mean_std'] = mn_std
        units[day, 'mod_idx'] = mod_idx

    return units, days, dat

def analyze_unit_props(units, days, dat, nlags=3, animal='grom'):
    
    master_table = {}

    if animal == 'cart':
        canolty = pickle.load(open('all_cells_cheif_res_three_fourth_mc.pkl'))

    elif animal == 'grom':
        #canolty = pickle.load(open('grom_cheif_res.pkl', 'wb'))
        canolty = pickle.load(open('grom_cheif_res.pkl'))

    # Properties of chosen vs. unchosen units: 
    # Sign of neurons, intercepts of neurons
    for i_d, day in enumerate(days):

        table2 = {}
        d = dat[day][0]
        nunits = units[day, 'nunits']
        chosen = units[day, 'chosen'][units[day, 'chosen'] < nunits]
        unchosen =units[day, 'nonchosen'][units[day, 'nonchosen'] < nunits] 
        assert len(chosen)+len(unchosen) == nunits


        ###########################################
        ########## Sign of Weights ################
        ###########################################
        wts = units[day][chosen]
        wts2 = np.array(units[day, 'all'][unchosen])

        chosen_perc = [np.nonzero(wts > 0)[0], np.nonzero(wts < 0)[0]]
        nonchosen_perc = [np.nonzero(wts2 > 0)[0], np.nonzero(wts2 < 0)[0]]
        table2['wt_sign'] = [chosen_perc, nonchosen_perc]


        ###########################################
        #### Change in FR w/ existence of beta ####
        ###########################################
        beta_fr = {}
        beta_fr['chosen'] = np.zeros((len(chosen), 2, 2)) # unit, task, beta off/on
        beta_fr['unchosen'] = np.zeros((len(unchosen), 2, 2))

        spd = np.sum(d['co_kin']**2, 1)**.5
        spd_binary = spd > 3.5
        beta_binary = d['co_binary_beta_all']
        ix_co_0 = np.nonzero(np.logical_and(spd_binary == 0, beta_binary == 0))[0]
        ix_co_1 = np.nonzero(np.logical_and(spd_binary == 0, beta_binary == 1))[0]
        demean_X = d['co_spks_all']

        spd = np.sum(d['nf_kin']**2, 1)**.5
        spd_binary2 = spd > 3.5
        beta_binary2 = d['nf_binary_beta']
        ix_nf_0 = np.nonzero(np.logical_and(spd_binary2 == 0, beta_binary2 == 0))[0]
        ix_nf_1 = np.nonzero(np.logical_and(spd_binary2 == 0, beta_binary2 == 1))[0]
        demean_X2 = d['nf_spks']

        for ii, ic in enumerate(chosen):
            beta_fr['chosen'][ii, 0, 0] = np.mean(demean_X[ix_co_0, ic])
            beta_fr['chosen'][ii, 0, 1] = np.mean(demean_X[ix_co_1, ic])
            beta_fr['chosen'][ii, 1, 0] = np.mean(demean_X2[ix_nf_0, ic])
            beta_fr['chosen'][ii, 1, 1] = np.mean(demean_X2[ix_nf_1, ic])

        for ii, ic in enumerate(unchosen):
            beta_fr['unchosen'][ii, 0, 0] = np.mean(demean_X[ix_co_0, ic])
            beta_fr['unchosen'][ii, 0, 1] = np.mean(demean_X[ix_co_1, ic])
            beta_fr['unchosen'][ii, 1, 0] = np.mean(demean_X2[ix_nf_0, ic])
            beta_fr['unchosen'][ii, 1, 1] = np.mean(demean_X2[ix_nf_1, ic])


        ########################################################
        ### Modulation  during CO/NF task (avg max, avg min) ###
        ########################################################
        mod_idx = {}
        mod_idx['chosen'] = np.zeros((len(chosen), 2, 2)) # unit, task, min/max
        mod_idx['chosen', 'arg'] = np.zeros((len(chosen), 2, 2))
        mod_idx['unchosen'] = np.zeros((len(unchosen), 2, 2))  
        mod_idx['unchosen', 'arg'] = np.zeros((len(unchosen), 2, 2))

        ky = d['lfp_lab'].keys()
        blks = np.vstack((ky))
        ib = np.argsort(blks[:, 0])

        lfp_labs = []
        for i, b in enumerate(ib):
            lfp_labs.append(d['lfp_lab'][ky[b]])
        lfp_labs = np.hstack((lfp_labs))

        co_ix = np.nonzero(lfp_labs < 80)[0]
        nf_ix = np.nonzero(lfp_labs > 80)[0]

        for i, ix in enumerate([co_ix, nf_ix]):
            X = np.mean(d['S_bin'][day][ix, :, :], 0)
            
            for j, key in enumerate(zip(['chosen', 'unchosen'], [chosen, unchosen])):

                for ii, ic in enumerate(key[1]):
                    mod_idx[key[0]][ii, i, 0] = np.min(X[:, ii])
                    mod_idx[key[0]][ii, i, 1] = np.max(X[:, ii])
                    
                    mod_idx[key[0], 'arg'][ii, i, 0] = np.argmin(X[:, ii])
                    mod_idx[key[0], 'arg'][ii, i, 1] = np.argmax(X[:, ii])
                    
        ########################################
        ######## Canolty Tuning Params ########
        ########################################
        canoltyd = {} # unit, task, slope/int
        canoltyd['chosen'] = np.zeros((len(chosen), 2, 2))
        canoltyd['unchosen'] = np.zeros((len(unchosen), 2, 2))

        can = canolty[day]

        # All: 
        for i, ic in enumerate(chosen):
            canoltyd['chosen'][i, :, 0] = np.hstack((can[0, 'slp', 2, 'hold']['slope'][ic], can[1, 'slp', 2, 'hold']['slope'][ic] ))
            canoltyd['chosen'][i, :, 1] = np.hstack((can[0, 'slp', 2, 'hold']['intcp'][ic], can[1, 'slp', 2, 'hold']['intcp'][ic] ))

        for i, ic in enumerate(unchosen):
            canoltyd['unchosen'][i, :, 0] = np.hstack((can[0, 'slp', 2, 'hold']['slope'][ic], can[1, 'slp', 2, 'hold']['slope'][ic] ))
            canoltyd['unchosen'][i, :, 1] = np.hstack((can[0, 'slp', 2, 'hold']['intcp'][ic], can[1, 'slp', 2, 'hold']['intcp'][ic] ))
        
        master_table[day, 'wts'] = table2
        master_table[day, 'beta_fr'] = beta_fr
        master_table[day, 'mod_idx'] = mod_idx
        master_table[day, 'canoltyd'] = canoltyd
    pickle.dump(master_table, open(animal+'_master_metrics_important_units.pkl', 'wb'))

def plot_unit_analysis(animal='grom'):
    if animal == 'grom':
        master = pickle.load(open('grom_master_metrics_important_units.pkl'))
        keys = np.vstack((master.keys()))
        days = np.sort(np.unique(keys[:, 0]))

        for i_d, dy in enumerate(days):
            pos_master = master[dy, 'wts']['wt_sign'][0][0]
            neg_master = master[dy, 'wts']['wt_sign'][0][1]

            p = master[dy, 'wts']['wt_sign'][1][0]
            n = master[dy, 'wts']['wt_sign'][1][1]
            print dy, ' proportion pos: ', len(pos_master)/float(len(pos_master)+len(neg_master)), ', unchosen: ', len(p)/float(len(p)+len(n))

            ##################################
            #### CO and NF pos / neg plot ####
            ##################################
            #beta2 = [master[dy, 'beta_fr']['chosen'], master[dy, 'beta_fr']['unchosen']]
            master_ix = [[master[dy, 'beta_fr']['chosen'], pos_master, neg_master], [master[dy, 'beta_fr']['unchosen'], p, n]]
            
            fig, ax = plt.subplots(nrows = 2, ncols = 2) # task x pos / neg for chosen
            col = ['k', 'r']
            for ib,  (beta, pos, neg) in enumerate(master_ix):    
                ax[0, 0].plot(beta[pos, 0, 0], beta[pos, 0, 1], col[ib]+'.')
                ax[0, 1].plot(beta[pos, 1, 0], beta[pos, 1, 1], col[ib]+'.')
                
                ax[1, 0].plot(beta[neg, 0, 0], beta[neg, 0, 1], col[ib]+'.')
                ax[1, 1].plot(beta[neg, 1, 0], beta[neg, 1, 1], col[ib]+'.')
            
            tsk = ['CO ', 'NF ']
            sign = [' pos', ' neg']

            for i in range(2):
                for j in range(2):
                    axi = ax[i, j]
                    axi.set_xlim([-.6, .6])
                    axi.set_ylim([-.6, .6])
                    axi.plot([-.6, .6], [-.6, .6], 'k-')
                    axi.set_title(sign[i] + ' ' + tsk[j])
                    axi.set_xlabel('Beta Off mFR')
                    axi.set_ylabel('Beta On mFR')
            plt.tight_layout()

            ##################################
            #### CO and NF pos / neg plot ####
            ##################################
            can2 = [[master[dy, 'canoltyd']['chosen'], pos_master, neg_master], [master[dy, 'canoltyd']['unchosen'], p, n]] # unit, task, slope/int
            fig, ax = plt.subplots(ncols = 2) # task 
            col = ['k', 'r']
            for ib,  (beta, pos, neg) in enumerate(can2):    
                ax[0].plot(beta[pos, 0, 0], beta[pos, 1, 0], col[ib]+'.')
                ax[1].plot(beta[neg, 0, 0], beta[neg, 1, 0], col[ib]+'.')

            sign = [' pos', ' neg']

            for i in range(2):
                axi = ax[i]
                axi.plot([-.6, .6], [-.6, .6], 'k-')
                axi.set_xlim([-.01, .01])
                axi.set_ylim([-.01, .01])
                
                axi.set_title(sign[i])
                axi.set_xlabel('CO beta slope')
                axi.set_ylabel('NF beta slope')

            plt.tight_layout()

            ##################################
            #### Mod Index for Pos vs. Neg ###
            ##################################

            mi2 = [[master[dy, 'mod_idx']['chosen'], pos_master, neg_master], [master[dy, 'mod_idx']['unchosen'], p, n]] # unit, task, min/max
            fig, ax = plt.subplots(ncols = 2) # task 
            col = ['k', 'r']
            for ib,  (beta, pos, neg) in enumerate(mi2):    
                ax[0].plot(beta[pos, 0, 1]-beta[pos, 0, 0], beta[pos, 1, 1]-beta[pos, 1, 0], col[ib]+'.')
                ax[1].plot(beta[neg, 0, 1]-beta[neg, 0, 0], beta[neg, 1, 1]-beta[neg, 1, 0], col[ib]+'.')

            sign = [' pos', ' neg']

            for i in range(2):
                axi = ax[i]
                axi.plot([-.6, .6], [-.6, .6], 'k-')
                axi.set_xlim([-.01, .01])
                axi.set_ylim([-.01, .01])
                
                axi.set_title(sign[i])
                axi.set_xlabel('CO mod idx')
                axi.set_ylabel('NF mod idx')
            plt.tight_layout()

def plot_analyzed_ind_log_R_chosen_pos_vs_neg(units, dat, days, animal='grom'):
    
    if animal == 'grom':
        ac = pickle.load(open('grom_master_auto_correlations_pls_off.pkl'))
        canolty = pickle.load(open('grom_cheif_res.pkl'))
    elif animal == 'cart':
        ac = pickle.load(open('cart_master_auto_correlations_pls_off.pkl'))
        canolty = pickle.load(open('all_cells_cheif_res_three_fourth_mc.pkl'))

    # Weight Distribution:
    f, ax = plt.subplots()

    # Mean FR
    f2, ax2 = plt.subplots()

    # Mod Idx
    f3, ax3 = plt.subplots(ncols = 2)
    f4, ax4 = plt.subplots()
    # Canolty 
    f5, ax5 = plt.subplots(ncols=2)

    dt = .2
    beta_range = [20, 45]

    osc_ch = []
    osc_uch = []


    for i_d, day in enumerate(days):

        chosen = np.array(units[day, 'chosen'])
        wts = np.array([ np.mean(units[day, 'wts'][n][0]) for n in range(len(chosen)) ])
        #ax.boxplot(wts, positions=[i_d])
        #unchosen = np.array(units[day, 'nonchosen'])
        unchosen = chosen[np.nonzero(wts<0)[0]]
        chosen = chosen[np.nonzero(wts>0)[0]]

        uchi = np.nonzero(wts<0)[0]
        chi = np.nonzero(wts>0)[0]

        wts_bad = wts[np.nonzero(wts<0)[0]]
        wts = wts[np.nonzero(wts>0)[0]]
        
        if len(wts) > 0:
            ax.boxplot(wts, positions=[i_d])


        #wts_bad = np.array([ np.mean(units[day, 'wts_bad'][n][0]) for n in range(len(unchosen)) ])
        if len(wts_bad) > 0:
            ax.boxplot(wts_bad, positions=[i_d+dt])
    
        ax.set_title('Wts: Chosen & Unchosen')

        # Mean / std: 
        mn = np.vstack((units[day, 'mean_std']))[chosen][:, 0]
        mn2 = np.vstack((units[day, 'mean_std']))[unchosen][:, 0]
        ax2.boxplot(mn, positions=[i_d])
        if len(unchosen)>0:
            ax2.boxplot(mn2, positions=[i_d+dt])
        ax2.set_title('Mean: Chosen & Unchosen')

        # Mod Idx: 
        #n, i = np.histogram(units[day, 'mod_idx']['chosen'][:, 0, 1] - units[day, 'mod_idx']['chosen'][:, 0, 0])
        #n2, i2 = np.histogram(units[day, 'mod_idx']['unchosen'][:, 0, 1] - units[day, 'mod_idx']['unchosen'][:, 0, 0])
        
        #ax3[0].boxplot(units[day, 'mod_idx']['chosen'][:, 0,  1] - units[day, 'mod_idx']['chosen'][:, 0, 0], positions=[i_d])
        #ax3[0].boxplot(units[day, 'mod_idx']['unchosen'][:, 0, 1] - units[day, 'mod_idx']['unchosen'][:, 0, 0], positions=[i_d+dt])

        ax3[0].boxplot(units[day, 'mod_idx']['chosen'][chi, 0,  1] - units[day, 'mod_idx']['chosen'][chi, 0, 0], positions=[i_d])
        if len(unchosen) > 0:
            ax3[0].boxplot(units[day, 'mod_idx']['chosen'][uchi, 0, 1] - units[day, 'mod_idx']['chosen'][uchi, 0, 0], positions=[i_d+dt])
        ax3[0].set_title('Mod Idx CO : Chosen & Unch.')

        # ax3[1].boxplot(units[day, 'mod_idx']['chosen'][:, 1, 1] - units[day,'mod_idx']['chosen'][:, 1, 0], positions=[i_d])
        # ax3[1].boxplot(units[day, 'mod_idx']['unchosen'][:, 1, 1] - units[day,'mod_idx']['unchosen'][:, 1, 0], positions=[i_d+dt])
        ax3[1].boxplot(units[day, 'mod_idx']['chosen'][chi, 1, 1] - units[day,'mod_idx']['chosen'][chi, 1, 0], positions=[i_d])
        if len(unchosen) > 0:   
            ax3[1].boxplot(units[day, 'mod_idx']['chosen'][uchi, 1, 1] - units[day,'mod_idx']['unchosen'][uchi, 1, 0], positions=[i_d+dt])

        ax3[1].set_title('Mod Idx OB : Chosen & Unch.')

        ############################
        ##### AUTO CORR HIST #######
        ############################

        a = ac[day, 'cnts']
        a[a==0] = 1
        A = ac[day]/a

        a_off = ac[day, 'cnts_off']
        a_off[a_off==0] = 1
        A_off = ac[day, 'off']/a_off

        frange = np.nonzero(np.logical_and(f > beta_range[0], f < beta_range[1]))[0]

        A[:, 200] = 0
        A_off[:, 200] = 0

        A[chosen, :]/=mn[:, None]
        A[unchosen, :]/=mn2[:, None]

        f, AF = fft_simple(A)
        f, AF_off = fft_simple(A_off)
        
        A_chosen = AF[chosen, :] - AF_off[chosen, :]
        A_unchosen = AF[unchosen, :] - AF_off[unchosen, :]

        ach =np.log(A_chosen[:, 2:]/np.sum(A_chosen, axis=1)[:, None])
        auc =np.log(A_unchosen[:, 2:]/np.sum(A_unchosen, axis=1)[:, None])

        osc_ch.append(np.nanmean(ach[:, frange], axis=1))
        osc_uch.append(np.nanmean(auc[:, frange], axis=1))

        ax4.boxplot(np.mean(ach[:, frange], axis=1), positions=[i_d])
        try:
            ax4.boxplot(np.mean(auc[:, frange], axis=1), positions=[i_d+.2])
        except:
            print 'no unch'        
        ax4.set_xlim([0., 6.])
        # for axx in [ax4]:
        #     for a in [0, 1]:
        #         axx[a].set_xlim([3.,100.])

        ######################
        ##### Canolty Slps  ##
        ######################
        can = canolty[day]
        canolty_ch_co = []
        canolty_ch_nf = []

        canolty_unch_co = []
        canolty_unch_nf = []

        # All: 
        for i, ic in enumerate(chosen):
            canolty_ch_co.append(can[0, 'slp', 2, 'hold']['slope'][ic])
            canolty_ch_nf.append(can[1, 'slp', 2, 'hold']['slope'][ic])

        for i, ic in enumerate(unchosen):
            canolty_unch_co.append(can[0, 'slp', 2, 'hold']['slope'][ic])
            canolty_unch_nf.append(can[0, 'slp', 2, 'hold']['slope'][ic])

        ax5[0].boxplot(canolty_ch_co, positions=[i_d])
        if len(unchosen)>0:
            ax5[0].boxplot(canolty_unch_co, positions=[i_d+dt])
        ax5[0].set_title('CO Canolty Slopes')
        ax5[1].boxplot(canolty_ch_nf, positions=[i_d])
        if len(unchosen)>0:
            ax5[1].boxplot(canolty_unch_nf, positions=[i_d+dt])
        ax5[1].set_title('NF Canolty Slopes')


    for a in [ax, ax2]:
        a.set_xlim([-.5, 6])

    for axx in [ax3, ax5]:
        for a in [0, 1]:
            axx[a].set_xlim([-0.5, 6])

    UC = np.hstack((osc_uch))
    CH = np.hstack((osc_ch))
    print 'mann whitney on '
    print scipy.stats.mannwhitneyu(UC, CH)
    print 'mean osc ch: ', np.mean(CH)
    print 'mean osc unch: ', np.mean(UC)

def plot_analyzed_ind_log_R_chosen_vs_unchosen(units, dat, days, animal='grom'):
    
    if animal == 'grom':
        ac = pickle.load(open('grom_master_auto_correlations_pls_off.pkl'))
        canolty = pickle.load(open('grom_cheif_res.pkl'))

    elif animal == 'cart':
        ac = pickle.load(open('cart_master_auto_correlations_pls_off.pkl'))
        canolty = pickle.load(open('all_cells_cheif_res_three_fourth_mc.pkl'))

    # Weight Distribution:
    f, ax = plt.subplots()

    # Mean FR
    f2, ax2 = plt.subplots()

    # Mod Idx
    f3, ax3 = plt.subplots(ncols = 2)
    f4, ax4 = plt.subplots()
    # Canolty 
    f5, ax5 = plt.subplots(ncols=2)

    dt = .2
    beta_range = [20, 45]

    osc_ch = []
    osc_uch = []

    for i_d, day in enumerate(days):

        chosen = np.array(units[day, 'chosen'])
        wts = np.array([ np.mean(units[day, 'wts'][n][0]) for n in range(len(chosen)) ])
        ax.boxplot(wts, positions=[i_d])
        unchosen = np.array(units[day, 'nonchosen'])
        
        wts_bad =np.array([ np.mean(units[day, 'wts_bad'][n][0]) for n in range(len(chosen)) ])
        ax.boxplot(wts_bad, positions=[i_d+dt])
        ax.set_title('Wts: Chosen & Unchosen')

        # Mean / std: 
        mn = np.vstack((units[day, 'mean_std']))[chosen][:, 0]
        mn2 = np.vstack((units[day, 'mean_std']))[unchosen][:, 0]
        ax2.boxplot(mn, positions=[i_d])
        if len(unchosen)>0:
            ax2.boxplot(mn2, positions=[i_d+dt])
        ax2.set_title('Mean: Chosen & Unchosen')

        # Mod Idx:        
        ax3[0].boxplot(units[day, 'mod_idx']['chosen'][:, 0,  1] - units[day, 'mod_idx']['chosen'][:, 0, 0], positions=[i_d])
        ax3[0].boxplot(units[day, 'mod_idx']['unchosen'][:, 0, 1] - units[day, 'mod_idx']['unchosen'][:, 0, 0], positions=[i_d+dt])
        ax3[0].set_title('Mod Idx CO : Chosen & Unch.')

        ax3[1].boxplot(units[day, 'mod_idx']['chosen'][:, 1, 1] - units[day,'mod_idx']['chosen'][:, 1, 0], positions=[i_d])
        ax3[1].boxplot(units[day, 'mod_idx']['unchosen'][:, 1, 1] - units[day,'mod_idx']['unchosen'][:, 1, 0], positions=[i_d+dt])
        ax3[1].set_title('Mod Idx NF : Chosen & Unch.')

        ############################
        ##### AUTO CORR HIST #######
        ############################

        a = ac[day, 'cnts']
        a[a==0] = 1
        A = ac[day]/a

        a_off = ac[day, 'cnts_off']
        a_off[a_off==0] = 1
        A_off = ac[day, 'off']/a_off

        frange = np.nonzero(np.logical_and(f > beta_range[0], f < beta_range[1]))[0]

        A[:, 200] = 0
        A_off[:, 200] = 0

        A[chosen, :]/=mn[:, None]
        A[unchosen, :]/=mn2[:, None]

        f, AF = fft_simple(A)
        f, AF_off = fft_simple(A_off)
        
        A_chosen = AF[chosen, :] - AF_off[chosen, :]
        A_unchosen = AF[unchosen, :] - AF_off[unchosen, :]

        ach =np.log(A_chosen[:, 2:]/np.sum(A_chosen, axis=1)[:, None])
        auc =np.log(A_unchosen[:, 2:]/np.sum(A_unchosen, axis=1)[:, None])

        osc_ch.append(np.nanmean(ach[:, frange], axis=1))
        osc_uch.append(np.nanmean(auc[:, frange], axis=1))

        ax4.boxplot(np.mean(ach[:, frange], axis=1), positions=[i_d])
        try:
            ax4.boxplot(np.mean(auc[:, frange], axis=1), positions=[i_d+.2])
        except:
            print 'no unch'        
        ax4.set_xlim([0., 6.])
        # for axx in [ax4]:
        #     for a in [0, 1]:
        #         axx[a].set_xlim([3.,100.])

        ######################
        ##### Canolty Slps  ##
        ######################
        can = canolty[day]
        canolty_ch_co = []
        canolty_ch_nf = []

        canolty_unch_co = []
        canolty_unch_nf = []

        # All: 
        for i, ic in enumerate(chosen):
            canolty_ch_co.append(can[0, 'slp', 2, 'hold']['slope'][ic])
            canolty_ch_nf.append(can[1, 'slp', 2, 'hold']['slope'][ic])

        for i, ic in enumerate(unchosen):
            canolty_unch_co.append(can[0, 'slp', 2, 'hold']['slope'][ic])
            canolty_unch_nf.append(can[0, 'slp', 2, 'hold']['slope'][ic])

        ax5[0].boxplot(canolty_ch_co, positions=[i_d])
        if len(unchosen)>0:
            ax5[0].boxplot(canolty_unch_co, positions=[i_d+dt])
        ax5[0].set_title('CO Canolty Slopes')
        ax5[1].boxplot(canolty_ch_nf, positions=[i_d])
        if len(unchosen)>0:
            ax5[1].boxplot(canolty_unch_nf, positions=[i_d+dt])
        ax5[1].set_title('NF Canolty Slopes')


    for a in [ax, ax2]:
        a.set_xlim([-.5, 6])

    for axx in [ax3, ax5]:
        for a in [0, 1]:
            axx[a].set_xlim([-0.5, 6])

    UC = np.hstack((osc_uch))
    CH = np.hstack((osc_ch))
    print 'mann whitney on '
    print scipy.stats.mannwhitneyu(UC, CH)

def big_plt_analyzed_chosen_pos_neg_vs_unchosen(beta_range = [20, 45]):
    d = pickle.load(open('cart_25_lag_3_master_logistic_regression_zsc_traintest_dict.pkl'))
    d2 = pickle.load(open('grom_25_lag_3_master_logistic_regression_zsc_traintest_dict.pkl'))

    D = [d2, d]
    subj = ['grom', 'cart']

    f, ax = plt.subplots(nrows=2, ncols=5)
    master_mets = {}
    for s in ['grom','cart']:
        master_mets[s] = {}
        for i in ['wt', 'slp_co','slp_nf', 'mod_co', 'mod_nf', 'mn', 'ac']:
            for j in range(2):
                try:
                    master_mets[s][j, i]['chosen'] = []
                    master_mets[s][j, i]['unchosen'] = []
                except:
                    master_mets[s][j, i] = {}
                    master_mets[s][j, i]['chosen'] = []
                    master_mets[s][j, i]['unchosen'] = []

    for i, (dat, animal) in enumerate(zip(D, subj)):
        units, days, dat = analyze_ind_log_R(dat)

        if animal == 'grom':
            ac = pickle.load(open('grom_master_auto_correlations_pls_off.pkl'))
            canolty = pickle.load(open('grom_cheif_res.pkl'))
        
        elif animal == 'cart':
            ac = pickle.load(open('cart_master_auto_correlations_pls_off.pkl'))
            canolty = pickle.load(open('all_cells_cheif_res_three_fourth_mc.pkl'))

        for i_d, day in enumerate(days):

            chosen0 = np.array(units[day, 'chosen'])
            unchosen0 = np.array(units[day, 'nonchosen'])            

            wts_0 = np.array([ np.mean(units[day, 'wts'][n][0]) for n in range(len(chosen0)) ])
            unchosen_neg = chosen0[np.nonzero(wts_0<0)[0]]
            chosen_pos = chosen0[np.nonzero(wts_0>0)[0]]

            wts_master = np.array([ np.mean(units[day, 'wts'][n][0]) for n in range(len(chosen0)) ])

            for c, (chosen, unchosen) in enumerate(zip([chosen0, chosen_pos], [unchosen0, unchosen_neg])):
                
                if c == 0:   
                    wts = wts_master.copy() 
                    wts_bad =np.array([ np.mean(units[day, 'wts_bad'][n][0]) for n in range(len(unchosen)) ])
                
                elif c == 1:
                    wts_bad = wts_master[wts_master<0]
                    chi = np.nonzero(wts_master >0)[0]
                    uchi= np.nonzero(wts_master <0)[0]
                    assert len(chi) == len(chosen_pos)
                    assert len(uchi) == len(unchosen_neg)
                    
                    wts = wts_master[wts_master > 0]

                master_mets[animal][c, 'wt']['chosen'].append(wts)
                master_mets[animal][c, 'wt']['unchosen'].append(wts_bad)
            
                # Mean / std: 
                mn = np.vstack((units[day, 'mean_std']))[chosen][:, 0]
                mn2 = np.vstack((units[day, 'mean_std']))[unchosen][:, 0]
                master_mets[animal][c, 'mn']['chosen'].append(mn)
                master_mets[animal][c, 'mn']['unchosen'].append(mn2)            

                # Mod Idx: Max - Min      
                comodix = units[day, 'mod_idx']['chosen'][:, 0,  1] - units[day, 'mod_idx']['chosen'][:, 0, 0]
                nfmodix = units[day, 'mod_idx']['chosen'][:, 1, 1] - units[day,'mod_idx']['chosen'][:, 1, 0]

                if c == 0:
                    master_mets[animal][c, 'mod_co']['chosen'].append(comodix)
                    comodix_unch = units[day, 'mod_idx']['unchosen'][:, 0, 1] - units[day, 'mod_idx']['unchosen'][:, 0, 0]
                    master_mets[animal][c, 'mod_co']['unchosen'].append(comodix_unch)
                    
                    master_mets[animal][c, 'mod_nf']['chosen'].append(nfmodix)    
                    nfmodix_unch = units[day, 'mod_idx']['unchosen'][:, 1, 1] - units[day,'mod_idx']['unchosen'][:, 1, 0]
                    master_mets[animal][c, 'mod_nf']['unchosen'].append(nfmodix_unch)
                
                elif c == 1:
                    master_mets[animal][c, 'mod_co']['chosen'].append(comodix[chi])
                    master_mets[animal][c, 'mod_co']['unchosen'].append(comodix[uchi])
                    master_mets[animal][c, 'mod_nf']['chosen'].append(nfmodix[chi])    
                    master_mets[animal][c, 'mod_nf']['unchosen'].append(nfmodix[uchi])

                ############################
                ##### AUTO CORR HIST #######
                ############################

                a = ac[day, 'cnts']
                a[a==0] = 1
                A = ac[day]/a

                a_off = ac[day, 'cnts_off']
                a_off[a_off==0] = 1
                A_off = ac[day, 'off']/a_off

                frange = np.nonzero(np.logical_and(f > beta_range[0], f < beta_range[1]))[0]

                A[:, 200] = 0
                A_off[:, 200] = 0

                A[chosen, :]/=mn[:, None]
                A[unchosen, :]/=mn2[:, None]

                f, AF = fft_simple(A)
                f, AF_off = fft_simple(A_off)
                
                A_chosen = AF[chosen, :] - AF_off[chosen, :]
                A_unchosen = AF[unchosen, :] - AF_off[unchosen, :]

                ach =np.log(A_chosen[:, 2:]/np.sum(A_chosen, axis=1)[:, None])
                auc =np.log(A_unchosen[:, 2:]/np.sum(A_unchosen, axis=1)[:, None])

                osc_ch =np.nanmean(ach[:, frange], axis=1)
                osc_uch = np.nanmean(auc[:, frange], axis=1)

                master_mets[animal][c, 'ac']['chosen'].append(osc_ch)
                master_mets[animal][c, 'ac']['unchosen'].append(osc_uch)

                ######################
                ##### Canolty Slps  ##
                ######################
                can = canolty[day]
                canolty_ch_co = []
                canolty_ch_nf = []

                canolty_unch_co = []
                canolty_unch_nf = []

                # All: 
                for i, ic in enumerate(chosen):
                    canolty_ch_co.append(can[0, 'slp', 2, 'hold']['slope'][ic])
                    canolty_ch_nf.append(can[1, 'slp', 2, 'hold']['slope'][ic])

                for i, ic in enumerate(unchosen):
                    canolty_unch_co.append(can[0, 'slp', 2, 'hold']['slope'][ic])
                    canolty_unch_nf.append(can[0, 'slp', 2, 'hold']['slope'][ic])

                master_mets[animal][c, 'slp_co']['chosen'].append(canolty_ch_co)
                master_mets[animal][c, 'slp_co']['unchosen'].append(canolty_unch_co)
                master_mets[animal][c, 'slp_nf']['chosen'].append(canolty_ch_nf)
                master_mets[animal][c, 'slp_nf']['unchosen'].append(canolty_unch_nf)
    return master_mets

def plot_big_plt(master_mets):
    f, ax = plt.subplots(ncols= 5, nrows=2, figsize=(20, 5))
    Ylims = [[-0.1, .2], [-.0025, 0.001], [0, 20], [0, 20], [-5.5, -5]]
    Ylabs = ['Classifier Wt.', 'Beta-to-FR Slope', 'CO Mod. Idx. (Hz)', 'Mean FR (Hz)', 'Beta Rhythmicity']
    N = [2, 3, 2, 1, 2]
    for i_s, (offs, subj) in enumerate(zip([0, .5],['grom', 'cart'])):
        mm = master_mets[subj]
    
        for ch in range(2):

            for im, met in enumerate(['wt', 'slp', 'mod', 'mn', 'ac']):

                axi = ax[ch, im]
                
                if met in ['mn', 'mod']:
                    fact = 40*40
                else:
                    fact = 1

                if met in ['slp', 'mod']:
                    x0 = fact*np.hstack((mm[ch, met+'_co']['chosen']))
                    axi.bar(offs, np.mean(x0), .2, color='k')
                    
                    x = fact*np.hstack((mm[ch, met+'_co']['unchosen']))
                    axi.bar(offs+.2, np.mean(x), .2,color='grey')
                    #axi.bar(i_s+.4, np.mean(np.hstack((mm[ch, met+'_nf']['chosen']))), .2, color='k', edgecolor='white', hatch="o")
                    #axi.bar(i_s+.6, np.mean(np.hstack((mm[ch, met+'_nf']['unchosen']))), .2, color='grey',edgecolor='white', hatch="o")
                    
                    axi.errorbar(offs+.1, np.mean(x0), 
                        yerr= np.std(x0)/np.sqrt(len(x0)),color='k')
                    
                    axi.errorbar(offs+.3, np.mean(x), 
                        yerr= np.std(x)/np.sqrt(len(x)),color='grey')
                    
                    # x = np.hstack((mm[ch, met+'_nf']['chosen']))
                    # axi.errorbar(i_s+.5, np.mean(x),
                    #     yerr= np.std(x)/np.sqrt(len(x)), color='k')
                    
                    # x = np.hstack((mm[ch, met+'_nf']['unchosen']))
                    # axi.errorbar(i_s+.7, np.mean(x),
                    #     yerr= np.std(x)/np.sqrt(len(x)), color='grey')
                    
                else:
                    x = fact*np.hstack((mm[ch, met]['chosen']))
                    axi.bar(offs, np.nanmean(x), .2, color='k')
                    axi.errorbar(offs+.1, np.nanmean(x),
                        yerr=np.nanstd(x)/np.sqrt(len(x)), color='k')
                    x0 = x.copy()

                    x = fact*np.hstack((mm[ch, met]['unchosen']))
                    axi.bar(offs+.2, np.nanmean(x), .2, color='grey')
                    axi.errorbar(offs+.3, np.nanmean(x),
                        yerr = np.nanstd(x)/np.sqrt(len(x)), color='grey')

                u, p = scipy.stats.kruskal(x, x0)
                print subj, ', 0) ch-unch, 1) chpos-chneg: ', ch, ', metric: ', met, ', KW: ', u, p, len(x0), len(x)

                axi.set_ylabel(Ylabs[im])
                axi.set_ylim(Ylims[im])
                axi.set_yticks(np.linspace(Ylims[im][0], Ylims[im][1], 4).round(N[im]))
                axi.set_xticks([])
                axi.set_xlim([-.1, 1])
    plt.tight_layout()
    #plt.savefig('fig8_maybe.eps', format='eps', dpi=300)
    plt.show()


def fft_simple(X):
    # Input X is units x timepoints
    import nitime.algorithms as tsa

    f, psd_mt, nu = tsa.multi_taper_psd(
        X, adaptive=False, jackknife=False, Fs=1000
        )

    return f, psd_mt

        # n1, i1 = np.histogram(mn)
        # ax[1].plot((i1[1:]+(.5*(i1[1]-i1[0])))*40., n1/float(np.sum(n1)))

        # n1, i1 = np.histogram(mn2)
        # ax[1].plot((i1[1:]+(.5*(i1[1]-i1[0])))*40., n1/float(np.sum(n1)))

        # n1, i1 = np.histogram(st)
        # ax[2].plot(i1[1:]+(.5*(i1[1]-i1[0])), n1/float(np.sum(n1)))

        # n1, i1 = np.histogram(st2)
        # ax[2].plot(i1[1:]+(.5*(i1[1]-i1[0])), n1/float(np.sum(n1)))
        
        ### Story thus far ###
        # 1. Postively weighted units are ones driving trend of further from movement during high beta
        # 2. They have higher FR (+1 cart)
        # 3. Task modulation index doesnt matter (+1 cart)
        # 4. Some sort of autocorrelation (doesn't work)
        # 5. FR to Beta slope is negative (increase FR w/ low beta) (+1 cart)
        # 6. Negative weighted lower FR units are slightly more rhythmic
