import predict_kin_w_spk
import sklearn.linear_model
import state_space_spks as sss
import state_space_cart as ssc
import numpy as np
import matplotlib.pyplot as plt
import state_space_w_beta_bursts as ssbb
import pickle
import scipy.stats

cmap1 = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]
cmap2  = ['lightseagreen','dodgerblue','gold','orangered','k']

def pre_process(day, test, all_cells, animal, bin, lags):
    if test:
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=None)
    else:
        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, all_cells=all_cells, animal=animal, days=[day])

    binsize = bin
    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt_sig = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize, smooth = 50,  animal=animal)
    #New -- binary kin signal

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

    #Use points before reach that are less than 3.5 cm/sec and after reach that are greater than 3.5 cm/sec
    for it, trl in enumerate(LAB_master[day]):
        rt_bin = RT_bin_master[day][it]
        spd = np.sqrt(np.sum(KC_bin_master[day][it, [2, 3], :]**2, 0))
        
        for tm in range(lags, int(2500/binsize)):
            ad = 0
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
                    
    X = np.vstack((X))
    Y = np.hstack((Y))      
    X2 = np.vstack((X2))
    Y2 = np.hstack((Y2))        
    trl2 = np.hstack((trl2))

    tm_trl_key = np.vstack((tm_trl_key))
    tm_trl_key2 = np.vstack((tm_trl_key2))

    binary_beta = np.hstack((binary_beta))
    binary_beta2 = np.hstack((binary_beta2))
    

    #Incorporate lags: 

    # Train Logistic Regression: 
    lg = sklearn.linear_model.LogisticRegression(fit_intercept=True)
    lg.fit(X, Y)
    y0 = lg.predict(X)

    # Predict
    ft = -1*np.squeeze(np.array((np.mat(lg.coef_)*X.T + lg.intercept_)))
    sc = 1/(1+np.exp(ft))

    distance_from_hyperplane = sc - 0.5
    plt.plot(distance_from_hyperplane, dist_from_onset, '.')
    plt.xlabel('Distance from Hyperplane')
    plt.ylabel('Dist from Move Onset')

    co_chance = []
    for i in range(10):
        ix = np.random.permutation(Y.shape[0])
        co_chance.append(lg.score(X, Y[ix]))

    plt.title(day+', perc corr: '+str(lg.score(X, Y))+ '\n'+', perc_chance: '+str(np.mean(co_chance)))


    # Test on NF: 
    y1 = lg.predict(X2)

    nf_chance = []
    for i in range(10):
        ix = np.random.permutation(Y2.shape[0])
        nf_chance.append(lg.score(X2, Y2[ix]))

    ft2 = -1*np.squeeze(np.array((np.mat(lg.coef_)*X2.T + lg.intercept_)))
    sc2 = 1/(1+np.exp(ft2))
    distance_from_hyperplane = sc2 - 0.5
    dist_from_onset2 = np.array(dist_from_onset2)

    cmap =['g','b','y','r']

    for it, t in enumerate(range(84, 88)):
        ix = np.nonzero(trl2 == t)[0]
        plt.plot(distance_from_hyperplane[ix], dist_from_onset2[ix], cmap[it]+'.')

    save_dict = dict()
    save_dict['co_spks'] = X
    save_dict['co_label'] = Y
    save_dict['co_t2rt'] = dist_from_onset
    save_dict['co_log_R'] = lg
    save_dict['score_co'] = sc - 0.5
    save_dict['co_y0'] = y0
    save_dict['co_tm_trl_key'] = tm_trl_key
    save_dict['co_binary_beta'] = binary_beta
    save_dict['co_chance_perf'] = co_chance

    save_dict['nf_spks'] = X2
    save_dict['nf_label'] = Y2
    save_dict['nf_t2rt'] = dist_from_onset2
    save_dict['score_nf'] = sc2 - 0.5
    save_dict['nf_y0'] = y1
    save_dict['nf_targ_num'] = trl2
    save_dict['nf_tm_trl_key'] = tm_trl_key2
    save_dict['nf_binary_beta'] = binary_beta2
    save_dict['nf_chance_perf'] = nf_chance

    return save_dict, day

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
        pickle.dump(d_dict, open('/home/lab/code/beta_single_unit/c_data/'+animal+'_'+day+'_'+str(bin)+'_lag_'+str(lags)+'_logistic_regression_dict.pkl', 'wb'))
    pickle.dump(master_dict, open('/home/lab/code/beta_single_unit/c_data/'+animal+'_'+str(bin)+'_lag_'+str(lags)+'_master_logistic_regression_dict.pkl', 'wb'))

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

def plot_perc_all(dat, offs=0):
    f, ax = plt.subplots()

    co_0 = []
    co_1 = []

    nf_0 = []
    nf_1 = []

    days = dat.keys()
    for i_d, dy in enumerate(np.sort(days)):
        d = dat[dy][0]
        co_ = d['co_label']

        ix0 = np.nonzero(co_==0)[0]
        ix1 = np.nonzero(co_ ==1)[0]

        co_0.append(d['score_co'][ix0])
        co_1.append(d['score_co'][ix1])

        nf_ = d['nf_label']
        ix0 = np.nonzero(nf_==0)[0]
        ix1 = np.nonzero(nf_==1)[0]

        nf_0.append(d['score_nf'][ix0])
        nf_1.append(d['score_nf'][ix1])


        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        fpr, tpr, _ = sklearn.metrics.roc_curve(d['co_label'], d['score_co']) 
        roc_auc =  sklearn.metrics.auc(fpr, tpr)
        #print 'co auc: ', roc_auc

        fpr, tpr, _ = sklearn.metrics.roc_curve(d['nf_label'], d['score_nf']) 
        roc_auc =  sklearn.metrics.auc(fpr, tpr)
        #print 'nf auc: ', roc_auc

    co00 = np.array([np.mean(co) for co in co_0])
    co11 = np.array([np.mean(co) for co in co_1])
    nf00 = np.array([np.mean(co) for co in nf_0])
    nf11 = np.array([np.mean(co) for co in nf_1])

    print 'CO ttest:', scipy.stats.ttest_rel(co00, co11) , 'n = ', len(co00), len(co11)
    print 'NF ttest:', scipy.stats.ttest_rel(nf00, nf11) , 'n= ', len(nf00), len(nf11)

    co0 = np.hstack((co_0))
    co1 = np.hstack((co_1))
    nf0 = np.hstack((nf_0))
    nf1 = np.hstack((nf_1))



    ax.bar(0, np.mean(co0)-offs, color='blue', width = .6)
    ax.bar(1, np.mean(co1)-offs, color='blue', width = .6)
    ax.bar(2, np.mean(nf0)-offs, color='red', width = .6)
    ax.bar(3, np.mean(nf1)-offs, color='red', width = .6)
    
    ax.errorbar(0+.3, np.mean(co0)-offs, yerr=np.std(co0)/np.sqrt(len(co0)), color='k')
    ax.errorbar(1+.3, np.mean(co1)-offs, yerr=np.std(co1)/np.sqrt(len(co1)), color='k')
    ax.errorbar(2+.3, np.mean(nf0)-offs, yerr=np.std(nf0)/np.sqrt(len(nf0)), color='k')
    ax.errorbar(3+.3, np.mean(nf1)-offs,  yerr=np.std(nf1)/np.sqrt(len(nf1)), color='k')

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

def line_plots_all(dat, offs=0.0):
    days = dat.keys()
    f, axi = plt.subplots()
    for it, tsk in enumerate(['co', 'nf']):
        tskd = dict()
        tskd[0] = []
        tskd[1] = []

        for i_d, dy in enumerate(days):
            d = dat[dy][0]

            tsk_trls = np.max(d[tsk+'_tm_trl_key'][:, 1])
            tsk_t2rt = -1*np.array(d[tsk+'_t2rt'])
            tsk_score = np.array(d['score_'+tsk])

            beta_tg = dict()
            beta_tg[0] = []
            beta_tg[1] = []
            beta_cnt = np.zeros((2, ))
            label = d[tsk+'_binary_beta']

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

            tskd[0].append(np.mean(B0))
            tskd[1].append(np.mean(B1))

        print tsk+ ' ttest:', scipy.stats.ttest_rel(tskd[0], tskd[1]) , 'n = ', len(tskd[0]), len(tskd[1])


    axi.set_xlabel('CO, NF')
    axi.set_ylabel('Distance from Thresh')
            
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









