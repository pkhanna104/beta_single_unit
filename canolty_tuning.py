#Re-extract kin signals: 
# Generate a) MC vs. NF b) MC vs. NF pref phase c) e.g. MC amp and NF amp d) e.g. MC phz and NF phz plots


import tables
import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import state_space_spks as sss
import state_space_w_beta_bursts as ssbb
import load_files
import pickle

from riglib.bmi import train
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi import ppfdecoder

from sklearn.lda import LDA

import gc
import multiprocessing as mp
import datetime
import predict_kin_w_spk
import math
from collections import namedtuple
import pickle

import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.2,style='white')

import multiprocessing as mp

def get_all(test, animal, all_cells):
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test, animal=animal, all_cells=all_cells)

    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, binsize=1, animal=animal)

    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, 1, smooth = 50, animal=animal)

    return keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator, B_bin, BC_bin, S_bin, Params, kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt

def get_all_day(day):
    # test = false, all_cells = False, 3/4 = true
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False, animal='cart', all_cells=False, days=[day])

    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, binsize=1, animal=animal)

    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, 1, smooth = 50, animal=animal)

    args= (keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator, B_bin, BC_bin, S_bin, Params, kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, rt)
    cr = run_canolty(*args)
    return cr[day]

def main(test=True, animal='grom'):
    args = get_all(test, animal)
    cr = run_canolty(*args)
    return cr

def main_mp():

    import pickle
    pool = mp.Pool()
    days = ['011315', '011415', '011515', '011615']
    results = pool.map(get_all_day, days)
    cr = {}
    for i_d, day in enumerate(days):
        pickle.dump(results[i_d], open('day'+'_sub_canolty_select_cells.pkl'))
        cr[day] = results[i_d]

    plot_chief_results(cr)

def run_canolty(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, 
    bef, aft, go_times_dict, mc_indicator, B_bin, BC_bin, S_bin, Params, kin_signal_dict, 
    binned_kin_signal_dict, bin_kin_signal, rt):

    n_b = 10
    color_dict = dict(b0='r', b1='g',b2='k')
    spk_sig = dict()

    chief_res = {}

    for i_d, d in enumerate(days):
        master_res = beta_met_to_mapping(d, blocks[i_d], mc_indicator[i_d], lfp_lab, BC_bin, S_bin, Params, bin_kin_signal, n_b=n_b)
        chief_res[d] = master_res
    return chief_res

def plot_chief_results(cheif_res):
    # define the colormap
    #cmap = ['maroon', 'firebrick', 'orangered', 'darksalmon', 'powderblue', 'lightslategrey']
    cmap = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]

    f, ax = plt.subplots(figsize=(15, 10),nrows=2, ncols=3)

    for i_d, d in enumerate(np.sort(cheif_res.keys())):
        cr = cheif_res[d]
        
        for im, (metric, metric2) in enumerate(zip(['slp', 'pref_phz'], ['slope', 'pref_phz'])):
            for ie, epoch in enumerate(['hold', 'reach']):
                for it in [0, 1]:
                    if im == 0:
                        axi = ax[ie, it]
                        slp, intc, r_v2, p_v, ste = scipy.stats.linregress(cr[(it, metric, 0, epoch)][metric2], cr[(it, metric, 1, epoch)][metric2])
                        try:
                            axi.plot(cr[(it, metric, 0, epoch)][metric2], cr[(it, metric, 1, epoch)][metric2], '.', markerfacecolor=tuple(np.array(cmap[i_d])/255.), label='$R^2=$'+str(int(1000*(r_v2**2))/1000.))
                        except:
                            print 'skippping ', d
                
                if im == 0:
                    slp, intc, r_v1, p_v, ste = scipy.stats.linregress(cr[(0, metric, 2, epoch)][metric2], cr[(1, metric, 2, epoch)][metric2])
                    try:
                        ax[ie, 2].plot(cr[(0, metric, 2, epoch)][metric2], cr[(1, metric, 2, epoch)][metric2], '.', markerfacecolor=tuple(np.array(cmap[i_d])/255.), label='$R^2=$'+str(int(1000*(r_v1**2))/1000.))
                    except:
                        print 'skipping', d        
    for ie in [0, 1]:
        ax[ie, 0].set_xlabel('Centerout Slice 1 Slope')
        ax[ie, 0].set_ylabel('Centerout Slice 2 Slope')

        ax[ie, 1].set_xlabel('Neurofeedback Slice 1 Slope')
        ax[ie, 1].set_ylabel('Neurofeedback Slice 2 Slope')

        ax[ie, 2].set_xlabel('Centerout Slope')
        ax[ie, 2].set_ylabel('Neurofeedback Slope')

        for i in range(3):
            ax[ie, i].plot([-1, 1], [0, 0], '-', color='gray', linewidth=0.4)
            ax[ie, i].plot([0, 0], [-1, 1], '-', color='gray', linewidth=0.4)
            ax[ie, i].legend(loc=0,ncol=2,fontsize=14)
            ax[ie, i].set_xlim([-0.005, .005])
            ax[ie, i].set_ylim([-0.005, .005])
            ax[ie, i].set_xticks([-0.001, 0.001])
            ax[ie, i].set_yticks([-0.001, 0.001])
            ax[ie, i].set_xticklabels(['-0.001', '0.001'])
            ax[ie, i].set_yticklabels(['-0.001', '0.001'])

        
    plt.tight_layout()
    plt.savefig('canolty_mc_vs_nf_tuning_for_hold_row1_and_reach_row2_three_fourth_mc.eps', format='eps', dpi=300)
    pickle.dump(cheif_res, open('select_cells_cheif_res_three_fourth_mc.pkl', 'wb'))

def old_stuff():
    # Plot Amp: 


    #Plot Phz: 



    #     #Log Beta Amp: 
    #     amp = np.log10(BC_bin[d])
    #     phz = BC_bin[d, 'phz']
    #     spk = S_bin[d]


    

    #     ntrls, nbins, nunits = spk.shape

    #     day_trial_type = []
    #     day_lfp_lab = []
    #     for ib, b in enumerate(blocks[i_d]):
    #         n_trials = len(lfp_lab[b, d])

    #         if int(mc_indicator[i_d][ib]) == 1:
    #             tt = ['mc']*n_trials
    #         elif int(mc_indicator[i_d][ib]) == 0:
    #             tt = ['beta']*n_trials
    #         else:
    #             raise

    #         day_trial_type.append(tt)
    #         day_lfp_lab.append(lfp_lab[b, d])

    #     day_trial_type = np.hstack(day_trial_type)
    #     day_lfp_lab = np.hstack((day_lfp_lab))

    #     train_ix = np.nonzero(day_trial_type == 'mc')[0]
    #     test_ix = np.nonzero(day_trial_type == 'beta')[0]
    #     #test_ix = np.nonzero(np.logical_and(day_trial_type == 'beta', Params[d]['day_lfp_labs']!=87))[0]

    #     mc_ix0 = np.arange(25)
    #     beta_ix1 = np.arange(15, 25)
    #     beta_ix2 = np.arange(15)
    #     beta_ix3 = mc_ix0.copy()

    #     #b_ix=[beta_ix1, beta_ix2, beta_ix3]
    #     b_ix = [beta_ix3]
    #     for bi, beta in enumerate(b_ix):
    #         tmp_ix_ = np.ix_(train_ix, mc_ix0, np.arange(nunits))
    #         MC = spk[tmp_ix_]

    #         tmp_ix_2 = np.ix_(train_ix, mc_ix0)
    #         AMC = amp[tmp_ix_2]
    #         PMC = phz[tmp_ix_2]

    #         tmp_ix_ = np.ix_(test_ix, beta, np.arange(nunits))
    #         B = spk[tmp_ix_]

    #         tmp_ix_2 = np.ix_(test_ix, beta)
    #         AB = amp[tmp_ix_2]
    #         PB = phz[tmp_ix_2]

    #         #Now, reshape, then sort!
    #         mc = MC.reshape(len(train_ix)*len(mc_ix0), nunits)
    #         amc = AMC.reshape(len(train_ix)*len(mc_ix0))
    #         pmc = PMC.reshape(len(train_ix)*len(mc_ix0))

    #         b = B.reshape(len(test_ix)*len(beta), nunits)
    #         ab = AB.reshape(len(test_ix)*len(beta))
    #         pb = PB.reshape(len(test_ix)*len(beta))

    #         # amc = amc - np.mean(amc)
    #         # ab = ab - np.mean(ab)

    #         mc_ix = np.argsort(amc)
    #         b_ix = np.argsort(ab)

    #         pmc_ix = np.argsort(pmc)
    #         pb_ix = np.argsort(pb)

    #         mc = mc[mc_ix, :] #SPKS,sorted by amp ix
    #         amc = amc[mc_ix] #AMP, sorted by amp ix

    #         mc_p = mc[pmc_ix, :] #SPKS sorted by phz ix
    #         pmc = pmc[pmc_ix] #Phz, sorted by phz ix

    #         b = b[b_ix, :] #Beta osrted by amp ix
    #         ab = ab[b_ix] #Beta Amp sorted by aetea amp ix

    #         pbeta = b[pb_ix, :] #Beta spks sorted by phz ix
    #         pb = pb[pb_ix] #Beta phz sorted by phz ix


    #         #Use the same amplitude bins for MC and Beta control: 

    #         fact = np.floor(len(amc)/float(n_b))
    #         fact2 = np.floor(len(ab)/float(n_b))

    #         tmp_am= np.hstack((amc, ab))
    #         bins = []
    #         for i in np.arange(0, 101, 5):
    #             bins.append(np.percentile(tmp_am, i))

    #         amc_labs = np.digitize(amc, bins)
    #         ab_labs = np.digitize(ab, bins)


    #         #data frame creation: 
    #         # Sub = namedtuple('Sub', ['beta_amp_id', 'spk','task_id']) 
    #         # for un_id in range(nunits):              
    #         #     df = pt.DataFrame()
    #         #     for ii, ma in enumerate(amc_labs):
    #         #         df.insert(Sub(ma, mc[ii, un_id], 'mc')._asdict())  
    #         #     for ii, ba in enumerate(ab_labs):
    #         #         df.insert(Sub(ba, b[ii, un_id], 'beta')._asdict()) 
    #         #     aov = df.anova('spk', sub='beta_amp_id', wfactors=['task_id'])
    #         #     res = extract_for_apa('task_id', aov)

    #         #     spk_sig[d, bi, un_id] = res['p']


    #         mc = mc[:fact*n_b, :].reshape(n_b, fact, nunits)
    #         amc = amc[:fact*n_b].reshape(n_b, fact)

    #         mc_p = mc_p[:fact*n_b, :].reshape(n_b, fact, nunits)
    #         pmc = pmc[:fact*n_b].reshape(n_b, fact)

    #         b = b[:fact2*n_b, :].reshape(n_b, fact2, nunits)
    #         ab = ab[:fact2*n_b].reshape(n_b, fact2)

    #         pbeta = pbeta[:fact2*n_b].reshape(n_b, fact2, nunits)
    #         pb = pb[:fact2*n_b].reshape(n_b, fact2)


    #         mn_mc = np.sum(mc, axis=1)/(fact/1000.)
    #         mn_amc = np.mean(amc, axis=1)

    #         mn_mc_p = np.sum(mc_p, axis=1)/(fact/1000.)
    #         mn_pmc = np.array([ang_mean(pmc[i,:]) for i in range(n_b)])

    #         mn_b = np.sum(b, axis=1)/(fact2/1000.)
    #         mn_ab = np.mean(ab, axis=1)

    #         mn_pbeta = np.sum(pbeta, axis=1)/(fact2/1000.)
    #         mn_pb = np.array([ang_mean(pb[i, :]) for i in range(n_b)])

    #         popts = {}
    #         popts2 = {}
    #         f, ax = plt.subplots(nrows=10 , ncols=10)
    #         beta_hat = np.linspace(-2.8, -1.6, 100)
    #         beta_hat_p = np.linspace(-np.pi, np.pi, 100)

    #         stats={}
    #         diff = 0
    #         one_diff_ttest = 0
    #         one_diff_slp = 0
    #         same = 0

    #         for i in range(nunits):
    #             axi = ax[i/10, i%10]
                
    #             if plot_amp: 

    #                 slp, intc, r_v, p_v, ste = scipy.stats.linregress(mn_amc, mn_mc[:, i])

    #                     # try:
    #                     #     pop, pcov = scipy.optimize.curve_fit(fcn2, mn_amc, mn_mc[:, i])
    #                     #     #popts[i] = pop
    #                     #     axi.plot(beta_hat, fcn2(beta_hat, *pop), 'b')
    #                     # except:
    #                     #     print 'mc: ', i
                    
    #                 #axi.plot(beta_hat, intc+(slp*beta_hat), 'b')
    #                 #axi.plot(mn_amc, mn_mc[:,i], 'b.')
    #                     #axi.text(10, 30, 'R2: '+str(r_v**2), fontsize=5)
                    
    #                 slp2, intc2, r_v2, p_v2, ste2 = scipy.stats.linregress(mn_ab, mn_b[:, i])
    #                     # try:
    #                     #     pop2, pcov2 = scipy.optimize.curve_fit(fcn2, mn_ab, mn_b[:, i])
    #                     #     #popts2[i] = pop2
    #                     #     axi.plot(beta_hat, fcn2(beta_hat, *pop2), 'r')
    #                     #     #
    #                     # except:
    #                     #     print i
    #                     #     pass
    #                 #axi.plot(beta_hat, intc2+(slp2*beta_hat), color_dict['b'+str(bi)])
    #                 #axi.plot(mn_ab, mn_b[:, i], color_dict['b'+str(bi)]+'.')
    #                     #axi.text(10, 40, 'R2: '+str(r_v2**2), fontsize=5)
    #                 #axi.set_title('Unit: '+Params[d]['sorted_un'][i])

    #                 #Test for slope significance of slopes and means: 
    #                 #See: http://www.real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
    #                 z = (slp - slp2)/np.sqrt(ste**2 + ste2**2)
    #                 p = 2*(1 - scipy.stats.norm.cdf(np.abs(z)))
    #                 try:
    #                     spk_sig[d][i] = (slp, slp2, p)
    #                 except:
    #                     spk_sig[d] = {}
    #                     spk_sig[d][i] = (slp, slp2, p)
    # f, ax = plt.subplots(3, 2)
    # for i_d, d in enumerate(days):
    #     axi = ax[i_d/2, i_d%2]
    #     for i in spk_sig[d]:
    #         axi.plot(spk_sig[d][i][0], spk_sig[d][i][1], '.')


    #                 t, stats[d, i, 'mn'] = scipy.stats.ttest_ind(mn_mc[:, i], mn_b[:, i])
    #                 stats[d, i, 'slp'] = p

    #                 if np.logical_and(stats[d, i, 'mn'] < 0.05, stats[d, i, 'slp'] < 0.05):
    #                     diff += 1
    #                 elif stats[d, i, 'mn'] < 0.05:
    #                     one_diff_ttest += 1
    #                 elif stats[d, i, 'slp'] < 0.05:
    #                     one_diff_slp += 1
    #                 else:
    #                     same += 1
    #             elif plot_phz:
    #                 p3, p2, p1 = fitSine(mn_pmc, mn_mc_p[:, i], 1.)
    #                 axi.plot(beta_hat_p, fcn_cos(beta_hat_p, p1, p2, p3), 'b')
    #                 axi.plot(mn_pmc, mn_mc_p[:, i], 'b.')

    #                 p3, p2, p1 = fitSine(mn_pb, mn_pbeta[:, i], 1.)
    #                 axi.plot(beta_hat_p, fcn_cos(beta_hat_p, p1, p2, p3), color_dict['b'+str(bi)])
    #                 axi.plot(mn_pb, mn_pbeta[:, i], 'r.')
    #                 axi.set_title('Unit: '+Params[d]['sorted_un'][i])


    #         # print d, diff, one_diff_ttest, one_diff_slp, same
    #         # tot = np.sum([same, one_diff_slp, one_diff_ttest, diff])
    #         # print 'frac: ', d, diff/float(tot), one_diff_ttest/float(tot), one_diff_slp/float(tot), same/float(tot)
    #         plt.tight_layout()
    return 10

def beta_met_to_mapping(day, blocks, mc_indicator, lfp_lab, BC_bin, S_bin, Params, bin_kin_signal, n_b = 15,):
    ''' Method to downsample FR vs. continuous beta metrics
        Input: n_b = number of bins to downsample to 
    '''

    #Index metics: 
    amp = np.log10(BC_bin[day])
    phz = BC_bin[day, 'phz']
    spk = S_bin[day]

    ntrls, nbins, nunits = spk.shape

    #Get MC vs. NF trials: 
    day_trial_type = []
    day_bin_kin = []
    day_lfp_lab = []

    for ib, b in enumerate(blocks):
        n_trials = len(lfp_lab[b, day])
        if int(mc_indicator[ib]) == 1:
            tt = ['mc']*n_trials
        elif int(mc_indicator[ib]) == 0:
            tt = ['beta']*n_trials
        else:
            raise

        day_trial_type.append(tt)
        day_bin_kin.append(bin_kin_signal[b, day])
        day_lfp_lab.append(lfp_lab[b, day])

    day_trial_type = np.hstack(day_trial_type)
    day_bin_kin = np.vstack((day_bin_kin))
    day_lfp_lab = np.hstack((day_lfp_lab))

    train_ix = np.nonzero(day_trial_type == 'mc')[0]
    test_ix = np.nonzero(day_trial_type == 'beta')[0]

    ix_hold = np.arange(nbins)
    #ix_hold = np.arange(nbins*0.6)
    ix_reach = np.arange(nbins*0.6, nbins)

    ix_all = [ix_hold.astype(int), ix_reach.astype(int)]
    master_res = {}

    for ix, (trial_ix, epoch_name) in enumerate(zip(ix_all, ['hold', 'reach'])):
        f, ax = plt.subplots(nrows=5, ncols=5)

        #Use only trials to target 64: 
        #ix64 = np.nonzero(np.logical_or(day_lfp_lab ==64, day_lfp_lab==88))[0]
        ix64 = np.arange(len(train_ix))
        tmp_ix_train = np.ix_(train_ix[ix64], trial_ix)
        tmp_ix_test = np.ix_(test_ix, trial_ix)

        #Manual control: 
        if ix == -1:
            hold_kin_ix0_mc, hold_kin_ix1_mc = np.nonzero(day_bin_kin[tmp_ix_train]==0)
            hold_kin_ix0_b,  hold_kin_ix1_b =  np.nonzero(day_bin_kin[tmp_ix_test]==0)
        #NF control
        else:
            hold_kin_ix0_mc, hold_kin_ix1_mc = np.nonzero(day_bin_kin[tmp_ix_train]>-1)
            hold_kin_ix0_b,  hold_kin_ix1_b =  np.nonzero(day_bin_kin[tmp_ix_test]>-1)

    
        AMC = amp[tmp_ix_train]
        AB = amp[tmp_ix_test]

        # MC hold:
        tmp_ix_ = np.ix_(train_ix, trial_ix, np.arange(nunits))
        MC = spk[tmp_ix_]
        mc = []
        amc = []

        for z, (iy, iz) in enumerate(zip(hold_kin_ix0_mc, hold_kin_ix1_mc)):
            mc.append(MC[iy, iz, :])
            amc.append(AMC[iy, iz])
        mc = np.vstack((mc))
        amc = np.hstack((amc))


        tmp_ix_ = np.ix_(test_ix, trial_ix, np.arange(nunits))
        B = spk[tmp_ix_]
        #Beta hold: 
        b = []
        ab = []
        for z, (iy, iz) in enumerate(zip(hold_kin_ix0_b, hold_kin_ix1_b)):
            try:
                b.append(B[iy, iz, :])
            except:
                print 'no b'
                b = np.array([])
            try:
                ab.append(AB[iy, iz])
            except:
                print 'no ab'
                ab = np.array([])
        try:
            b = np.vstack((b))
            ab = np.hstack((ab))
        except:
            print 'no ab stack'

        # else:
        #     #MC reach
        #     tmp_ix_ = np.ix_(train_ix, trial_ix, np.arange(nunits))
        #     MC = spk[tmp_ix_]
        #     mc = MC.reshape(len(train_ix)*len(trial_ix), nunits)
        #     hold_kin_ix0_mc, hold_kin_ix1_mc = np.nonzero(day_bin_kin[train_ix, trial_ix] > -1)

        #     #Beta reach: 
        #     tmp_ix_ = np.ix_(test_ix, trial_ix, np.arange(nunits))
        #     B = spk[tmp_ix_]
        #     b = B.reshape(len(test_ix)*len(trial_ix), nunits)




        # tmp_ix_2 = np.ix_(train_ix, trial_ix)
        # AMC = amp[tmp_ix_2]
        # PMC = phz[tmp_ix_2]



        # tmp_ix_2 = np.ix_(test_ix, beta)
        # AB = amp[tmp_ix_2]
        # PB = phz[tmp_ix_2]

        # #Now, reshape, then sort!
        # mc = MC.reshape(len(train_ix)*len(trial_ix), nunits)
        # amc = AMC.reshape(len(train_ix)*len(trial_ix))
        # pmc = PMC.reshape(len(train_ix)*len(trial_ix))

        # b = B.reshape(len(test_ix)*len(beta), nunits)
        # ab = AB.reshape(len(test_ix)*len(beta))
        # pb = PB.reshape(len(test_ix)*len(beta))

        spk_bundle = [mc, b]
        amp_bundle = [amc, ab]
        #phz_bundle = [pmc, pb]

        #Use 3/4 of the data instead 1/2 of data, for MC only. 

        tmp1 = set(range(len(mc)))
        tmp2 = set(range(0, len(mc), 4))
        tmp3 = set(range(1, len(mc), 4))

        ix1 = np.array(list(tmp1.difference(tmp2)))
        ix2 = np.array(list(tmp1.difference(tmp3)))

        #ix1 = np.arange(0, len(mc), 2)
        #ix2 = np.arange(1, len(mc), 2)

        ix1b = np.arange(0, len(b), 2)
        ix2b = np.arange(1, len(b), 2)

        portion_bundle = [[ix1, ix2, np.hstack((ix1,ix2))], [ix1b, ix2b, np.hstack((ix1b, ix2b))]]

        
        color = dict(hold=['b', 'r'], reach=['g','k'])

        for task, spks in enumerate([mc, b]):
            #for metric_ix, (mets, metric, ang_mean_use) in enumerate(zip([amp_bundle, phz_bundle], ['slp', 'pref_phz'], [False, True])):
            for metric_ix, (mets, metric, ang_mean_use) in enumerate(zip([amp_bundle], ['slp'], [False])):    
                met = mets[task]
                spks = spk_bundle[task]
                port = portion_bundle[task]

                for p, portion_ix in enumerate(port):
                    if len(portion_ix) > 0:
                        res, met_x, spk_x = sort_stuff(met[portion_ix], spks[portion_ix, :], n_b, metric=metric, ang_mean_use=ang_mean_use)
                        master_res[task, metric, p, epoch_name] = res

                        if np.logical_and(p == 2, metric=='slp'):
                            for jj in range(spk_x.shape[1]):
                                try:
                                    axi = ax[jj/5, jj%5]
                                    axi.plot(met_x, 1000*spk_x[:, jj], color[epoch_name][task]+'.')
                                    axi.plot(met_x, 1000*((np.array(met_x)*float(res['slope'][jj]))+ float(res['intcp'][jj])), color[epoch_name][task]+'-')
                                    #pop, pcov = scipy.optimize.curve_fit(fcn, met_x, 1000*spk_x[:, jj])
                                    #axi.plot(met_x, fcn2(met_x, *pop), color[task])
                                    axi.set_title('sig'+Params[day]['sorted_un'][jj])
                                except:
                                    pass
    plt.tight_layout()
    #plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/NeuronPaper/JNeuroDraft/canolty_amp_tuning_'+day+'.eps', format='eps', dpi=300)
    return master_res

def sort_stuff(met, spks, n_b, metric='slp', ang_mean_use=False):
    #Put metric into bins:
    bins = []
    for i in np.linspace(0, 100, n_b+1):
        bins.append(np.percentile(met, i))
    bins = np.array(bins)
    dig_met = np.digitize(met, bins)
    bins_x = np.array([(b1+b2)/2. for ib, (b1, b2) in enumerate(zip(bins[:-1], bins[1:]))])
    
    met_x = []
    spk_x = []
    for i in np.unique(dig_met):
        if i > n_b:
            print 'ignoring ', i
        else:
            ix = np.nonzero(dig_met == i)[0]
            if ang_mean_use:
                met_x.append(ang_mean(met[ix]))
            else:
                i = np.nonzero(np.abs(met)==np.inf)[0]
                met[i] = np.nan
                met_x.append(np.nanmean(met[ix]))
            spk_x.append(np.mean(spks[ix, :], axis=0))

    spk_x = np.vstack((spk_x))
    
    if metric == 'slp':
        slope = []
        stder = []
        intcp = []
        pv = []
        for i in range(spk_x.shape[1]):
            slp, intc, r_v, p_v, ste = scipy.stats.linregress(met_x, spk_x[:, i])
            slope.append(slp)
            stder.append(stder)
            intcp.append(intc)
            pv.append(p_v)
        res = dict(slope=slope, stder=stder, intcp=intcp, p_v=pv)
    elif metric == 'pref_phz':
        pp = []
        for i in range(spk_x.shape[1]):
            p_ix = np.argmax(spk_x[:,i])
            pp.append(bins_x[p_ix])
        res = dict(pref_phz=pp)
    return res, met_x, spk_x

def fcn(x, p1, p2, p3, p4):
    ''' Canolty beta - to - rate mapping , 2012'''
    return p1 + p2*(np.tanh((x - p3)/(2*p4)))

def fcn2(x, p1, p2, p3, p4):
    return p1 + p2/((1 + np.exp(-1*p3*(x - p4))))

def fcn_cos(x, p1, p2, p3):
    return p1+p2*np.cos(x - p3)

def ang_mean(x):
    x_ = np.cos(x)
    y_ = np.sin(x)
    vect_ = np.sum(x_) +np.sum(y_)*1j
    return np.angle(vect_)

from pylab import *
from math import atan2
 
def fitSine(tList,yList,freq):
   '''
       freq in Hz
       tList in sec
   returns
       phase in degrees
   '''
   b = matrix(yList).T
   rows = [ [sin(freq*2*pi*t), cos(freq*2*pi*t), 1] for t in tList]
   A = matrix(rows)
   (w,residuals,rank,sing_vals) = lstsq(A,b)
   phase = atan2(w[1,0],w[0,0])*180/pi
   amplitude = norm([w[0,0],w[1,0]],2)
   bias = w[2,0]
   return phase,amplitude,bias

def extract_for_apa(factor, aov, values = ['F', 'mse', 'eta', 'p']):
    results = {}
    for key,result in aov[(factor,)].iteritems():
        if key in values:
            results[key] = result
    return results


if __name__ == '__main__':
    main_mp()
    # cr = main(test=False, animal='cart')
    # plot_chief_results(cr)

