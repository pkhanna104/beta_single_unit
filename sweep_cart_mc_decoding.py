
import state_space_cart as ssc
import scipy.stats as stats
import predict_kin_w_spk
import numpy as np
import statsmodels.api as sm
import sklearn.lda 
import matplotlib.pyplot as plt
import pickle
import gc
''' Goal: for each channel, determine the optimal binsize, lag, and decoding model based on signfiicant R^2'''


def extract_all():
    Stats = {}

    for i_d, day in enumerate(ssc.master_days):
        R2_dict = {}
        PV_dict = {}

        keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False, 
            days=[day], blocks=['a'], animal='cart', all_cells=True, mc_indicator=['1'])
    
        print 'decoder dict starting for ', day, ' using days: ', days
        spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = predict_kin_w_spk.get_full_blocks_cart(keep_dict, days, blocks, mc_indicator, kin_type='endpt', mc_only=False)
        blk = 'a'
            #Smooth kinematics: 
            #kin_smooth_len = 151
            #kin_smooth_std = 51
            #window = scipy.signal.gaussian(kin_smooth_len, std=kin_smooth_std)

            # window = window/np.sum(window)
            # smooth_kin_dict = {}
            # for k in kin_dict.keys():
            #     smooth_kin_dict[k] = np.zeros_like(kin_dict[k])
            #     for j in range(4):
            #         smooth_kin_dict[k][:, j] = np.convolve(window, kin_dict[k][:,j], mode='same')

            #Bin stuff: 
        for binsize in [75, 100, 125, 150, 200]:
            Bin_K = {}
            Bin_N = {}
            Bin_B = {}
            Bin_H = {}
            Order_Spks = []
            for k in spk_dict.keys():
                Bin_N[k] = {}
                for u in spk_dict[k].keys():
                    Order_Spks.append(u)
                    Bin_N[k][u] = predict_kin_w_spk.bin_(spk_dict[k][u], binsize, mode='cnts')
                print 'done binning for neural :', k
                Bin_K[k] = predict_kin_w_spk.bin_(kin_dict[k], binsize, mode='mean')
                print 'done binning for kin :', k
                Bin_B[k] = predict_kin_w_spk.bin_(beta_dict[k][1,:], binsize, mode='mode')
                #Bin_H[k] = predict_kin_w_spk.bin_(hold_dict[k], binsize, mode='mode')
                print 'done binning for beta :', k
        
            k = (blk, day)
            mat_Bin_N_tmp = []

            for u in Bin_N[k]:
                mat_Bin_N_tmp.append(Bin_N[k][u])

            obs = np.hstack((mat_Bin_N_tmp))

            for model in ['vel', 'spd', 'vel_spd', 'pos_vel_spd']:
                if model == 'vel':
                    kin = Bin_K[k][:, [2, 3]]
                elif model == 'spd':
                    spd = np.sum(Bin_K[k][:, [2, 3]]**2, axis=1)
                    kin = spd[:, np.newaxis]
                elif model == 'vel_spd':
                    spd = np.sum(Bin_K[k][:, [2, 3]]**2, axis=1)
                    kin = np.hstack(( Bin_K[k][:, [2]], spd[:, np.newaxis], Bin_K[k][:, [3]] ))
                elif model == 'pos_vel_spd':
                    spd = np.sum(Bin_K[k][:, [2, 3]]**2, axis=1)
                    kin = np.hstack(( Bin_K[k][:, [0, 1]], Bin_K[k][:, [2]], spd[:, np.newaxis], Bin_K[k][:, [3]] ))                    


                if obs.shape[0] != kin.shape[0]:
                    mx = np.min([obs.shape[0], kin.shape[0]])
                    print 'obs, kin mismatch: ', obs.shape[0], kin.shape[0]
                    raise Exception
                    # obs = obs[:mx, :]
                    # kin = kin[:mx, :]

                #For each channel, get R^2 and P-value
                for ik, key in enumerate(Order_Spks):
                    x = kin
                    y = obs[: , ik]
                    res = sm.OLS(y, x).fit()
                    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    R2_dict[binsize, model, key] = res.rsquared
                    PV_dict[binsize, model, key] = res.pvalues
        Stats[day, 'R2'] = R2_dict
        Stats[day, 'PV'] = PV_dict
        Stats[day, 'Order_Spks'] = Order_Spks
    return Stats

def plot_stats(Stats):
    days = ssc.master_days
    for i_d, day in enumerate(days):
        f, axi = plt.subplots()

        for i_b, bin in enumerate([75, 100, 125, 150, 200]):
            
            R2 = []
            for i_m, mod in enumerate(['vel', 'spd', 'vel_spd', 'pos_vel_spd']):
                r2 = []
                
                for i_k, key in enumerate(Stats[day, 'Order_Spks']):
                    pv = Stats[day, 'PV'][bin, mod, key]
                    if np.any(pv<0.05):
                        r2.append(Stats[day, 'R2'][bin, mod, key])

                R2.append(r2)
            axi.boxplot(R2, positions = (i_b*4) + np.arange(4))
        axi.set_xlim([0, 20])

def train_lda():
    Clf_Dict = {}
    f, ax = plt.subplots(ncols=5)

    dict_fname='ssc_get_cells_output_all_cells.pkl'
    keep_dict = pickle.load(open(dict_fname))
    days = keep_dict.pop('days')
    blocks = keep_dict.pop('blocks')
    mc_indicator = keep_dict.pop('mc_indicator')

    for i_d, day in enumerate(days):
        # keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False, 
        #     days=[day], blocks=['a'], animal='cart', all_cells=True, mc_indicator=['1'])
    
        kd = {}
        kd[day] =keep_dict[day]

        print 'decoder dict starting for ', day, ' using block: ', blocks[i_d][0]
        spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = predict_kin_w_spk.get_full_blocks_cart(kd, [day], ['a'], ['1'], kin_type='endpt', mc_only=False)
        blk = 'a'
            #Smooth kinematics: 
            #kin_smooth_len = 151
            #kin_smooth_std = 51
            #window = scipy.signal.gaussian(kin_smooth_len, std=kin_smooth_std)

            # window = window/np.sum(window)
            # smooth_kin_dict = {}
            # for k in kin_dict.keys():
            #     smooth_kin_dict[k] = np.zeros_like(kin_dict[k])
            #     for j in range(4):
            #         smooth_kin_dict[k][:, j] = np.convolve(window, kin_dict[k][:,j], mode='same')

            #Bin stuff: 
        for bi, binsize in enumerate([100]):
            Bin_K = {}
            Bin_N = {}
            Bin_B = {}
            Bin_H = {}
            Order_Spks = []
            for k in spk_dict.keys():
                Bin_N[k] = {}
                for u in spk_dict[k].keys():
                    Order_Spks.append(u)
                    Bin_N[k][u] = predict_kin_w_spk.bin_(spk_dict[k][u], binsize, mode='cnts')
                print 'done binning for neural :', k
                Bin_K[k] = predict_kin_w_spk.bin_(kin_dict[k], binsize, mode='mean')
                print 'done binning for kin :', k
                Bin_B[k] = predict_kin_w_spk.bin_(beta_dict[k][1,:], binsize, mode='mode')
                #Bin_H[k] = predict_kin_w_spk.bin_(hold_dict[k], binsize, mode='mode')
                print 'done binning for beta :', k
        
            k = (blk, day)
            mat_Bin_N_tmp = []

            for u in Bin_N[k]:
                mat_Bin_N_tmp.append(Bin_N[k][u])

            obs = np.hstack((mat_Bin_N_tmp))

            #train LDA on 'slow', 'fast': 0, 1:
            spd = np.sum(Bin_K[k][:, [2, 3]]**2, axis=1)

            cmap = ['r', 'g', 'b', 'k', 'c']
            for bini, thresh in enumerate([3.5]):
                spd_binary = np.zeros_like(spd)
                spd_binary[spd > thresh] = 1

                clf = sklearn.lda.LDA()
                clf.fit(obs, spd_binary)
                y_hat = clf.predict(obs)

                kix0 = np.nonzero(spd_binary==0)[0]

                bix0 = np.nonzero(Bin_B[k][0, kix0]==0)[0]
                bix1 = np.nonzero(Bin_B[k][0, kix0]==1)[0]

                mn_bix0_kix0_hat = np.mean(y_hat[kix0[bix0]])
                mn_bix1_kix0_hat = np.mean(y_hat[kix0[bix1]])

                t, p = stats.ttest_ind(y_hat[kix0[bix0]], y_hat[kix0[bix1]])
                print thresh, day, p                

                sem_b0 = np.std(y_hat[kix0[bix0]])/np.sqrt(len(bix0))
                sem_b1 = np.mean(y_hat[kix0[bix1]])/np.sqrt(len(bix1))

                ax[bini].errorbar([0, 1], [mn_bix0_kix0_hat, mn_bix1_kix0_hat], yerr=[sem_b0, sem_b1], color=cmap[bini])
                ax[bini].set_ylim([0, 1])

                Clf_Dict[day, binsize, thresh] = clf
    pickle.dump(Clf_Dict, open('day_clf_dict_all.pkl', 'wb'))
    return Clf_Dict

def test_lda(day, mc_test=False, Clf_Dict=None, dict_fname='ssc_get_cells_output_all_cells.pkl'):
    if Clf_Dict is None:
        Clf_Dict = pickle.load(open('day_clf_dict_all.pkl'))
    f, ax = plt.subplots()

    if dict_fname is None:
        keep_dict, days, blocks, mc_indicator = ssc.get_cells(plot=False, all_cells=True, only3478=False)
    else:
        keep_dict = pickle.load(open(dict_fname))
        days = keep_dict.pop('days')
        blocks = keep_dict.pop('blocks')
        mc_indicator = keep_dict.pop('mc_indicator')

    gc.collect()
    ix = np.array([j for j, i in enumerate(days) if day==i])
    if len(ix) == 1:
        i_d = int(ix)
        print 'blocks: ', blocks[i_d]
        print 'mc: ', mc_indicator[i_d]
    else:
        raise Exception
    # keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False, 
    #     days=[day], blocks=[ssc.master_blocks[i_d]], animal='cart', all_cells=True, mc_indicator=[ssc.master_mc_indicator[i_d]])
    
    Y = dict()
    gc.collect()


    YHAT0 = []
    YHAT1 = []

    for ib, blk in enumerate(blocks[i_d]):
        proceed = False
        if mc_test:
            if mc_indicator[i_d][ib] == '1':
                proceed = True
        else:
            if mc_indicator[i_d][ib] == '0':
                proceed = True

        if proceed:
            print 'testing day: ', day, ' block: ', blk
            get_blocks = [blocks[i_d][ib]]
            get_mci = [mc_indicator[i_d][ib]]
            kd = {}
            kd[day] = keep_dict[day]
            print 'decoder dict starting for ', day, ' using blocks: ', get_blocks
            spk_dict, lfp_dict, beta_dict, kin_dict, hold_dict = predict_kin_w_spk.get_full_blocks_cart(kd, [day], get_blocks, get_mci, kin_type='endpt', mc_only=False, nf_only=False)
            print 'done w/ line 226!'
            gc.collect()
            #Bin stuff: 
            binsize = 100
            Bin_K = {}
            Bin_N = {}
            Bin_B = {}
            Bin_H = {}
            Order_Spks = []
            for k in spk_dict.keys():
                Bin_N[k] = {}
                for u in spk_dict[k].keys():
                    Order_Spks.append(u)
                    Bin_N[k][u] = predict_kin_w_spk.bin_(spk_dict[k][u], binsize, mode='cnts')
                print 'done binning for neural :', k
                Bin_K[k] = predict_kin_w_spk.bin_(kin_dict[k], binsize, mode='mean')
                print 'done binning for kin :', k
                Bin_B[k] = predict_kin_w_spk.bin_(beta_dict[k][1,:], binsize, mode='mode')
                #Bin_H[k] = predict_kin_w_spk.bin_(hold_dict[k], binsize, mode='mode')
                print 'done binning for beta :', k


                k = (blk, day)
                mat_Bin_N_tmp = []

                for u in Bin_N[k]:
                    mat_Bin_N_tmp.append(Bin_N[k][u])

                obs = np.hstack((mat_Bin_N_tmp))

                #train LDA on 'slow', 'fast': 0, 1:
                spd = np.sum(Bin_K[k][:, [2, 3]]**2, axis=1)

                cmap = ['r', 'g', 'b', 'k', 'c']
                bini = 0
                thresh = 3.5

                spd_binary = np.zeros_like(spd)
                spd_binary[spd > thresh] = 1

                clf = Clf_Dict[day, binsize, thresh]
                y_hat = clf.predict(obs)

                kix0 = np.nonzero(spd_binary==0)[0]

                bix0 = np.nonzero(Bin_B[k][0, kix0]==0)[0]
                bix1 = np.nonzero(Bin_B[k][0, kix0]==1)[0]

                YHAT0.append(y_hat[kix0[bix0]])
                YHAT1.append(y_hat[kix0[bix1]])

                #mn_bix0_kix0_hat = np.mean(y_hat[kix0[bix0]])
                #mn_bix1_kix0_hat = np.mean(y_hat[kix0[bix1]])

                #t, p = stats.ttest_ind(y_hat[kix0[bix0]], y_hat[kix0[bix1]])
                #print thresh, day, p                

                #sem_b0 = np.std(y_hat[kix0[bix0]])/np.sqrt(len(bix0))
                #sem_b1 = np.mean(y_hat[kix0[bix1]])/np.sqrt(len(bix1))

                Y['yhat', blk] = y_hat
                Y['spd', blk] = spd
                Y['kin_binary', blk] = spd_binary
                Y['beta_binary', blk] = Bin_B[k][0, :]
            
    y0 = np.hstack((YHAT0))
    y1 = np.hstack((YHAT1))

    ax.errorbar([0, 1], [np.mean(y0), np.mean(y1)], yerr=[np.std(y0)/np.sqrt(len(y0)), np.std(y1)/np.sqrt(len(y1))], color=cmap[i_d])
    ax.set_ylim([0, 1])
    if mc_test:
        pickle.dump(Y, open('day_'+day+'_lda_res_mctest.pkl', 'wb'))
    else:
        pickle.dump(Y, open('day_'+day+'_lda_res.pkl', 'wb'))

def plot_tested_lda(mc_test=False):
    cmap = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]
    f, ax = plt.subplots()

    days = ['011315', '011415', '011515', '011615']
    mn_days = []
    sum_days = np.zeros((4, 2, 2))
    for i_d, day in enumerate(days):
        if mc_test:
            dat = pickle.load(open('day_'+day+'_lda_res_mctest.pkl'))
        else:
            dat = pickle.load(open('day_'+day+'_lda_res.pkl'))
        keys = np.vstack((dat.keys()))
        blocks = np.array([k[1] for i, k in enumerate(keys) if k[0]=='yhat'])

        y0 = []
        y1 = []

        for i_b, blk in enumerate(blocks):
            kix0 = np.nonzero(dat['kin_binary', blk]==0)[0]

            bix0 = np.nonzero(dat['beta_binary', blk][kix0]==0)[0]
            bix1 = np.nonzero(dat['beta_binary', blk][kix0]==1)[0]

            y0.append(dat['yhat', blk][kix0[bix0]])
            y1.append(dat['yhat', blk][kix0[bix1]])

            sum_days[i_d, 0, 0] += np.sum(np.abs(dat['yhat', blk][kix0[bix0]]-1))
            sum_days[i_d, 0, 1] += np.sum(np.abs(dat['yhat', blk][kix0[bix0]]))
            sum_days[i_d, 1, 0] += np.sum(np.abs(dat['yhat', blk][kix0[bix1]]-1))
            sum_days[i_d, 1, 1] += np.sum(np.abs(dat['yhat', blk][kix0[bix1]]))

        mn0 = np.mean(np.hstack((y0)))
        st0 = np.std(np.hstack((y0)))/np.sqrt(len(np.hstack((y0))))
        
        mn1 = np.mean(np.hstack((y1)))
        st1 = np.std(np.hstack((y1)))/np.sqrt(len(np.hstack((y1))))

        ax.errorbar([0, 1], [mn0, mn1], yerr=[st0, st1], color=np.array(cmap[i_d])/255., linewidth=5)
        mn_days.append([mn0, mn1])

    ax.set_xlim([-.5, 1.5])

    sides = np.vstack((mn_days))
    t, p = stats.ttest_rel(sides[:, 0], sides[:, 1])
    print t, p

    if mc_test:
        ax.plot([0, 0], [.5, .52], 'k-', linewidth=2)
        ax.plot([0, 1], [.52, .52], 'k-', linewidth=2)
        ax.plot([1, 1], [.46, .52], 'k-', linewidth=2)
        ax.text(0.5, .55, 'p = 0.307', horizontalalignment='center')

    else:
        ax.plot([0, 0], [.45, .47], 'k-', linewidth=2)
        ax.plot([0, 1], [.47, .47], 'k-', linewidth=2)
        ax.plot([1, 1], [.37, .47], 'k-', linewidth=2)
        ax.text(0.5, .5, '**')
    if mc_test:
        plt.savefig('cart_lda_decoding_Fig7end_mc.eps', format='eps')
    else:
        plt.savefig('cart_lda_decoding_Fig7end.eps', format='eps')




