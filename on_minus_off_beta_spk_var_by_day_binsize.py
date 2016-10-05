import predict_kin_w_spk
import state_space_w_beta_bursts as ssbb
import numpy as np
import matplotlib.pyplot as plt

cmap=plt.get_cmap('RdBu')
cmap.N = 8


def get_cell_var_wrt_beta():
    #binsizes = [5, 10, 25, 50, 75, 100]
    binsizes= [100]
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False)

    for b in binsizes:
        print 'starting bin: ', b
        # #Bin / smooth spike and beta dictionaries
        B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, beta_amp_smooth=50, binsize=b)

        kin_signal_dict, binned_kin_signal_dict, binary_kin_signal, rt_bin = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, b, smooth = 50)

        #f, ax = plt.subplots(nrows=2,ncols=2)
        #f2, ax2 = plt.subplots(nrows=2,ncols=2)
        f3, ax3 = plt.subplots(nrows=2, ncols=2)
        master_mets = []

        for i_d, d in enumerate(days):

            ntrl, nbins, nunits = S_bin[d].shape

            mc_ix = np.nonzero(Params[d]['day_lfp_labs']<80)[0]
            beta_ix = np.nonzero(Params[d]['day_lfp_labs']>80)[0]

            day_kin = []
            day_binary_kin = []
            for ib, blk in enumerate(blocks[i_d]):
                day_kin.append(np.swapaxes(binned_kin_signal_dict[blk, d], 1, 2))
                day_binary_kin.append(binary_kin_signal[blk, d])
            day_kin = 100*np.vstack((day_kin)) # meters --> cm
            day_binary_kin = np.vstack((day_binary_kin))

            # bin_day_kin = np.zeros((day_kin.shape[0], day_kin.shape[1]))
            # for ii in range(day_kin.shape[0]):
            #     for jj in range(day_kin.shape[1]):
            #         if np.logical_or(day_kin[ii, jj, 2] > 0.5, day_kin[ii, jj, 3] > 0.5):
            #             bin_day_kin[ii, jj] = 1

            var_dict = {}
            mn_dict = {}
            #ax[i_d/2, i_d%2].plot([-.2, 4], [0, 0], 'k-')
            #ax[i_d/2, i_d%2].plot([0, 0], [-.2, 2], 'k-')

            for iix, trl_ix in enumerate([mc_ix, beta_ix]):

                bin_kn = day_binary_kin[trl_ix, :.6*nbins].reshape(-1)
                k_ix = np.nonzero(bin_kn==0)[0] #Extract hold indices

                beta = B_bin[d][trl_ix, :.6*nbins].reshape(-1)
                beta = beta[k_ix] #Extract beta events that occur during hold indices

                ix0 = np.nonzero(beta==0)[0]
                ix1 = np.nonzero(beta==1)[0]

                spk = S_bin[d][trl_ix, :.6*nbins, :].reshape(len(trl_ix)*(0.6*nbins), nunits)
                spk = spk[k_ix, :]
                spk = spk * b

                P = []
                W = []
                for i in range(nunits):
                    w, p = scipy.stats.levene(spk[ix0, i], spk[ix1, i])
                    dvar = np.var(spk[ix0, i]) - np.var(spk[ix1, i]) # off - on (+ is good!)

                    d2, p2 = scipy.stats.ks_2samp(spk[ix0, i], spk[ix1, i])
                    dmn = np.mean(spk[ix0, i]) - np.mean(spk[ix1, i])

                    var_dict[iix, i] = (dvar, p)
                    mn_dict[iix, i] = (dmn, p2)

            day_cnt = np.zeros((2, 2, 3)) #task x mn/var x (on>off, on=off, on<off)

            for i in range(nunits):
                for j in range(2):
                    for k, dct in enumerate([mn_dict, var_dict]):
                        if np.logical_and(dct[j, i][1] <= 0.05, dct[j, i][0] < 0):
                            day_cnt[j, k, 0] += 1
                        elif np.logical_and(dct[j, i][1] <= 0.05, dct[j, i][0] > 0):
                            day_cnt[j, k, 2] += 1
                        else:
                            day_cnt[j, k, 1] += 1

            day_cnt2 = day_cnt.copy()
            for j in range(2):
                for k in range(2):
                    day_cnt2[j, k, :] = day_cnt2[j, k, :]/float(np.sum(day_cnt2[j, k, :]))
            master_mets.append(day_cnt2)

            for j in range(2): #task
                for k in range(2): #Mean, variance
                    for l in range(3):
                        if l == 0:
                            ax3[j, k].bar((i_d*.1)+l, day_cnt[j, k, l]/float(np.sum(day_cnt[j, k, :])), width=.1, label=d, color=cmap(i_d))
                        else:
                            ax3[j, k].bar((i_d*.1)+l, day_cnt[j, k, l]/float(np.sum(day_cnt[j, k, :])), width=.1, color=cmap(i_d))
        tsks = ['Manual Control Hold', 'Neurofeedback Epoch']
        mtrc = ['Mean', 'Variance']
        res = np.mean(np.concatenate([arr[np.newaxis] for arr in master_mets]), axis=0)
        
        for j in range(2):
            for k in range(2):
                ax3[j, k].set_xticks([0.5, 1.5, 2.5])
                ax3[j, k].set_xticklabels(['On > Off', 'On == Off', 'On < Off'])
                ax3[j, k].set_ylabel('Perc of Cells per Day')
                ax3[j, k].set_ylim([0., 1.2])
                ax3[j, k].set_title(tsks[j]+ ', Metric: '+mtrc[k])
                for l, il in enumerate([0.1, 1.1, 2.1]):
                    if l == 1:
                        offs = 0.6
                    elif np.logical_and(j==1, l!=1):
                        offs = -0.2
                    else:
                        offs = 0.0
                    ax3[j, k].annotate(int(1000*res[j, k, l])/1000., xy=(il, 0.5+offs))
        ax3[1,1].legend()

        plt.tight_layout()
        plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/NeuronPaper/JNeuroDraft/mn_var_cells_on_off_25_40.eps', format='eps',dpi=300)


        #     for i in range(nunits):
        #         mc_ = var_dict[0, i]
        #         bt_ = var_dict[1, i]

        #         if np.logical_and(np.array([mc_[1]< 0.05]), np.array([bt_[1] < 0.05])):
        #             col = 'r*'
        #         elif np.logical_xor(np.array([mc_[1]< 0.05]), np.array([bt_[1] < 0.05])):
        #             col = 'b*'
        #         elif np.logical_not(np.array([mc_[1]< 0.05]), np.array([bt_[1] < 0.05])):
        #             col = 'k.'

        #         ax[i_d/2, i_d%2].plot(mc_[0], bt_[0], col)

        #     for i in range(nunits):
        #         mc_ = mn_dict[0, i]
        #         bt_ = mn_dict[1, i]

        #         if np.logical_and(np.array([mc_[1]< 0.05]), np.array([bt_[1] < 0.05])):
        #             col = 'r*'
        #         elif np.logical_xor(np.array([mc_[1]< 0.05]), np.array([bt_[1] < 0.05])):
        #             col = 'b*'
        #         elif np.logical_not(np.array([mc_[1]< 0.05]), np.array([bt_[1] < 0.05])):
        #             col = 'k.'

        #         ax2[i_d/2, i_d%2].plot(mc_[0], bt_[0], col)


        #     ax[i_d/2, i_d%2].set_xlabel('Var Off Beta - Var On Beta, Manual Control')
        #     ax[i_d/2, i_d%2].set_ylabel('Var Off Beta - Var On Beta, Beta Control')
        #     ax[i_d/2, i_d%2].set_title(d)

        #     ax2[i_d/2, i_d%2].set_xlabel('Mean Off Beta - Mean On Beta, Manual Control')
        #     ax2[i_d/2, i_d%2].set_ylabel('Mean Off Beta - Mean On Beta, Beta Control')
        #     ax2[i_d/2, i_d%2].set_title(d)

        # f.tight_layout()
        # f2.tight_layout()



