import numpy as np
import predict_kin_w_spk
import state_space_w_beta_bursts as ssbb
from state_space_spks import master_cell_list
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import seaborn

def plot_psth_by_task_by_ts(binsize=100, test=False, plot=False):
    keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=test)

    B_bin, BC_bin, S_bin, Params =  ssbb.bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, 
            beta_cont_dict, bef, aft, smooth=-1, binsize=binsize)

    kin_signal_dict, binned_kin_signal_dict, bin_kin_signal, binned_rt = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, binsize, smooth = 50)

    color_dict = {}
    color_dict['mc', 0] = 'k'
    color_dict['beta', 0] = 'lightseagreen'
    color_dict['beta', 1] = 'dodgerblue'
    color_dict['beta', 2] = 'gold'
    color_dict['beta', 3] = 'orangered'

    chi_sq_master_mat = dict()

    for i_d , day in enumerate(days):
        if plot:
            f, ax = plt.subplots(nrows=6, ncols=6)
            f2, ax2 = plt.subplots(nrows=6, ncols=6)
        day_spks = S_bin[day]*100 #Convert to Spike Cnts

        stat_mat = {}
        stat_mat_rt = {}


        rt_day = []
        for i_b, blk in enumerate(blocks[i_d]):
            rt_day.append(binned_rt[blk, day])
        rt_day = np.hstack((rt_day))

        lfp_labels = Params[day]['day_lfp_labs']
        unit_names = Params[day]['sorted_un']
        desired_unit_names = master_cell_list['date_'+day]
        
        for un in desired_unit_names:
            stat_mat[un] = np.zeros((2, 5, 25))
            stat_mat_rt[un] = np.zeros((2, 5, 25))
            # mc, b1, b2, .. x spk cnts x bins
      
        t_axis = np.linspace(-1.5, 1.0-.1, 25)
        go_ind = 0.6*day_spks.shape[1]

        for trl_type in ['mc', 'beta']:
            if trl_type == 'mc':
                ntypes = 1
                ix = [np.nonzero(lfp_labels<80)[0]]
                stat_mat_row = [0]

            elif trl_type == 'beta':
                ntypes = 4
                ix = []
                stat_mat_row = [1]
                for ii in range(84, 88):
                    tmp__ = np.nonzero(lfp_labels==ii)[0]
                    ix.append(tmp__)
                    stat_mat_row.append([1])


            for nt in range(ntypes):
                ix_ = ix[nt]
                stat_mat_row_trl = stat_mat_row[nt]

                cnt = 0
                for iu, un in enumerate(desired_unit_names):

                    iu_ix = np.nonzero(unit_names==un)[0]

                    if len(iu_ix) > 0:
                        #Aligned to go: 
                        spk_ind = day_spks[ix_, :, iu_ix]

                        for trl in range(len(ix_)):
                            for bn in range(25):
                                ztmp = int(spk_ind[trl, bn])
                                if ztmp>4:
                                    ztmp = 4
                                stat_mat[un][stat_mat_row_trl, ztmp, bn] += 1
                        
                        if plot:
                            if cnt < 36:
                                axi = ax[cnt/6, cnt%6]

                                clr = color_dict[trl_type, nt]
                                axi = plt_fill_between(axi, spk_ind, t_axis, clr)

                                if (cnt%6) == 0:
                                    axi.set_ylabel ('FR (Hz)')
                                if (cnt/6) == 5:
                                    axi.set_xlabel('Sec w.r.t Go')
                                axi.set_title('sig_'+un)

                        #Aligned to RT
                        spks_ind_rt_aligned = np.zeros_like(spk_ind)
                        rt_ind = rt_day[ix_]
                        for ri, rt_i in enumerate(rt_ind):
                            if rt_i > go_ind:
                                dt = rt_i - go_ind
                                spks_ind_rt_aligned[ri, :-dt] = spk_ind[ri, dt:]
                            elif rt_i < go_ind:
                                dt = go_ind - rt_i
                                spks_ind_rt_aligned[ri, dt:] = spk_ind[ri, :-dt]
                            elif rt_i == go_ind:
                                spks_ind_rt_aligned[ri, :] = spk_ind[ri, :]
                        
                        for trl in range(len(ix_)):
                            for bn in range(25):
                                ztmp = int(spks_ind_rt_aligned[trl, bn])
                                if ztmp>4:
                                    ztmp = 4
                                stat_mat_rt[un][stat_mat_row_trl, ztmp, bn] += 1


                        if plot:
                            if cnt < 36:
                                axi = ax2[cnt/6, cnt%6]
                                axi = plt_fill_between(axi, spks_ind_rt_aligned, t_axis, clr)
                                
                                if (cnt%6) == 0:
                                    axi.set_ylabel ('FR (Hz)')
                                if (cnt/6) == 5:
                                    axi.set_xlabel('Sec w.r.t RT')
                                axi.set_title('sig_'+un)

                        cnt += 1     

        chi_mat, chi_diff_mat, diff_mat, cnt = chi_sq(stat_mat)
        chi_mat2, chi_diff_mat2, diff_mat2, cnt2 = chi_sq(stat_mat_rt)

        
        chi_sq_master_mat[day, 'diff_mat'] = diff_mat/float(cnt)
        chi_sq_master_mat[day, 'chi_diff_mat'] = chi_diff_mat
        chi_sq_master_mat[day, 'chi_mat'] = chi_mat

        chi_sq_master_mat[day, 'diff_mat_rt'] = diff_mat2/float(cnt2)
        chi_sq_master_mat[day, 'chi_diff_mat_rt'] = chi_diff_mat2
        chi_sq_master_mat[day, 'chi_mat_rt'] = chi_mat2
        pickle.dump(chi_sq_master_mat, open('sub_'+day+'_chi_stats.pkl', 'wb'))

    pickle.dump(chi_sq_master_mat, open('chi_stats_psth_all_days.pkl', 'wb'))



def plt_fill_between(ax,  spk, t, color):
    ''' Spk  -- trials x bins '''

    sem = np.nanstd(spk, axis=0)/np.sqrt(spk.shape[0])
    mn = np.nanmean(spk, axis = 0)
    
    ax.plot(t, mn, color=color)
    ax.fill_between(t, mn-sem, mn+sem, color=color, 
        alpha = 0.5)
    ax.plot([0, 0], [np.min(mn), np.max(mn)], 'b-')
    return ax 


def chi_sq(stat_mat):
    ''' Gives chi_sq for each unit, bin, and mc-beta pair '''
    chi_mat = {}
    chi_diff_mat = {}
    diff_mat = np.zeros((25, ))
    cnt = 0
    for un in stat_mat.keys():
        if stat_mat[un].sum() > 0:
            cnt += 1
            chi_mat[un] = np.zeros((25, ))
            chi_diff_mat[un] = np.zeros((25, ))
            
            for b in range(25):
                f_mat = stat_mat[un][:, :, b]
                sub_f_mat = f_mat[[0, 1], :]
                exp_f_mat = np.zeros_like(sub_f_mat)

                total_sum = np.sum(sub_f_mat)
                row_mn = np.sum(sub_f_mat, axis=1)
                col_mn = np.sum(sub_f_mat, axis=0)

                dof = len(row_mn) + len(col_mn) - 2
                chi_2 = scipy.stats.chi2(dof)
                chi_crit = chi_2.ppf(0.95)

                for i in range(len(row_mn)):
                    for j in range(len(col_mn)):
                        exp_f_mat[i, j] = row_mn[i]*col_mn[j]/total_sum
                chi_mat[un][b] = np.nansum(((sub_f_mat-exp_f_mat)**2)/exp_f_mat)
                chi_diff_mat[un][b] = chi_mat[un][b] > chi_crit
            diff_mat = diff_mat +chi_diff_mat[un]
    return chi_mat, chi_diff_mat, diff_mat, cnt


def plot_psth_by_tsk(fname='chi_stats_psth_all_days.pkl'):
    dat = pickle.load(open(fname))
    k = np.vstack((dat.keys()))
    days = np.unique(k[:, 0])

    f, ax = plt.subplots(nrows = 2)
    t_axis = np.linspace(-1.5, .9, 25)+0.05

    for d in days:
        ax[0].plot(t_axis, dat[d, 'diff_mat'], label=d)
        ax[1].plot(t_axis, dat[d, 'diff_mat_rt'])
    ax[0].legend()
    for j, (i, nm) in enumerate(zip([0, 1], ['Go Cue', 'Rxn Time'])):
        ax[i].set_ylim([0., 1.])
        ax[i].set_ylabel(' Perc. Neurons')
        ax[i].set_xlabel('Time w.r.t '+nm+ ' (secs)')
        ax[i].set_title('Perc. Neurons Sig. Diff. in MC and Beta Mod. Trials Aligned to '+nm)
    plt.tight_layout()



if __name__ == '__main__':
    plot_psth_by_task_by_ts(test=False)









                

