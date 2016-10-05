
import scipy.stats

import numpy as np
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)

import kin_plots_and_stats as kps
import scipy.stats as scistats
import matplotlib.pyplot as plt

#MC_rt = fcns.print_stats_from_BMI3D(d, lfp_targ[:,1], k_rxn, mc_lab=66)

def get_cmap():
    return ['lightseagreen', 'dodgerblue', 'gold', 'orangered', 'k']

def print_stats_from_BMI3D(d, lfp_targ, k_rxn, mc_lab=64, lfp_lab=[84, 85, 86, 87],print_on=True):
    MC_rt = map_d_to_MC_rt(d, mc_lab = mc_lab, lfp_lab=lfp_lab)
    mc_lab_arr = np.array([mc_lab]*len(k_rxn))
    lfp_lab_arr = (lfp_targ/4.875)+1+84

    #Test for normality
    KS_res = kps.test_for_norm(lfp_lab_arr, mc_lab_arr, MC_rt,mc_set=set([mc_lab]))

    #Kruskal Wallis and Mann-Whitney Tests: 
    KW_res, MW_res, n = kps.test_with_KruskalWallis(mc_lab_arr, MC_rt, mc_set=set([mc_lab]), lfp_set=lfp_lab)

    #Anova: 
    AN_res = kps.test_ANOVA(mc_lab_arr,MC_rt,mc_set=([mc_lab]))

    if print_on:
        for k,ke in enumerate(KS_res.keys()):
            print ke, KS_res[ke]

        for k,ke in enumerate(MW_res.keys()):
            a = MW_res[ke]
            try:
                print ke, a[1]*n
            except:
                print ke, MW_res[ke]
    return MC_rt

def plt_hist_money(lfp_targ, k_rxn, bins=None,plt_on=True):
    d = dict()
    for i,l in enumerate(np.sort(np.unique(lfp_targ))):
        l_ind = np.nonzero(lfp_targ==l)[0]
        print 'length of lfp_targ: ', l, ': ', len(l_ind)
        x, n = cum_dist(k_rxn[l_ind],bins=bins)
        if plt_on:
            plt.plot(x,n,color=cmap[i],linewidth=4,label='BBM Target '+str(i))
        d[i] = k_rxn[l_ind]
    if plt_on:
        plt.xlabel('Movement Onset Time (secs)')
        plt.ylabel('Cumulative Frequency')
        plt.legend()
    h, p = scistats.mstats.kruskalwallis(d[0],d[1],d[2],d[3])
    return p, d

def cum_dist(array,bins=None):
    if bins is None:
        n, x = np.histogram(array,bins=np.linspace(np.percentile(array,20), np.percentile(array,80), 10))
    else:
        n, x = np.histogram(array,bins=bins)
    return x[1:], np.cumsum(n)/float(np.sum(n))

def get_common_trials(beh, kin):
        #get common TEs: 
    a = set(list(np.unique(kin.root.trl_bhv[:]['task_entry'])))
    b = set(list(np.unique(beh.root.trl_bhv[:]['task_entry'])))
    a.difference(b)

    b_ind = []
    k_ind = []

    for it, te in enumerate(a):
        st_b = np.array([[i, beh.root.trl_bhv[i]['start_time'],s] for i, s in enumerate(beh.root.trl_bhv[:]['task_entry']) if s==te])
        st_k = np.array([[i, kin.root.trl_bhv[i]['start_time'],s] for i, s in enumerate(kin.root.trl_bhv[:]['task_entry']) if s==te])

        if len(st_b)==len(st_k):
            b_ind.extend(list(st_b[:,0]))
            k_ind.extend(list(st_k[:,0]))
        else:
            for i_s, st in enumerate(st_b[:,1]):
                if st in list(st_k[:,1]):
                    b_ind.extend([st_b[i_s,0]])

                    ik = np.nonzero(st_k[:,1]==st)[0][0]
                    k_ind.extend([st_k[ik, 0]])
    for i, b in enumerate(beh.root.trl_bhv[np.array(b_ind).astype(int)]['start_time']):
        if b!=kin.root.trl_bhv[int(k_ind[i])]['start_time']:
            print i
            
    return np.array(b_ind).astype(int), np.array(k_ind).astype(int)

def map_d_to_MC_rt(d, mc_lab=64,lfp_lab = [84, 85, 86, 87]):
    MC_rt= dict()
    for il, lf in enumerate(lfp_lab):
        MC_rt[(mc_lab, lf)] = d[il]
    return MC_rt


def return_cmap():
    return ['lightseagreen','dodgerblue','gold','orangered','k']

def detrend_1_f(X, freq, detrend_ax=2):
    f_vs_pxx = np.mean(np.reshape(X, [X.shape[0]*X.shape[1], X.shape[2]]), axis=0)
    p_func = np.poly1d(np.polyfit(freq, f_vs_pxx, 4))
    f_norm = p_func(freq)
    F_Norm = np.tile(f_norm, [X.shape[0], X.shape[1], 1])
    return X - F_Norm

def zscore_along_axis(X, zsc_ax=[0, 1]):
    #Get mean and std of X along zsc_ax: 
    dim_prod = np.product(X.shape)
    if len(zsc_ax)==2:
        dim_mn = np.product((X.shape[zsc_ax[0]], X.shape[zsc_ax[1]]))
        X_resh = np.reshape(X, [dim_mn, dim_prod/dim_mn])
        mn = np.mean(X_resh, axis=0)
        std = np.std(X_resh, axis=0)
        X_dmn = X - mn
        X_zsc = np.divide(X_dmn, std)
    elif len(zsc_ax) == 1:
        mn = np.mean(X)
        std = np.std(X)
        X_dmn = X - mn
        X_zsc = np.divide(X_dmn, std)
    else:
        print 'more/less than two dims not implmented yet /utils/fcns.py'
    return X_zsc

def plot_mean_and_sem(x ,array, ax, color='b', array_axis=1,label='0',
    log_y=False, make_min_zero=[False,False]):
    
    mean = array.mean(axis=array_axis)
    sem_plus = mean + scipy.stats.sem(array, axis=array_axis)
    sem_minus = mean - scipy.stats.sem(array, axis=array_axis)
    
    if make_min_zero[0] is not False:
        bi, bv = get_in_range(x,make_min_zero[1])
        add = np.min(mean[bi])
    else:
        add = 0

    ax.fill_between(x, sem_plus-add, sem_minus-add, color=color, alpha=0.5)
    ax.plot(x,mean-add, '-',color=color,label=label)
    if log_y:
    	ax.set_yscale('log')

def get_in_range(array, range,inclusive_both_sides=False):
    if inclusive_both_sides:
        idx = np.array([i for i, ind in enumerate(array) if range[0]<= ind and ind<=range[1]])
        val = np.array([ind for i, ind in enumerate(array) if range[0]<= ind and ind<=range[1]])
    else:
	   idx = np.array([i for i, ind in enumerate(array) if range[0]<= ind and ind<range[1]])
	   val = np.array([ind for i, ind in enumerate(array) if range[0]<= ind and ind<range[1]])
    return idx, val

def test_for_normality(array_of_arrays):
    ks_test_result = np.zeros((len(array_of_arrays),))
    for i, a in enumerate(array_of_arrays):
        a_ar = np.squeeze(np.array(a))
        zsc = (a_ar - np.mean(a_ar))/np.std(a_ar)
        ks_test_result[i] = scipy.stats.kstest(zsc,'norm')[1]
        print scipy.stats.kstest(zsc,'norm')
    return ks_test_result

def test_ANOVA_and_T_HSD(array_of_arrays):
    ANOVA_f, ANOVA_p = scipy.stats.f_oneway(*array_of_arrays)

    if ANOVA_p < 0.05:
        x = []
        label = []
        for grp_num in range(len(array_of_arrays)):
            x.extend(array_of_arrays[grp_num])
            label.extend([grp_num]*len(array_of_arrays[grp_num]))

        tukey_HSD_result = pairwise_tukeyhsd(np.array(x), np.array(label))
        return ANOVA_p, tukey_HSD_result
    else:
        return ANOVA_p, False

def test_KW_and_MW(array_of_arrays):
    KW_stat, KW_p = scipy.stats.mstats.kruskalwallis(*array_of_arrays)

    if KW_p < 0.05:
        MW_result = np.zeros((len(array_of_arrays), len(array_of_arrays)))
        for grp_num1 in range(len(array_of_arrays)): 
            x = array_of_arrays[grp_num1]
            for grp_num2 in range(grp_num1+1,len(array_of_arrays)):  
                y = array_of_arrays[grp_num2]
                MW_result[grp_num1,grp_num2] = scipy.stats.mannwhitneyu(x,y)[1]
        return KW_p, MW_result
    return KW_p, False

def get_common_trials(beh, kin , kin_table_name = None):
        #get common indices between kinematics pytables and behavior pytable: 
    a = set(list(np.unique(kin.root.trl_bhv[:]['task_entry'])))
    b = set(list(np.unique(beh.root.trl_bhv[:]['task_entry'])))
    a.difference(b)

    b_ind = []
    k_ind = []
    
    for it, te in enumerate(a):
        st_b = np.array([[i, beh.root.trl_bhv[i]['start_time'],s] for i, s in enumerate(beh.root.trl_bhv[:]['task_entry']) if s==te])
        st_k = np.array([[i, kin.root.trl_bhv[i]['start_time'],s] for i, s in enumerate(kin.root.trl_bhv[:]['task_entry']) if s==te])

        if len(st_b)==len(st_k):
            b_ind.extend(list(st_b[:,0]))
            k_ind.extend(list(st_k[:,0]))
        else:
            for i_s, st in enumerate(st_b[:,1]):
                if st in list(st_k[:,1]):
                    b_ind.extend([st_b[i_s,0]])

                    ik = np.nonzero(st_k[:,1]==st)[0][0]
                    k_ind.extend([st_k[ik, 0]])

    for i, b in enumerate(beh.root.trl_bhv[np.array(b_ind).astype(int)]['start_time']):
        if b!=kin.root.trl_bhv[int(k_ind[i])]['start_time']:
            print i
    return np.array(b_ind).astype(int), np.array(k_ind).astype(int)

def cuzicks_test(list_of_rts, ranking_to_test=[1, 2, 3, 4], twosided=True):
    #Do Cuzick's test for group (rank) trend: http://www.stata.com/manuals13/rnptrend.pdf
    
    print 'order: ', ranking_to_test
    args = []
    args1 = []
    for lfp_i, lfp in enumerate(ranking_to_test):
        x = list(list_of_rts[lfp_i])
        y = list(np.zeros((len(x)))+lfp_i)

        args.append(y)
        args1.append(x)

    Y = np.hstack((args)) 
    X = np.hstack((args1)) 

    #Rankings of X
    N = len(X)
    Rnk = scipy.stats.rankdata(X)
    R = np.zeros(( len(ranking_to_test), ))
    L = np.zeros(( len(ranking_to_test), ))
    L2 = np.zeros(( len(ranking_to_test), ))
    Ni = np.zeros(( len(ranking_to_test), ))
    T = np.zeros(( len(ranking_to_test), ))

    for li, lfp in enumerate(ranking_to_test):
        
        #Find labels equal to this label
        ix = np.nonzero(Y==lfp)[0]

        #Get rank fo data: 
        R[li] = np.sum(Rnk[ix])

        #Get wts
        L[li] = lfp*len(ix)
        L2[li] = lfp*lfp*len(ix)
        T[li] = lfp*R[li]
        Ni[li] = len(ix)

    print 'T', np.sum(T)
    e_t = 0.5*(N+1)*np.sum(L)
    var_t = (N*np.sum(L2) - (np.sum(L)**2))*((N+1)/12)
    se_t = np.sqrt( var_t )
    z = ( np.sum(T) - e_t ) / se_t
    print 'z', z
    if twosided:
        if z > 0:
            p = (1 - scipy.stats.norm.cdf(z))*2 #Two tailed test
        else:
            p = (scipy.stats.norm.cdf(z))*2 #Two-tailed test
    else:
        p = (1 - scipy.stats.norm.cdf(z))
    return p
