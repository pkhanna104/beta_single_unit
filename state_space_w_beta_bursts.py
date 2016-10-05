import parse_spks
import tables 
import numpy as np
import scipy.signal
import collections 
import matplotlib.pyplot as plt
import pickle
import datetime
from load_files import master_save_directory
import un2key
from scipy import ndimage
import math

# hdf_bcd = tables.openFile('/Volumes/TimeMachineBackups/v2_Beta_pap_rev_data/2016-05-09pap_rev_grom_behav_t1lfp_mod_mc_reach_out.h5')
# ix = hdf_bcd.root.behav[:]['task_entry']
# ix2 = np.array([i for i,j in enumerate(ix) if j[:6]=='022715'])
# kin_sig = hdf_bcd.root.kin[ix2]['kin_sig']

master_save_directory2 = '/Volumes/TimeMachineBackups/v2_Beta_pap_rev_data/'
master_beta_filt = [25., 40.]


def get_beta_bursts(keep_dict, days, blocks, mc_indicator, perc_beta=60, bp_filt=master_beta_filt, save=False, fname=None, 
    min_beta_burst_len=100, animal='grom'):

    ''' Method to yield a number of relevant LFP & spike metrics aligned to 
    relevant task points: 

    keep_dict : dict with cells for each day (keys are '0227155' format)

    perc_beta: percentile to use as threshold for calling a bin a 'beta event' bin

    '''

    spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, bef, aft, go_times_dict = parse_spks.parse_spks_and_LFP(keep_dict,
        days, blocks, mc_indicator, animal=animal)

    beta_dict = {}
    beta_dict_cont = {}
    for i_d, d in enumerate(days):
        for i_b, b in enumerate(blocks[i_d]):
            #Filter LFPs between 20 - 45 Hz: 
            nyq = 0.5* 1000
            bw_b, bw_a = scipy.signal.butter(5, [bp_filt[0]/nyq, bp_filt[1]/nyq], btype='band')
            data_filt = scipy.signal.filtfilt(bw_b, bw_a, lfp_dict[b, d], axis=1)

            #Get amplitude of filtered signal: 
            try:
                sig = np.abs(scipy.signal.hilbert(data_filt, N=None, axis=1))
            except:
                print b, d, 'error in state_space_w_beta_bursts, get_beta_bursts function'
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
    if save:
        save_dict = dict(spk_dict=spk_dict, lfp_dict=lfp_dict, lfp_lab=lfp_lab, blocks=blocks,
            days=days, rt_dict=rt_dict, beta_dict=beta_dict, bef=bef, aft=aft, bp_filt=bp_filt)
        if fname is None:
            raise
        
        else:
            tdy = datetime.date.today()
            fname = tdy.isoformat() + '_'+fname
            pickle.dump(save_dict, open(master_save_directory2+'/'+fname, 'wb'))

    return spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_dict_cont, bef, aft, go_times_dict

def bin_spks_and_beta(keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, beta_dict, beta_dict_cont, bef, aft, 
    smooth=-1, beta_amp_smooth=-1, binsize=20):
    '''
    Analyze decoding on a day by day basis

    smooth = -1 if no smoothing, +20 (e.g.) for gaussian smoothing w/ std = 20 ms

    beta_amp_smooth = -1 for no smoothing, +20 (e.g.) for gaussian smoothing w/ std = 20 ms for continuous beta
        signal only
    binsize: size of bin for binned spikes

    save: Flag to save resultant data into a pkl file for later access. Will save to designated directory
        defined at top of file (master_save_directory) 


    '''

    B_bin = dict()
    B_C_bin = dict()
    S_bin = dict()
    Params = dict()

    for i_d, d in enumerate(days):
        Params[d] = dict()

        #Get units for the day
        sorted_un = np.sort(keep_dict[d])
        Params[d]['sorted_un'] = sorted_un
        n_units = len(sorted_un)

        spk_arr = np.nan

        #Convert to spike array instead of spk dict (will be trials x time x units)
        for b in blocks[i_d]:
            Params[d][b] = dict()
        
            mini_spk_arr = []

            for un_ in sorted_un:
                #Convert: 
                un = un2key.convert(un_)
                try:
                    mini_spk_arr.append(spk_dict[b, d][un])
                except:
                    print d, b, un, sorted_un
            try:
                spk_arr = np.vstack((spk_arr, np.dstack((mini_spk_arr))))
                beta_arr = np.vstack((beta_arr, beta_dict[b, d]))
                beta_c_arr = np.vstack((beta_c_arr, beta_dict_cont[b, d]))
                beta_p_arr = np.vstack((beta_p_arr, beta_dict_cont[b, d, 'phz']))

            except:
                spk_arr = np.dstack((mini_spk_arr))
                beta_arr = beta_dict[b, d]
                beta_c_arr = beta_dict_cont[b, d]
                beta_p_arr = beta_dict_cont[b, d, 'phz']
                
            Params[d][b]['n_trials'] = beta_dict[b, d].shape[0]

        #Compile lfp_labels:
        day_lfp_labs = np.hstack(([lfp_lab[b, d] for b in blocks[i_d]]))
        Params[d]['day_lfp_labs'] = day_lfp_labs

        n_trials = len(day_lfp_labs)
        n_trials_beta = beta_arr.shape[0]

        #Num trials from lfp label = num trials from spike arr and beta arr: 
        try:
            assert n_trials == spk_arr.shape[0]
            assert n_trials == beta_arr.shape[0]
            print n_trials, spk_arr.shape, beta_arr.shape, b, d
        except:
            print n_trials, spk_arr.shape, beta_arr.shape, b, d
            raise Exception

        #Optionally smooth spikes and non-optionally bin them: 
        day_spks = []
        if smooth > -1:
            window = scipy.signal.gaussian(101, std=smooth)
            window = window/np.sum(window)

            for trl in range(len(day_lfp_labs)):
                for iu in range(n_units):
                    spk_arr[trl, :, iu] = np.convolve(window, spk_arr[trl, :, iu], mode='same')

        else:
            print 'No spike smoothing :)'

        Params[d]['spk_smooth'] = smooth
        
        #Optionally smooth continuous beta signal 
        if beta_amp_smooth > -1:
            print 'Beta Smoothing! :) '
            window = scipy.signal.gaussian(101, std=beta_amp_smooth)
            window = window/np.sum(window)

            for trl in range(len(day_lfp_labs)):
                beta_c_arr[trl, :] = np.convolve(window, beta_c_arr[trl, :], mode='same')
        else:
            print 'No Beta smoothing :)'

        Params[d]['beta_c_smooth'] = beta_amp_smooth

        #Bin spikes and beta data:
        sz = int((aft+bef)*1000/float(binsize))

        spks_bin = np.zeros((n_trials, sz, n_units ))
        beta_bin = np.zeros((n_trials, sz))
        beta_c_bin = np.zeros((n_trials, sz))
        beta_p_bin = np.zeros((n_trials, sz))

        for t in range(sz):
            start = t*binsize
            ix_ = np.ix_(range(n_trials), range(start, start+binsize), range(n_units))
            spks_bin[:, t, :] = np.mean(spk_arr[ix_], axis=1)


            bix_ = np.ix_(range(n_trials), range(start, start+binsize))
            beta_bin[:, t] = np.around(np.mean(beta_arr[bix_], axis=1)).astype(int)
            beta_c_bin[:, t] = np.mean(beta_c_arr[bix_], axis=1)

            #Circular mean
            tmp = beta_p_arr[bix_]
            tmp_x = np.mean(np.cos(tmp), axis=1)
            tmp_y = np.mean(np.sin(tmp), axis=1)
            tmp_mn_ang = np.array([math.atan2(y, x) for i, (x, y) in enumerate(zip(tmp_x, tmp_y))])
            beta_p_bin[:, t] = tmp_mn_ang

        B_bin[d] = beta_bin
        B_C_bin[d] = beta_c_bin
        B_C_bin[d, 'phz'] = beta_p_bin
        S_bin[d] = spks_bin

    return B_bin, B_C_bin, S_bin, Params

def get_ov(X_, X2_, B_):
    #Ok, now let's fit 3 state space models: 1) 1500-2000, 2) beta burst in 1000-1500, 3) nonbeta bursts in 1000-1500
    U1 = fit_FA(X_, B_, beta_inds=[1.])
    U2 = fit_FA(X_, B_, beta_inds=[0.])
    U3 = fit_FA(X2_, B2_, beta_inds=[])

    xx = np.hstack((X_, X2_))
    bb = np.hstack((B_, B2_))
    u1 = ssbb.fit_FA(xx, bb, beta_inds=[1.])
    u2 = ssbb.fit_FA(xx, bb, beta_inds=[0.])

    #Outcome: 
    #Overlap between beta and non-beta in hold period is GREATER than 
    #beta-hold and move, non-beta-hold and move. Beta and non-beta overlap is ~88% or 93% -- visually exactly the same
    B_full = np.tile(B_[:, :, np.newaxis], [1, 1, 28])
    B_off = -1*(B_full-1)
    dfr = np.mean(np.mean(X_*B_full, axis=1), axis=0) - (np.mean(np.mean(X_*B_off, axis=1), axis=0))
    #FR is greater for B_off compared to B_on (dfr < 0 for ALL units)

def get_overlap2(U_B, U_A, ndims=2):
    v, s, vt = np.linalg.svd(U_B*U_B.T)
    s_cum = np.cumsum(s)/np.sum(s)
    red_s = np.zeros((v.shape[1], ))

    #Find shared space that occupies > 90% of var:
    ix = np.nonzero(s_cum>0.9)[0]
    nf = ix[0] + 1
    #fa_dict[te, 'main_nf'] = nf

    red_s[:nf] = 1
    Pb = v*np.diag(red_s)*vt #orthonormal cov (nfact_b x nfact_b)

    vv, ss, vvt = np.linalg.svd(U_A*U_A.T)
    ss_cum = np.cumsum(ss)/np.sum(ss)
    ix = np.nonzero(ss_cum>0.9)[0]
    nnf = ix[0] + 1

    red_s = np.zeros((vv.shape[1], ))
    red_s[:nnf] = ss[:nnf]
    A_shar = vv*np.diag(red_s)*vvt
    proj_A_B = Pb*A_shar*Pb.T
    return np.trace(proj_A_B)/float(np.trace(A_shar))


import sklearn.decomposition as sd
def fit_FA(X_full, B_full, beta_inds=[0, 1]):
    if len(beta_inds)>0:
        X_sub = []
        for i in range(B_full.shape[0]):
            for j in range(B_full.shape[1]):
                if B_full[i, j] in beta_inds:
                    X_sub.append(X_full[i, j, :])

        X = np.vstack((X_sub))
    else:
        X = X_full.reshape(X_full.shape[0]*X_full.shape[1], X_full.shape[2])

    LL = []
    for i in range(28):
        FA = sd.FactorAnalysis(n_components=i+1)
        FA.fit(X)
        LL.append(np.sum(FA.score(X)))

    #Orthogonalized Factors: 
    ndim = np.argmax(LL)+1
    FA = sd.FactorAnalysis(n_components=ndim)
    FA.fit(X)
    U = np.mat(FA.components_).T
    A = U*U.T
    u, s, v = np.linalg.svd(A)
    s_red = np.zeros_like(s)
    s_hd = np.zeros_like(s)

    #Plot s^2: 
    plt.plot(np.cumsum(s**2)/(float(np.sum(s**2))), '.')
    plt.show()

    ix = np.nonzero(np.cumsum(s**2)/float(np.sum(s**2))>.90)[0]
    if len(ix) > 0:
        n_dim_main_shared = ix[0]+1
    else:
        n_dim_main_shared = len(s)

    n_dim_main_shared = np.max([n_dim_main_shared, 3])

    #Computer Factor space: 
    s_red[:n_dim_main_shared] = s[:n_dim_main_shared]
    s_hd[n_dim_main_shared:] = s[n_dim_main_shared:]
    main_shared_A = u*np.diag(s_red)*v
    hd_shared_A = u*np.diag(s_hd)*v
    Psi = np.diag(FA.noise_variance_)
    main_shared_B = np.linalg.inv(main_shared_A + hd_shared_A + Psi)
    main_sharL = main_shared_A*main_shared_B

    return U








