import parse_spks
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import tables
import load_files
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

import un2key

NUM_COLORS = 15
cm = get_cmap('gist_heat')
cmap = []
for i in range(NUM_COLORS):
    cmap.append(cm(1.*i/NUM_COLORS))  # colors

from load_files import master_save_directory as master_path
nrows = 5
test_days = ['022315']
master_days = ['022315','022415', '022515','022715', '022815', '030215']
test_blocks = ['bd']

#Temporarily removing '022315a' for reasons: 
master_blocks = ['bdefghijkl', 'abdefghij', 'acdefhi','acdeg','adefghi','acdef'] 
#REMOVED 022715f because didn't understand AD33_ts (.007, 410, 422) ? and 022715h because no rewards and 022715b for same reason
#Removed 030215g because errors ... ? 
test_mc_indicator = ['10']
master_mc_indicator =  ['1000000000', '110000000', '1000000','10000','1000000','10000']

# Master cell list for sorted files: 
master_cell_list = dict(
    date_022315=np.hstack([['2a','5a','5b','21a','33a','34a','42a','68a','69a','71a','72a','75a','75b',
    '77a','78a','79a','81a','86a','88b','89a','90a','98a','98b','122a','128a','128b'],  #Good cell list
    ['13a','71a','76a','80a','88a','101a'],  #Great cell list
    ['1a','4b','6a','20a','26a','28a','29a','41a','46a','47a','56a','57a','58a','58b',
        '60a','61a','66a','70a','81b','83a','85a','114a']]), #Multiunit list

    date_022415 = np.hstack([['1a', '4a', '5a', '13b', '20a', '20b', '21b', '26a', '29a', '33a', '34a',
        '35a', '38a', '42a', '51b', '56a', '58a', '68a', '68b', '71a', '73a', '75a', '75b', '76a', 
        '79a', '81a', '81b', '83a', '88a', '89a', '90a', '98a', '98b', '98c', '99a', '101a', '103a',
        '112b', '113a', '128a', '128b'], #Good cell list
        ['28a', '32c', '34a', '47a', '51a', '72a', '77a', '77b', '78a'], #Great cell list
        ['2a', '3a', '3b', '4b', '5b', '6a', '12a', '13a', '13c', '16a', '18a', '24a', '27a', 
        '30a', '32a', '32b', '40b', '41a', '46a', '49a', '57a', '66a', '69a', '69b', '73b', '84a', '84b',
        '85a', '86a', '87a', '97a', '112a', '117a', '122a']]), #multiunit list

    date_022515 = np.hstack([['1a', '3a', '3b', '4a', '5a', '5b', '6a', '16a', '20a', '20b', '21b',
        '24b', '26a', '28a', '28b', '29a', '29b', '32a', '33a', '34a', '51a', '51b', '58a', '68a', 
        '69a', '69b', '71a', '72a', '75a', '75b', '76a', '77a', '78a', '79a', '79b', '80a', '81b',
        '82a', '83a', '85a', '88a', '88b', '89a', '90a', '94a', '98a', '98b', '98c', '103a', '103b', 
        '103c', '114a', '114b', '128a'], #Good cells
        ['101a'], #Great cell list
        ['2a', '4b', '11a', '12a', '13a', '14a', '18a', '21a', '24a', '30a', '31a', '32b', '35a', '38a', 
        '40a', '41a', '42a', '44a', '46a', '47a', '50a', '56a', '57a', '62a', '66a', '68b', '73a', 
        '81a', '84a', '87a', '91a', '94b', '96a', '97a', '99a', '105a', '113a', '122a', '122b', 
        '124a', '128b']]), #multiunit list

    date_022715=np.hstack([['004b', '005a', '013a', '020a', '020b', '020c', '021a', '021b', 
    '025a', '026a', '029a', '033a', '034a', '035b', '042a', '051a','056a', '071a', 
    '072a', '075a', '075b', '077a', '078a', '079a', '079b', '079c', '080a', '081a',
    '081b', '083a', '088a', '089a', '099a', '099b', '099c', '102a', '104a', '111a'], #Good cell list
    ['004a', '003b', '005a', '032a', '034a', '056a', '071a', '072a', 
    '076a', '078a', '080a', '085a', '111a']]), #GREAT cell list

    date_022815 = np.hstack([['1a','2a','3a','3b','4a','5a','5b','6a','16a','20a','20b','20c',
    '21a','33b','56b','58a','71a','72a','76a','77a','78a','79a','80a','83a','84a',
    '85a','88a','88b','90a','94a','97a','98a','98b','98c','112a'], #Good cell list
    ['13a','13b','34a','75a','81a','81b','101a'], #Great cell list
    ['4b','12','21b','24a','26a','28a','28b','29a','29b','31a','32a','32b','33a',
    '33c','34b','40b','41a','42b','46a','47a','50a','56a','57a','66a','67a','68a',
    '73a','79b','82a','86b','89a','94b','99a','103a','110a', '111a','113a','123a','128a']]), #M.U. list

    date_030215 = np.hstack([['3a','3b','4a','5a','13a','13b','20a','20b','20c','25a','28a','28b','32a',
    '33b','41a','41c','51a','56a','72a','76a','78a','79a','80a','80b','80c','81a', '81b',
    '83a','84a','85a','85b','86a','97a'], #Good cell list
    ['32b','33a','33c','34a','71a','75a','98a','98b'], #Great cell list
    ['1a','2a','4b','5b','6a','24a','24b','26a','27a','30a','31a','34b','38a','42a','49a','50a',
    '57a','58a','66a','68a','69a','69b','71b','77a','79b','86b','87a','88a','88b','89a', '90a',
    '94a','96a','99a','110a','114a','128a']]), #M.U. List
    )


def get_cells(plot=True, days=None, blocks=None, mc_indicator=None, test=False, animal='grom'):
    ''' Method to return cells after plotting their waveforms across all 
    blocks to ensure they are the same'''

    if days is None and blocks is None:
        if test:
            days = test_days
            blocks = test_blocks
            mc_indicator = test_mc_indicator
        else:

            days = master_days
            blocks = master_blocks
            mc_indicator = master_mc_indicator

    elif blocks is None:
        blks = []
        mc_ii = []
        for d in days:
            iid = np.nonzero(np.array(master_days) == d)[0]
            blks.append(master_blocks[iid])
            mc_ii.append(master_mc_indicator[iid])
        blocks = blks
        mc_indicator = mc_ii

    keep_cell_dict = dict()

    for i_d, d in enumerate(days):
        incomplete_list = []
        good_cell_list = list(np.unique(master_cell_list['date_'+str(d)]))

        #Init graphs for 1) WFs, 2) Mean FR 
        if plot:
            f, ax= plt.subplots(nrows = nrows, ncols = int(np.ceil(len(good_cell_list)/float(nrows))))
            f2, ax2 = plt.subplots(nrows=nrows,ncols = int(np.ceil(len(good_cell_list)/float(nrows))))


        for ib, b in enumerate(blocks[i_d]):
            spk = load_files.load(b, d)
            if spk is not None:
                for ig, gc in enumerate(good_cell_list):
                    un2 = un2key.convert(gc)
                    key = un2+'_wf'

                    if plot:
                        axi = ax[ig%nrows, ig/nrows]
                        axi2 = ax2[ig%nrows, ig/nrows]

                    if key in spk.keys():
                        secs = spk['AD74'].shape[0]/1000.

                        if plot:
                            mn = np.mean(spk[key], axis=0)
                            axi.plot(mn, color=cmap[ib])
                            sem = np.std(spk[key], axis=0)/float(np.sqrt(spk[key].shape[0]))
                            axi.fill_between(np.arange(32), mn-sem, mn+sem, color=cmap[ib], alpha=0.5)
                            axi.set_title(key)
                            axi2.bar(ib, spk[key].shape[0]/float(secs), color=cmap[ib])
                            axi2.set_title(key)

                        if spk[key].shape[0]/float(secs) < 0.5:
                            incomplete_list.append(gc)
                    else:
                        print 'No ', key, ' in block ', b, ' , date ', d
                        incomplete_list.append(gc)

        good_set = set(good_cell_list)
        incomp_set = set(incomplete_list)
        keep_ = np.array(list(good_set.difference(incomp_set)))
        keep_cell_dict[d] = keep_
    return keep_cell_dict, days, blocks, mc_indicator

def plot_wfs_by_ix(ix_beta, ix_kin, day, keep_dict, Params):
    cell_list = Params[day]['sorted_un']

    #Both beta and kin: 
    ix_both = set(ix_beta).intersection(set(ix_kin))
    ix_none = set(range(len(cell_list))).difference(ix_beta)
    ix_none = ix_none.difference(ix_kin)

    mast_ix = [ix_beta, ix_kin, list(ix_both), list(ix_none)]
    mast_nm = ['beta', 'kin', 'both', 'none']
    mast_c = ['red','cyan','black','green']
    #Spk: 
    day_ix = np.array([i for i, j in enumerate(master_days) if j == day])
    mc_day = master_mc_indicator[day_ix]
    ix = mc_day.find('1')
    block = master_blocks[day_ix][ix]

    #Load filename: 
    spk_file = load_files.load(block, day)

    p2p = {}
    width = {}
    std_ = {}
    n_ts = {}


    f2, ax2 = plt.subplots(nrows=2, ncols=3)
    ln = []
    for i, ixs in enumerate(mast_ix):
        #Plot Beta ix: 
        #f, ax = plt.subplots(nrows=nrows, ncols = int(np.ceil(len(ixs)/float(nrows))))
        p2p[mast_nm[i]] = []
        width[mast_nm[i]] = []
        std_[mast_nm[i]] = []
        n_ts[mast_nm[i]] = []
        
        for iu, un in enumerate(cell_list[ixs]):
            un2 = un2key.convert(un)
            key = un2+'_wf'
            #axi = ax[iu%5, iu/5]

            mn = np.mean(spk_file[key], axis=0)
            std = np.std(spk_file[key], axis=0)
            #axi.fill_between(np.arange(32), mn-std, mn+std, color=cmap[0], alpha=0.5)
            #axi.set_title(key)
            #axi.set_ylim([-.1, .1])

            #Metrics: 
            ix0 = np.argmin(mn)
            ix1 = np.argmax(mn[ix0:])

            p2p[mast_nm[i]].append(mn[ix1+ix0] - mn[ix0])
            width[mast_nm[i]].append(ix1-ix0)
            std_[mast_nm[i]].append(np.mean(std))
            n_ts[mast_nm[i]].append(spk_file[key].shape[0])

        #axi.set_title(key + mast_nm[i])
        #plt.tight_layout()
        
        ln.append(ax2[0, 0].plot(p2p[mast_nm[i]], width[mast_nm[i]], '.', color=mast_c[i], markersize=25))
        ax2[0, 0].set_xlabel('P2P')
        ax2[0, 0].set_ylabel('width')

        ax2[0, 1].plot(width[mast_nm[i]], std_[mast_nm[i]], '.', color=mast_c[i], markersize=25)
        ax2[0, 1].set_xlabel('Width')
        ax2[0, 1].set_ylabel('Std')

        ax2[0, 2].plot(std_[mast_nm[i]], p2p[mast_nm[i]], '.', color=mast_c[i], markersize=25)
        ax2[0, 0].set_xlabel('Std')
        ax2[0, 0].set_ylabel('P2P')

        ax2[1, 0].plot(n_ts[mast_nm[i]], p2p[mast_nm[i]], '.', color=mast_c[i], markersize=25)
        ax2[1, 0].set_xlabel('# Time Stamps in MC')
        ax2[1, 0].set_ylabel('P2P')

        ax2[1, 1].plot(n_ts[mast_nm[i]], width[mast_nm[i]], '.', color=mast_c[i], markersize=25)
        ax2[1, 1].set_xlabel('# Time Stamps in MC')
        ax2[1, 1].set_ylabel('Width')

        ax2[1, 2].plot(n_ts[mast_nm[i]], std_[mast_nm[i]], '.', color=mast_c[i], markersize=25)
        ax2[1, 2].set_xlabel('# Time Stamps in MC')
        ax2[1, 2].set_ylabel('Std')



        #plt.tight_layout()




    #Plot kin ix:

    #Plot none:

    #Plot both:




import collections
import scipy.signal

def red_dim(keep_):
    ''' Deprecated '''
    d, lfp_lab, lfp_d, blocks, rt = parse_spks.parse_spks_and_LFP(cell_list=keep_)

    # Bin and smooth spikes: 
    OrdD = collections.OrderedDict()
    X = []
    X_smooth = []
    window = scipy.signal.gaussian(101, std=20)

    for k in d.keys():
        OrdD[k] = d[k]
        X.append(d[k])
        tmp = []
        for t in range(d[k].shape[0]):
            tmp.append(np.convolve(window, d[k][t,:],mode='same')/np.sum(window))
        X_smooth.append(np.vstack((tmp)))

    Y = []
    ix = []
    cnt = 0
    for b in lfp_lab.keys():
        if b=='a':
            print 'skip a'
        else:
            Y.append(lfp_lab[b])
            ix.append(np.arange(cnt, len(lfp_lab[b])+cnt))
        
        cnt += len(lfp_lab[b])

    XX = np.dstack((X_smooth))

    YY = np.hstack((Y))
    IX = np.hstack((ix))

    bin_size = 10
    #Post Go:
    time_ix = [1500, 2500]

    #Pre Go:
    time_ix2 = [1000, 1500]

    sz = (time_ix[1]-time_ix[0])/50
    sz2 = (time_ix2[1]-time_ix2[0])/50

    n_units = len(keep_)
    X_ = np.zeros(( len(IX), sz, n_units ))
    X2_ = np.zeros((len(IX), sz2, n_units))

    Y_ = np.tile(YY[:, np.newaxis], (1, sz))
    Y2_ = np.tile(YY[:, np.newaxis], (1, sz2))
    
    for t in range(sz):
        st = time_ix[0]+t*(bin_size)
        ix_ = np.ix_(IX, range(st, st+bin_size), range(n_units))
        x_tmp = XX[ix_]
        X_[:, t, :] = np.mean(XX[ix_], axis=1)

    for t in range(sz2):
        st = time_ix2[0]+t*(bin_size)
        ix_ = np.ix_(IX, range(st, st+bin_size), range(n_units))
        x_tmp = XX[ix_]
        X2_[:, t, :] = np.mean(XX[ix_], axis=1)

    #50 ms bins:
    X = X_.reshape(X_.shape[0]*X_.shape[1], X_.shape[2])
    X2 = X2_.reshape(X2_.shape[0]*X2_.shape[1], X2_.shape[2])
    Y = Y_.reshape(Y_.shape[0]*Y_.shape[1])

    import sklearn.decomposition as sd
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

    #Transform X --> Y: 
    dmn_X = X - np.tile(np.mean(X, axis=0)[np.newaxis, :], [X.shape[0], 1])
    y = np.array(u[:, :n_dim_main_shared].T*dmn_X.T)

    dmn_X2 = X2 - np.tile(np.mean(X2, axis=0)[np.newaxis, :], [X2.shape[0], 1])
    y2 = np.array(u[:, :n_dim_main_shared].T*dmn_X2.T)

    # Factors x Trials x Time
    y_t = y.reshape(y.shape[0], Y_.shape[0], Y_.shape[1], )
    y2_t = y2.reshape(y2.shape[0], Y2_.shape[0], Y2_.shape[1])

    # 3D code!
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    f, ax2 = plt.subplots()

    #Order factor 1 by RT: 
    cutoff_ix = len(kin_rt)
    ordered_fact = np.zeros((cutoff_ix, y_t.shape[2]))
    ordered_fact = y_t[0, :cutoff_ix, :]

    ordering_rt = np.argsort(kin_rt)
    ordered_fact = ordered_fact[ordering_rt, :]
    
    #Plot ordered trials, activation of factor 1: 
    plt.pcolormesh(ordered_fact)

    f
    #Plot factor 1: 
    for il, l in enumerate(np.unique(YY)):
        ix = np.nonzero(YY==l)[0]
        for i in ix:
            #ax.plot3D(y_t[a2p[0], i,:], y_t[a2p[1], i,:], y_t[a2p[2], i,:], color=cmap[il])
            ax2[0].plot(y_t[0, i, :], color=cmap[il])
            #ax.scatter3D(y_t[i,0,a2p[0]], y_t[i,0,a2p[1]], y_t[i,0,a2p[2]], c='k')
            #ax.scatter3D(y_t[i,-1,a2p[0]], y_t[i,-1,a2p[1]], y_t[i,-1,a2p[2]], c='r')
            ax2[1].plot(y2_t[0, i, :], '-', color=cmap[il])
    #ax.scatter3D(y_t[ix,0], y[ix,1], y[ix,2], c=cmap[il], marker=cmark[il])


    # #Get neural speed projection: 
    # #For each bin get neural speed: 
    #     #Mean traj

    # mean_traj = np.mean(XX[IX,:,:], axis=0)
    # mean_vel = np.diff(mean_traj[np.arange(0, 2500, 10), :], axis=0)

    # ix_ = np.ix_(IX, np.arange(0, 2500, 10), np.arange(n_units))
    # X_vel = np.diff(XX[ix_], axis=1)

    # RT = []
    
    # for b in rt.keys():
    #     RT.append(rt[b])
    # RT = np.hstack((RT))
    # rt_sub = RT[IX]

    # proj = np.zeros((X_vel.shape[0], X_vel.shape[1]))
    # for t in range(X_vel.shape[1]):
    #     mn_t = mean_vel[t, :]
    #     for trl in range(len(IX)):
    #         t_x = X_vel[trl, t, :]
    #         proj[trl, t] = np.dot(mn_t, t_x)/np.linalg.norm(mn_t)













