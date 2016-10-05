from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import load_files
# Analyses you need to be able to run: 
    # 1. Beta PSTH
    # 2. Canolty
    # 3. 6class LDA
    # 4. Cts. decoding. 

NUM_COLORS = 15
cm = get_cmap('gist_heat')
cmap = []
for i in range(NUM_COLORS):
    cmap.append(cm(1.*i/NUM_COLORS))  # colors

test_days = ['011315']
master_days = ['011315','011415', '011515','011515']

test_blocks = ['abc']
master_blocks = ['abc', 'abc', 'abc','abcd'] 
master_blocks_te= [[6291, 6294, 6295], [6297, 6298, 6299 ], [6302, 6303, 6304], [6305, 6306, 6307, 6308]]

# [mc, [nf]]
# [6291, [6294, 6295, ]] # a:01, b:04, c:05
# [6297, [6298, 6299 ]] # a:01, b:02, c:03
# [6302, [6303, 6304, ]] # a:02, b:03, c:04
# [6305, [6306, 6307, 6308]] #a:01 , b:02, c:03, d:04



test_mc_indicator = ['100']
master_mc_indicator =  ['100', '100', '100', '1000']

# Master cell list for sorted files: 
master_cell_list = dict(
    date_011315=np.hstack([['22a', '38a', '51a', '52a', '53a', '73a', '96a', '97a', '105a', '106a', '130a', '176a', 
    '216a', '228a', '232a', '235a', '', ],  #Good multiunit list
    ['133a'],  #Great cell list
    ['1a', '3a', '23a', '26a', '36a', '37a', '41a', '46a', '54a', '55a', '56a', '57a', '58a', '59a', '60a', '71a',
    '77a', '90a','107a', '122a', '131a', '140a', '148a', '152a', '160a', '193a', '195a', '197a', '198a', '204a', 
    '205a', '212a', '215a', '226a', '227a', '245a', '248a', '256a']]), #Multiunit list


    date_011415=np.hstack([['22a', '215a', '228a', '235a'],  #Good cell list
    [''],  #Great cell list
    ['3a', '26a', '36a', '37a', '38a', '41a', '46a', '51a', '52a', '53a',
    '57a', '58a', '59a', '61a', '62a', '63a', '64a', '73a', '96a', '97a', 
    '102a', '105a', '107a', '109a', '129a', '130a', '131a', '132a', '133a', 
    '134a', '139a', '140a', '144a', '148a', '149a', '151a', '158a', '172a', 
    '195a', '198a', '204a', '207a', '211a', '216a', '227a', '230a', '233a', 
    '234a', '242a', '245a', '248a']]), #Multiunit list


    date_011515=np.hstack([[],  #Good cell list
    ['13a'],  #Great cell list
    ['1a', ]]), #Multiunit list

    date_011615=np.hstack([[],  #Good cell list
    ['13a'],  #Great cell list
    ['1a', ]]), #Multiunit list 
    )
#Timing splits:
#1_13_15, split a, b, c @: 307, 4219
#1_14_15, split a, b, c @: 309, 4792
nrows = 5

def get_cells(plot=True, days=None, blocks=None, mc_indicator=None, test=False):
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
        ix_good = np.array([i for i, j in enumerate(master_cell_list['date_'+str(d)]) if len(j) > 1])
        good_cell_list = list(np.unique(master_cell_list['date_'+str(d)]))

        #Init graphs for 1) WFs, 2) Mean FR 
        if plot:
            y = int(np.ceil(len(good_cell_list)/float(nrows)))
            f, ax= plt.subplots(nrows = nrows, ncols = y)
            f2, ax2 = plt.subplots(nrows=nrows,ncols = y)


        for ib, b in enumerate(blocks[i_d]):

            spk = load_files.load(b, d, subj='cart')
            
            if spk is not None:
                for ig, gc in enumerate(good_cell_list):
                    key = gc
                    if key in spk.keys():

                        #Time stamps for LFP in secs --> ms
                        secs = spk['ad124_ts']
                        axi = ax[ig/y, ig%y]
                        axi2 = ax2[ig/y, ig%y]

                        if plot:
                            mn = np.mean(spk['wf_dict'][key], axis=0)
                            axi.plot(mn, color=cmap[ib])
                            sem = np.std(spk[key], axis=0)/float(np.sqrt(spk[key].shape[0]))
                            #axi.fill_between(np.arange(32), mn-sem, mn+sem, color=cmap[ib], alpha=0.5)
                            axi.set_title(key)
                            axi2.bar(ib, spk[key].shape[0]/float(len(secs)/1000.), color=cmap[ib])
                            axi2.set_title(key)

                        #Remove cells w/ mean FR less than 0.5 spikes per second
                        if spk[key].shape[0]/(secs[-1]-secs[0]) < 0.5:
                            incomplete_list.append(gc)
                    else:
                        print 'No ', key, ' in block ', b, ' , date ', d
                        incomplete_list.append(gc)

        good_set = set(good_cell_list)
        incomp_set = set(incomplete_list)
        keep_ = np.array(list(good_set.difference(incomp_set)))
        keep_cell_dict[d] = keep_
    return keep_cell_dict, days, blocks, mc_indicator
