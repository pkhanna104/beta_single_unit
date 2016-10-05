import numpy as np
import scipy.io as sio
import os
from plexon import plexfile
from riglib.dio import parse
import glob
import tables

extensions = ['resort_all_all', 'resort_all_sub_all', 'resort_all']
master_save_directory = '/Volumes/My Book/pk/resorted_grom2015_mat'
#cart_save_directory = '/Volumes/My Book/pk/cart_2015'
cart_save_directory = '/media/lab/My Book/pk/cart_2015'


def load(block, day, animal='grom',return_name_only=False, include_hdfstuff=False):
    subj = animal
    if subj == 'grom':
        print 'processing grom data: ', block, day
        loaded = False
        fx = 0
        while fx < len(extensions):
            fname = master_save_directory+'/'+subj+day+block+'-'+extensions[fx]+'.mat'
            if os.path.isfile(fname):
                if return_name_only:
                    t = fname
                    loaded = True
                    break
                else:
                    t = sio.loadmat(fname)
                    loaded = True
                    break
            else:
                fx += 1
        
        if loaded:
            return t
        
        else:
            #Try Seba file set:
            fname = '/Volumes/TimeMachineBackups/Seba_MC_mat_files/'+subj+day+block+'.mat'
            if os.path.isfile(fname):
                if return_name_only:
                    t = fname
                    loaded=True
                else:
                    t = sio.loadmat(fname)
                    loaded = True
            if loaded:
                return t
            else:
                print 'No file: ', subj, day, block
                return None

    elif subj == 'cart':
        print 'processing cart data: ', block, day
        from state_space_cart import master_blocks_te, master_days, master_blocks
        i_d = np.nonzero(np.array(master_days)==day)[0]
        i_b = master_blocks[i_d].find(block)
        te = master_blocks_te[i_d][i_b]
        print 'te: ', te
        
        loaded = False
        fx = 0

        fname = cart_save_directory+'/'+subj+'20'+day[4:]+day[:2]+day[2:4]+'_*te'+str(te)+'.plx'
        print 'fname ', fname
        files = glob.glob(fname)
        print 'files ', files
        if len(files) == 1:
            proceed = True
            fname = files[0]
        else:
            proceed = False

        if proceed:
            if return_name_only:
                t = fname
                loaded = True
                
            else:
                plx = plexfile.openFile(fname)
                t, wf = plx_to_dict_format(plx)
                t['wf_dict'] = wf

                if include_hdfstuff:
                    hdf = tables.openFile(fname[:-4]+'.hdf')
                    ts_func = get_hdf_to_plx_fcn(hdf, plx)

                    t['hdf'] = hdf
                    t['ts_func'] = ts_func

                loaded = True

        if loaded:
            return t

def get_hdf_to_plx_fcn(hdf, plx):

    def sys_eq(sys1, sys2):
        return sys1 in [sys2, sys2[1:]]

    events = plx.events[:].data
    # get system registrations
    reg = parse.registrations(events)
    syskey = None

    # find the key for the task data
    for key, system in reg.items():
        if sys_eq(system[0], 'task'):
            syskey = key
            break

    ts = parse.rowbyte(events)[syskey] 

    # Use checksum in ts to make sure there are the right number of rows in hdf.
    if len(hdf.root.task)<len(ts):
        ts = ts[1:]
    assert np.all(np.arange(len(ts))%256==ts[:,1]), \
        "Dropped frames detected!"

    if len(ts) < len(hdf.root.task):
        print "Warning! Frames missing at end of plx file. Plx recording may have been stopped early."

    ts = ts[:,0]

    # Define a function to translate plx timestamps to hdf and vice versa for
    # this session.
    def ts_func(input_times, output_type):

        if output_type == 'plx':
            if len(input_times)>len(ts):
                input_times = input_times[:len(ts)]
            output = [ts[time] for time in input_times]

        if output_type == 'hdf':
            output = [np.searchsorted(ts, time) for time in input_times]

        return np.array(output)

    return ts_func



def plx_to_dict_format(plx):
    spk = dict()
    wf = dict()

    plx_spks = plx.spikes[:].data
    plx_wf = plx.spikes[:].waveforms

    for it, ts in enumerate(plx_spks):
        if ts[2] > 0:
            sig = str(ts[1])+num2alpha(ts[2])
            if sig in spk:
                spk[sig].append(ts[0])
                wf[sig].append(plx_wf[it, :])
            else:
                spk[sig] = [ts[0]]
                wf[sig] = [plx_wf[it, :]]

    #Add LFP channel:
    #Make sure there are 256 channels: 
    tmp = plx.lfp[:.1].data
    if tmp.shape[1] == 256:
        spk['ad124'] = plx.lfp[:].data[:, 124]
        spk['ad124_ts'] = plx.lfp[:].time
    else:
        raise

    for k in wf.keys():
        wf[k] = np.vstack((wf[k]))
        spk[k] = np.array(spk[k])

    return spk, wf


def num2alpha(i):
    #Remember, unit 0 is UNSORTED
    text = '_abcdefghijklmno'
    return text[i]



