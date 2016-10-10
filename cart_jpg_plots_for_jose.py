import state_space_cart as ssc
import matplotlib.pyplot as plt
import numpy as np
import load_files
import seaborn
import collections

seaborn.set(font='Arial',context='talk',font_scale=1.0,style='white')


def main():
	''' Plot 100 waveforms from each block on same plot '''

	master_blocks = ssc.master_blocks
	master_days = ssc.master_days

	for i_d, day in enumerate(master_days):

		f, ax = plt.subplots(nrows=9, ncols = 9)
		f2, ax2 = plt.subplots(nrows=9, ncols = 9)
		day_cells = collections.OrderedDict()

		for i_b, blk in enumerate(master_blocks[i_d]):
			spk = load_files.load(blk, day, animal='cart', include_hdfstuff=False)
			keys = np.array([k for k in spk.keys() if k not in ['ad124', 'ad124_ts', 'ts_func', 'hdf', 'wf_dict']])
			
			#Add new units
			for k in keys: 
				if k not in day_cells: 
					day_cells[k] = []

			for i_k, key in enumerate(day_cells.keys()):
				if i_k < 81:
					axi = ax[i_k/9, i_k%9]
				else:
					axi = ax2[(i_k-81)/9, (i_k-81)%9]

				if key in spk['wf_dict']:
					wf = spk['wf_dict'][key]

					#Plot 100: 
					ix = np.linspace(0, wf.shape[0]-1, 100).astype(int)
					ix = np.unique(ix)

					axi.plot(wf[ix,:].T, linewidth = 0.5)
					if i_k in [0, 81]:
						axi.set_title('Day: '+day+', sig'+key)
					else:
						axi.set_title('sig'+key)
					axi.set_xticklabels([])
					axi.set_yticklabels([])
				else:
					axi.plot([0, 32], [-0.3, 0], 'k-')
					axi.plot([0, 32], [0, -0.3], 'k-')

		f.savefig(day+'a.png')
		f2.savefig(day+'b.png')
if __name__ == "__main__":
	main()



   