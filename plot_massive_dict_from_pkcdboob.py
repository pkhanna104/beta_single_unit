import pickle
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy.stats
seaborn.set(font='Arial',context='talk',font_scale=1.5,style='whitegrid')
cmap = [[178, 24, 43], [239, 138, 98], [253, 219, 199], [209, 229, 240], [103, 169, 207], [33, 102, 172]]

def plot_(massive_dict):
	days = massive_dict.keys()
	c_d = dict(mc='b', nf='r')
	# Plot On vs. Off beta during MC
	f, ax  = plt.subplots(ncols=2, nrows=2,figsize=(10,8))
	measures = dict()

	for i_d, d in enumerate(days):
		

		md = massive_dict[d]
		n = -1
		for ik, k in enumerate(['mc', 'nf']):

			#First plot: on vs. off in actual NF vs. actual MC: 
			on_beta = np.nonzero(md[k+'_beta']==1)[0]
			off_beta = np.nonzero(md[k+'_beta']==0)[0]

			for im, m in enumerate(['', '_hat']):
				key = k+'_kin'+m
				try:
					measures[k, m, 'on'].append(np.mean(md[key][on_beta]))
					measures[k, m, 'off'].append(np.mean(md[key][off_beta]))

				except:
					measures[k, m, 'on'] = [np.mean(md[key][on_beta])]
					measures[k, m, 'off'] = [np.mean(md[key][off_beta])]

				# if np.logical_and(m=='', k=='nf'):
				# 	print 'skip'
				# else:
				n+=1
				sem_n = [len(md[key][off_beta]), len(md[key][on_beta])]
				sem = [np.std(md[key][off_beta])/np.sqrt(sem_n[0]), np.std(md[key][on_beta])/np.sqrt(sem_n[1])]
				ax[n/2, n%2].errorbar([0, 1], [np.mean(md[key][off_beta]), np.mean(md[key][on_beta])], yerr=sem, color=tuple(np.array(cmap[i_d])/255.), 
					markersize=20, linewidth=5) #linecolor = tuple(np.array(cmap[i_d])/255.))

	titles = ['Actual MC Hand Speed', 'Decoded MC Hand Speed', 'Actual NF Hand Speed', 'Decoded NF Hand Speed']
	for i in range(4):
		ax[i/2, i%2].set_xlim([-1, 2])
		ax[i/2, i%2].set_xticks([0, 1])
		ax[i/2, i%2].set_xticklabels(['Off Beta', 'On Beta'])
		ax[i/2, i%2].set_ylabel('Mean Speed (cm/sec)')
		ax[i/2, i%2].set_title(titles[i])

		if i==0:
			x0 = np.hstack((measures['mc', '', 'off']))
			x1 = np.hstack((measures['mc', '', 'on']))
			print 'mc actual: '
			ax[i/2, i%2].set_ylim([0., 8])
			ymax = 7.5
		if i == 1:
			x0 = np.hstack((measures['mc', '_hat', 'off']))
			x1 = np.hstack((measures['mc', '_hat', 'on']))
			print 'mc hat'
			ax[i/2, i%2].set_ylim([0., 8])
			ymax = 7.5
		if i == 2:
			x0 = np.hstack((measures['nf', '', 'off']))
			x1 = np.hstack((measures['nf', '', 'on']))
			print 'nf actual'
			ax[i/2, i%2].set_ylim([0., 4])
			ymax = 3.5
		if i==3:
			x0 = np.hstack((measures['nf', '_hat', 'off']))
			x1 = np.hstack((measures['nf', '_hat', 'on']))
			print 'nf hat'
			ax[i/2, i%2].set_ylim([1.5, 4])
			ymax = 3.5
		t, p = scipy.stats.ttest_rel(x0, x1)
		print p, t, len(x0), len(x1)

	
		ax[i/2, i%2].plot([0, 0], [ymax-.4, ymax-.2], 'k-')
		ax[i/2, i%2].plot([1, 1], [ymax-.4, ymax-.2], 'k-')
		ax[i/2, i%2].plot([0, 1], [ymax-.2, ymax-.2], 'k-')
		if p > 0.05:
			ax[i/2, i%2].text(0.5, ymax, 'p = '+str(int(1000*p)/1000.), horizontalalignment='center',fontsize=18)
		elif p > 0.01:
			ax[i/2, i%2].text(0.5, ymax, '*', horizontalalignment='center',fontsize=18 )
		elif p > 0.001:
			ax[i/2, i%2].text(0.5, ymax, '**', horizontalalignment='center',fontsize=18 )
		elif p <= 0.001:
			ax[i/2, i%2].text(0.5, ymax, '***', horizontalalignment='center',fontsize=18 )

	plt.tight_layout()
	#plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/grom_spk_v2_may2016/decoding_analysis.eps', format='eps', dpi=300)


def plot_error(massive_dict):
	days = massive_dict.keys()
	c_d = dict(mc='b', nf='r')
	# Plot On vs. Off beta during MC
	f, ax = plt.subplots(ncols=2,figsize=(18, 8))

	measures = dict()
	full_measures = dict()
	for k in ['mc', 'nf']:
		for b in ['on', 'off']:
			full_measures[k, b] = []

	for i_d, d in enumerate(days):
		
		md = massive_dict[d]
		n = -1
		for ik, k in enumerate(['mc', 'nf']):

			#First plot: on vs. off in actual NF vs. actual MC: 
			on_beta = np.nonzero(md[k+'_beta']==1)[0]
			off_beta = np.nonzero(md[k+'_beta']==0)[0]

			for ib, (beta, bt) in enumerate(zip([on_beta, off_beta], ['on', 'off'])):
				error = np.mean((md[k+'_kin'][beta] - md[k+'_kin_hat'][beta])**2)

				measures[k, bt] = error
				full_measures[k, bt].append(error)

			ax[ik].plot([0, 1], [measures[k, 'off'], measures[k, 'on']], c_d[k]+'.-', markersize=20)

	for ik in range(2):
		ax[ik].set_xlim([-1, 2])
		ax[ik].set_xticks([0, 1])
		ax[ik].set_xticklabels(['Off Beta', 'On Beta'])
		ax[ik].set_ylabel('Mean Squ. Error in Decoding')

	titles = ['MC Decoding Error', 'NF Decoding Error']
	for i in range(2):
		if i==0:
			ax[i].set_ylim([0, 30])
			x0 = np.hstack((full_measures['mc', 'off']))
			x1 = np.hstack((full_measures['mc', 'on']))
			print 'mc '
			
		if i == 1:
			ax[i].set_ylim([0, 10])
			x0 = np.hstack((full_measures['nf', 'off']))
			x1 = np.hstack((full_measures['nf', 'on']))
			print 'nf actual'
			
		t, p = scipy.stats.ttest_rel(x0, x1)
		print p

	
		ax[i/2, i%2].plot([0, 0], [ymax-.4, ymax-.2], 'k-')
		ax[i/2, i%2].plot([1, 1], [ymax-.4, ymax-.2], 'k-')
		ax[i/2, i%2].plot([0, 1], [ymax-.2, ymax-.2], 'k-')
		if p > 0.05:
			ax[i/2, i%2].text(0.5, ymax, 'p = '+str(int(1000*p)/1000.), horizontalalignment='center',fontsize=18)
		elif p > 0.01:
			ax[i/2, i%2].text(0.5, ymax, '*', horizontalalignment='center',fontsize=18 )
		elif p > 0.001:
			ax[i/2, i%2].text(0.5, ymax, '**', horizontalalignment='center',fontsize=18 )
		elif p <= 0.001:
			ax[i/2, i%2].text(0.5, ymax, '***', horizontalalignment='center',fontsize=18 )

	plt.tight_layout()
	plt.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/lfp_multitask/analysis/grom_spk_v2_may2016/decoding_analysis.eps', format='eps', dpi=300)

