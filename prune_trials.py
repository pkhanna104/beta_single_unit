import predict_kin_w_spk


keep_dict, spk_dict, lfp_dict, lfp_lab, blocks, days, rt_dict, beta_dict, beta_cont_dict, bef, aft, go_times_dict, mc_indicator = predict_kin_w_spk.get_unbinned(test=False)

b = 25
kin_signal_dict, binned_kin_signal_dict = predict_kin_w_spk.get_kin(days, blocks, bef, aft, go_times_dict, lfp_lab, b, smooth = 50)


for i_d, d in enumerate(days) :
	for i, (mc, blk) in enumerate(zip(mc_indicator[i_d], blocks[i_d])):
		if mc == '1':
			f, ax = plt.subplots(ncols = 3)
			ax[0].pcolormesh(binned_kin_signal_dict[blk, d][:, 2, :])
			ax[1].pcolormesh(binned_kin_signal_dict[blk, d][:, 3, :])
			ax[2].plot(binned_kin_signal_dict[blk, d][:, 0, :].reshape(-1), binned_kin_signal_dict[blk, d][:, 1, :].reshape(-1))
			plt.show()