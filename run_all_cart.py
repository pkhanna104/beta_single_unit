import canolty_tuning
import predict_kin_w_spk
import predict_w_mc_decoders
import gc
import predict_kin_w_spk
import predict_w_mc_decoders
import multiprocessing as mp
import pickle
import time
def main():
    # for i in range(60):
    #     print 'sleeping, min: ', i
    #     time.sleep(60)

    print '##################'
    print '##################'
    print '##################'
    print 'Beg. of Canolty!'
    #canolty_tuning.main_mp()
    #cr = canolty_tuning.main(test=False, animal='cart')
    #canolty_tuning.plot_chief_results(cr)
    print '##################'
    print '##################'
    print '##################'
    print '##################'
    print 'End of Canolty!'
    gc.collect()

    print '##################'
    print '##################'
    print '##################'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    print 'Beg. of LDA w/in task!'
    # predict_kin_w_spk.six_class_LDA(test=False, within_task_comparison=True, 
    #     x_task_comparison=False, all_cells = True, animal='cart')
    

    # days = ['011315', '011415', '011515', '011615']
    # pool1 = mp.Pool()
    # results1 = pool1.map(predict_kin_w_spk.six_class_LDA_by_day_within, days)
    # CONF = {}
    # for i_d, day in enumerate(days):
    #     CONF[day] = results1[i_d]
    # pickle.dump(CONF, open('six_class_LDA_within_task_xval5_all_cells_mp_attempt.pkl', 'wb'))

    # predict_kin_w_spk.six_class_LDA(test=False, within_task_comparison=False, 
    # x_task_comparison=True, all_cells = True, animal='cart')


    # days = ['011315', '011415', '011515', '011615']
    # pool2 = mp.Pool()
    # results2 = pool2.map(predict_kin_w_spk.six_class_LDA_by_day_across, days)
    # CONF = {}
    # for i_d, day in enumerate(days):
    #     CONF[day] = results2[i_d]
    # pickle.dump(CONF, open('six_class_LDA_x_task_compare_all6classes_all_cells_mp_attempt.pkl', 'wb'))

    print '##################'
    print '##################'
    print '##################'
    print 'Beg. of cts decoding!'

    gc.collect()
    predict_w_mc_decoders.main_full_file_plt('cart', True)

    print '##################'
    print '##################'
    print '##################'
    print 'End of All!'

if __name__ == '__main__':
    main()