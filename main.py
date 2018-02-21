from stress_marks import get_stress_marks
from outlier_calculation import compute_outlier_ecg,Quality
from window import get_windows
from get_test_files import get_test_data
from keras.models import load_model
import pandas as pd
import os
from scipy.io import loadmat
from predict_cnn import predict_likelihood
from get_JU_result import get_realizations
import numpy as np
from pylab import *


path_model="C:\\Users\\aungkon\\Desktop\\model\\files2Azim_25Jan\\"
final_model=load_model(path_model+'model_all_Br4_Tr5.h5')
data = loadmat('C:\\Users\\aungkon\\Desktop\\model\\JU\\int_RR_dist_obj_kernel_Fs25.mat')
int_RR_dist_obj = data['int_RR_dist_obj']




path_collection = []
path = 'C:\\Users\\aungkon\\Desktop\\model\\data'
participant_dir = list(os.listdir(path))
for participant in participant_dir:
    count = 0
    path_to_date = path + '\\'+str(participant)
    date_dir = list(filter(lambda x: x[0] == '2',os.listdir(path_to_date)))
    for date in date_dir:
        path_to_csv = path_to_date + '\\'+str(date)+ '\\'
        path_collection.append(path_to_csv)
final_RR = []
final_time = []
path_col = []
for path in path_collection[1:2]:
    ppg_left = pd.read_csv(path+'left_wrist.csv',header=None).values
    ppg_right = pd.read_csv(path+'right_wrist.csv',header=None).values
    label = pd.read_csv(path+'label.csv',sep=',')
    stress_times = get_stress_marks(label)
    windows = get_windows(ppg_left[:,0])
    for u,v in windows:
        arr_LA = ppg_left[(ppg_left[:,0]>=u) & (ppg_left[:,0]<=v)]
        arr_RA = ppg_right[(ppg_right[:,0]>=u) & (ppg_right[:,0]<=v)]
        if len(arr_RA[:,0]) > .5*60*25:
            list_test_RA,list_test_RA_t = get_test_data(arr_RA)
        if len(arr_LA[:,0]) > .5*60*25:
            list_test_LA,list_test_LA_t = get_test_data(arr_LA)
        if 'list_test_RA_t' in locals() and list_test_RA_t is not 0:
            ts_RA,pred_RA,Acc_RMS_RA = predict_likelihood(final_model,list_test_RA,list_test_RA_t)
        if 'list_test_LA_t' in locals() and list_test_LA_t is not 0:
            ts_LA,pred_LA,Acc_RMS_LA = predict_likelihood(final_model,list_test_RA,list_test_RA_t)


        if 'Acc_RMS_LA' in locals() and 'Acc_RMS_RA' in locals():
            if Acc_RMS_LA < Acc_RMS_RA:
                ts = ts_LA
                pred = pred_LA
            else:
                ts = ts_RA
                pred = pred_RA
        elif 'Acc_RMS_LA' in locals() and 'Acc_RMS_RA' not in locals():
            ts = ts_LA
            pred = pred_LA
        elif 'Acc_RMS_LA' not in locals() and 'Acc_RMS_RA' in locals():
            ts = ts_RA
            pred = pred_RA

        if 'ts' in locals() and 'pred' in locals():
            RR = get_realizations(int_RR_dist_obj,pred)
            ts = np.array(ts)
            total = len(RR)
            acceptable = 0
            final_RR.append([])
            final_time.append([])
            path_col.append([])
            for realization in RR:
                time_rr = ts[realization]
                rr_interval = ts[realization]
                time_rr = time_rr[1:]
                rr_interval = np.diff(rr_interval)
                outlier = compute_outlier_ecg(time_rr,rr_interval/1000)
                outlier1 = list(filter(lambda x:x[1]==Quality.ACCEPTABLE,outlier))
                if len(outlier1) > .64*len(time_rr):
                    print(len(outlier1),len(time_rr))
                    final_RR[-1].append(rr_interval)
                    final_time[-1].append(time_rr)
                    path_col[-1].append(path)
            print("-------------------------------------------------------------------------------------------------------------------")
            print("acceptable = ",acceptable," total = ",total)
            print("-------------------------------------------------------------------------------------------------------------------")
            # break

            # outlier_right = compute_outlier_ecg(right_ts,right_rr/1000)
    # outlier_left = compute_outlier_ecg(left_ts,left_rr/1000)
    #
final_RR = np.array(final_RR)
final_time = np.array(final_time)
path_col = np.array(path_col)

np.savez('all_windows',rr=final_RR,ts=final_time,path=path_col)