import pandas as pd
import numpy as np
import os
from stress_marks import get_stress_marks
from find_peaks import find_rr_interval
from outlier_calculation import compute_outlier_ecg,Quality
from window import get_windows
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

for path in path_collection:
    ppg_left = pd.read_csv(path+'left_wrist_prob.csv',names=[str(i) for i in range(3)],sep=',')
    ppg_right = pd.read_csv(path+'right_wrist_prob.csv',names=[str(i) for i in range(3)],sep=',')
    label = pd.read_csv(path+'label.csv',sep=',')
    stress_times = get_stress_marks(label)
    left_ts,left_rr = find_rr_interval(ppg_left.as_matrix())
    right_ts,right_rr = find_rr_interval(ppg_right.as_matrix())
    outlier_right = compute_outlier_ecg(right_ts,right_rr/1000)
    outlier_left = compute_outlier_ecg(left_ts,left_rr/1000)
    windows = get_windows(left_ts,right_ts)

