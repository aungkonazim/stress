import os
import numpy as np
import pandas as pd
from pylab import *


path = 'C:\\Users\\aungkon\\Desktop\\model\\data'
participant_dir = list(os.listdir(path))
for participant in participant_dir:
    count = 0
    path_to_date = path + '\\'+str(participant)
    date_dir = list(filter(lambda x: x[0] == '2',os.listdir(path_to_date)))
    for date in date_dir:
        path_to_csv = path_to_date + '\\'+str(date)+ '\\'
        data = pd.read_csv(path_to_csv+'left_wrist.csv',sep=',').values[:,2:]
        seq = np.zeros((np.shape(data)[0]))
        for i in range(np.shape(seq)[0]):
            seq[i] = ((uint8(data[i,18]) & int('00000011',2))<<8) | (uint8(data[i,19]))
        figure()
        hist(np.diff(seq),100)
        show()