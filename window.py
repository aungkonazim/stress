import numpy as np
def get_windows(left_ts):
    st = left_ts[0]
    et = left_ts[-1]
    itr = np.round((et-st)/30000)
    window = [(st+i*60000,st+(i+1)*60000) for i in range(int(itr-1)) if st+i*60000<et]
    return window