import numpy as np
def get_windows(left_ts,right_ts):
    st = min(left_ts[0],right_ts[0])
    et = max(right_ts[-1],left_ts[0])
    itr = np.round((et-st)/60000)
    window = [(st+i*60000,st+(i+1)*60000) for i in range(int(itr-1)) if st+i*60000<et]
    print([window[i][1]-window[i][0] for i in range(len(window))])