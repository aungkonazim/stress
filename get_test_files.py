# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:59:50 2018

@author: agarwal.270
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from pylab import *
from scipy import signal as sig
import peakutils as pk
import time
import os
from tomkin import detect_rpeak
import glob
# In[0]
close('all')

def Preprc(raw_data=[],data=None,seq=None,name='RA',filt=False,dsample=False):
    flag=0
    Fs=25
    k=1

    # process recieved arrays (data_arr1=data, data_arr2=time,seq)
    if len(raw_data)!=0:
        if name in ['RA','LA']:
            data_arr1,data_arr2,err_pkts=process_raw_PPG(raw_data)
            # print(data_arr1.shape,data_arr2.shape)
        else:
            data_arr1,data_arr2,err_pkts=process_raw_ECG(raw_data)
            # print(data_arr1.shape,data_arr2.shape)
    else:
        data_arr1=data[:,2:]; # [100:,:]
        data_arr_temp=seq;
        data_arr2=np.stack((data_arr_temp[:,0],data_arr_temp[:,2]),axis=1)
        err_pkts=0;

    #no. of packet errors check
    err_pkts_chk=(err_pkts>int(0.5*len(data_arr1)))  #check for no. of errors shouldn't be more than 5% of all data

    # Check if ECG signal
    if name=='ECG':
        flag=1
        Fs=100
        k=(1/4)

        #seq=seq[:round(len(data_arr1)/4)]   #match the length of seq. nos. and data

    seq=np.copy(data_arr2[:,1])

    # downsample ECG data
    if flag==1 and dsample==True:
        row_idx=range(3,len(data_arr1[:,0]),4)
        data_arr1=data_arr1[row_idx,:]
        Fs=25

    # Store averages
    avg=data_arr1.mean(axis=0)

    #Filter data
    if (filt):
        data_arr1=filtr(data_arr1,Fs,filt)
    else:
        data_arr1=sig.detrend(data_arr1,type='constant',axis=0)

    # make Sq no. ordered
    d=np.diff(seq);
    idx1=mlab.find(d<-(1023-50));
    idx1=np.append(idx1,len(seq)-1)
    for i in range(len(idx1)-1):
        seq[idx1[i]+1:idx1[i+1]+1]=seq[idx1[i]+1:idx1[i+1]+1]-(i+1)*d[idx1[i]]
    seq=(seq-seq[0]).astype(int).reshape((len(seq)))
    seq_max=max(seq) #just some heuristic to make ECG  seq value 4 times

    # model of Seq to time
    coeff=np.polyfit(seq,data_arr2[:,0],1)
    time_RA=np.polyval(coeff,seq)
    #figure()
    #plot(time_RA,'o')

    # "Upsample" (fractional) sequence number if ECG
    if flag==1 and dsample==False:
        lst=[]
        for x in seq[:int(len(data_arr1)/4)]:
            lst=lst+[x-0.75,x-0.5,x-0.25,x]
        seq=np.array(lst)

    # print('seq=',seq.shape,'data_arr1=',data_arr1.shape)
    arr1=np.concatenate([seq.reshape((len(seq),1)),data_arr1],axis=1)
    if flag==0:
        if raw_data.all!=None:
            df1=pd.DataFrame(arr1,columns=['Seq','AccX','AccY','AccZ','GyroX','GyroY','GyroZ','LED1','LED2','LED3'])
        else:
            df1=pd.DataFrame(arr1,columns=['Seq','LED1','LED2','LED3'])
    else:
        df1=pd.DataFrame(arr1,columns=['Seq','Data1'])
        df1.Data1[df1.Data1==(-avg[0])]=np.nan

    df1.drop_duplicates(subset=['Seq'],inplace=True) #remove any redundancies
    # print(max(seq))
    # managing lost packets
    if flag==1 and dsample==False:
        #df2=pd.DataFrame((1/4)*np.array(range(-3,(len(data_arr2)-1)*4+1)),columns=['Seq'])
        df2=pd.DataFrame((1/4)*np.array(range(-3,(int(seq_max))*4+1)),columns=['Seq'])
    else:
        df2=pd.DataFrame(np.array(range(seq_max+1)),columns=['Seq'])

    itime=data_arr2[0,0];ftime=data_arr2[-1,0]
    #itime=np.polyval(coeff,df2.values[0]) # Calculating initial time based on the fitted curve
    df3=df2.merge(df1,how='left',on=['Seq'])
    #print(df3.isnull().sum())
    #df3['time']=pd.to_datetime(np.polyval(coeff,df3['Seq']),unit='ms')  # Prof's suggestion
    #df3['time']=pd.to_datetime(itime+(40*df3['Seq']),unit='ms')  # After Ju's discussion
    df3['time']=pd.to_datetime(linspace(itime,ftime,len(df2)),unit='ms')  # After Ju's 2nd discussion
    df3.set_index('time',inplace=True)
    df3.interpolate(method='time',axis=0,inplace=True) #filling missing data
    df3.dropna(inplace=True);
    # print(df3.isnull().sum())
    seq_diff=np.diff(df3['Seq'].values)
    seq_diff_chk=not((int(sum(seq_diff))==len(seq_diff)) | (sum(seq_diff)==0.25*len(seq_diff))) # all values should be 1
    df3['time_stamps']=linspace(itime,ftime,len(df2))
    time_diff=(df3['time_stamps'].values[1]-df3['time_stamps'].values[0])  #randommlly selected timestamps
    time_diff_chk=((time_diff>(k*41)) | (time_diff<(k*39))) # all values should be 1
    df3.drop(['Seq','time_stamps'],axis=1,inplace=True)
    return (df3,coeff,avg,seq_diff_chk,time_diff_chk,err_pkts_chk)


def process_raw_PPG(raw_data):
    data = raw_data
    Vals = data[:,2:]
    num_samples = Vals.shape[0]
    ts = data[:,0]
    Accx=np.zeros((num_samples));Accy=np.zeros((num_samples))
    Accz=np.zeros((num_samples));Gyrox=np.zeros((num_samples))
    Gyroy=np.zeros((num_samples));Gyroz=np.zeros((num_samples))
    led1=np.zeros((num_samples));led2=np.zeros((num_samples))
    led3=np.zeros((num_samples));seq=np.zeros((num_samples))
    time_stamps=np.zeros((num_samples))
    n=0;i=0;s=0;mis_pkts=0
    while (n)<(num_samples):
        time_stamps[i]=ts[n]
        Accx[i] = int16((uint8(Vals[n,0])<<8) | (uint8(Vals[n,1])))
        Accy[i] = int16((uint8(Vals[n,2])<<8) | (uint8(Vals[n,3])))
        Accz[i] = int16((uint8(Vals[n,4])<<8) | (uint8(Vals[n,5])))
        Gyrox[i] = int16((uint8(Vals[n,6])<<8) | (uint8(Vals[n,7])))
        Gyroy[i] = int16((uint8(Vals[n,8])<<8) | (uint8(Vals[n,9])))
        Gyroz[i] = int16((uint8(Vals[n,10])<<8) | (uint8(Vals[n,11])))
        led1[i]=(uint8(Vals[n,12])<<10) | (uint8(Vals[n,13])<<2) | ((uint8(Vals[n,14]) & int('11000000',2))>>6)
        led2[i]=((uint8(Vals[n,14]) & int('00111111',2))<<12) | (uint8(Vals[n,15])<<4) | ((uint8(Vals[n,16]) & int('11110000',2))>>4)
        led3[i]=((uint8(Vals[n,16]) & int('00001111',2))<<14) | (uint8(Vals[n,17])<<6) | ((uint8(Vals[n,18]) & int('11111100',2))>>2)
        seq[i]=((uint8(Vals[n,18]) & int('00000011',2))<<8) | (uint8(Vals[n,19]))
        if i>0:
            difer=int((seq[i]-seq[i-1])%1024)
            if difer>50:
                s=s+1 # keep a record of how many such errors occured
                n=n+1
                continue
            mis_pkts=mis_pkts+(difer-1)
        n=n+1;i=i+1
    # removing any trailing zeros
    seq=seq[:i];time_stamps=time_stamps[:i]
    Accx=Accx[:i]; Accy=Accy[:i]; Accz=Accz[:i]
    Gyrox=Gyrox[:i]; Gyroy=Gyroy[:i]; Gyroz=Gyroz[:i]
    led1=led1[:i]; led2=led2[:i]; led3=led3[:i]
    # print('no. of unknown seq errors in PPG= ',s)
    # print('no. of missed packets= {}'.format(mis_pkts))
    data_arr1=np.stack((Accx,Accy,Accz,Gyrox,Gyroy,Gyroz,led1,led2,led3),axis=1)
    data_arr2=np.concatenate((time_stamps.reshape(1,-1),seq.reshape(1,-1))).T
    return data_arr1,data_arr2,(mis_pkts+s)

def process_raw_ECG(raw_data):
    fl=0
    data = raw_data
    Vals = data[:,2:]
    num_samples = Vals.shape[0]
    ts = data[:,0]
    Ecg=np.zeros((6*(num_samples+1)));Ecg1=np.zeros((6*(num_samples+1)))
    seq=np.zeros((3*num_samples));time_stamps=np.zeros((3*num_samples))
    n=0;i=0;j=3;k=15;s=0;mis_pkts=0
    # Combining Bits
    while (n)<num_samples:
        seq[i]=((uint8(Vals[n,18]) & int('00001111',2))<<8) | (uint8(Vals[n,19]))
        time_stamps[i]=ts[n]*1
        if (seq[i])%4==0 and fl==0:
            seq[0]=seq[i];i=0;fl=1
        if n==int(num_samples/20): #condition for debugging
            deb=1;
        if fl==1:  # start only when 1st seq no. is divisible by 4
            # Managing missing packets
            if i!=0:
                difer=int((seq[i]-seq[i-1])%4096);
                if (difer<=50) & (difer>0):
                    mis_pkts=mis_pkts+(difer-1)
                    j=int(j+4*(difer));t_old=time_stamps[i]*1;s_old=seq[i]*1
                    for l in range(difer-1):
                        seq[i]=(seq[i-1]+1)%4096;
                        time_stamps[i]=time_stamps[i-1]*1
                        i=i+1 #compensate last increment later
                    time_stamps[i]=t_old*1;seq[i]=s_old*1
                else:
                    s=s+1 # keep a record of how many such errors occured
                    n=n+1;
                    continue # forget about this sample
                    #Assigning values when packets not missing
                #            print('j=',j,' 4*(num_samples+1)=',4*(num_samples+1))
            if n>=num_samples:
                break
            Ecg[j-3]=(uint8(Vals[n,6])<<8) | (uint8(Vals[n,7]))
            Ecg[j-2]=(uint8(Vals[n,8])<<8) | (uint8(Vals[n,9]))
            Ecg[j-1]=(uint8(Vals[n,10])<<8) | (uint8(Vals[n,11]))
            Ecg[j]=(uint8(Vals[n,15])<<8) | (uint8(Vals[n,16]))

            # arranging packets after set of 16 arrives
            while(k<=j):
                lst=list(np.array([1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16])+(k-16))
                Ecg1[lst]=Ecg[k-15:k+1]
                k=k+16
        n=n+1;i=i+1 # update the looping variables
    # print('no. of unknown errors in ECG= {}'.format(s))
    # removing any trailing zeros
    while ((Ecg1[-1]==0) | ((len(Ecg1)%4)!=0)):
        Ecg1=Ecg1[:-1]
    Ecg1_end=int(len(Ecg1)/4)
    seq=seq[:Ecg1_end]
    time_stamps=time_stamps[:Ecg1_end]-160 # -160 accounts for interleaving delay at sensor end

    Ecg1=Ecg1.reshape((-1,1))
    data_arr1=Ecg1
    data_arr2=np.stack((time_stamps,seq),axis=1)
    # print('no. of missed packets= {}'.format(mis_pkts))
    return data_arr1,data_arr2,(mis_pkts+s)


# In[1]
def filtr(X0,Fs,filt=False):
    nyq=Fs/2
    X1 = sig.detrend(X0,type='constant',axis=0); # Subtract mean
    if filt:
        # filter design used from Ju's code with slight changes for python syntax
        b = sig.firls(129,np.array([0,0.6,0.7,3,3.5,nyq]),np.array([0,0,1,1,0,0]),np.array([100*0.02,0.02,0.02]),nyq=nyq);
        X=np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
    else:
        X=X1
    X=sig.detrend(X,type='constant',axis=0); # subtracted mean again to center around x=0 just in case things changed during filtering
    return X

def norma(df):
    std=df.std();tol=1e-10
    for i in range(len(std)):
        if abs(std[i])<=tol:
            print(std.index[i]+' is saturated')
            std[i]=inf
    dfnor=(df-df.mean())/std
    return dfnor


# making time lagged dataframe with a window w
def data_setup(df,w,PTT_past,df_Acc_RMS):
    df.reset_index(inplace=True)
    #df.drop(['time'],axis=1,inplace=True)

    # preprocessing signal for correlation
    df1=df.copy()
    #df1[df1<0]=0; #cliiping negative values
    mu=0;sigma=3
    gx = np.arange(-3*sigma, 3*sigma+1,1)
    gaussian = (1/(sigma*(2*pi)**0.5))*np.exp(-(gx/sigma)**2/2)
    df1.Peak=4*np.convolve(df1.Peak, gaussian, mode="same")

    # plot df1peak
    # =============================================================================
    #     figure()
    #     plot(df.Peak);hold(True);plot(df1.Peak);grid(True)
    #
    # =============================================================================
    # Finding R-peak indices
    idx1_ori=df[df.Peak==1].index.values

    # Finding avg PTT over time t min.
    t=60/60; delta_s=min(len(df1.Peak),int(t*60000/40)); num=int(len(df1.Peak)/delta_s); PTT=np.zeros(num); corr=np.zeros((num+1,2*delta_s-1))
    Acc_RMS_thres=1500;   #Adjust as appropriate

    for i in range(num):
        #Acc_RMS=mean((df1.AccX[i*delta_s:(i+1)*delta_s]**2+df1.AccY[i*delta_s:(i+1)*delta_s]**2+df1.AccZ[i*delta_s:(i+1)*delta_s]**2)**0.5)
        Acc_RMS=mean(df_Acc_RMS[i*delta_s:(i+1)*delta_s])
        #print('ACC_RMS',Acc_RMS);time.sleep(0.5)
        if Acc_RMS>Acc_RMS_thres:  #if mean RMS value of acceleration is greater than threshold, don't estimate or update PTT
            PTT[i]=PTT_past
        else:
            corr1=np.correlate(df1.LED1[i*delta_s:(i+1)*delta_s],df1.Peak[i*delta_s:(i+1)*delta_s],mode='full')
            corr2=np.correlate(df1.LED2[i*delta_s:(i+1)*delta_s],df1.Peak[i*delta_s:(i+1)*delta_s],mode='full')
            corr3=np.correlate(df1.LED3[i*delta_s:(i+1)*delta_s],df1.Peak[i*delta_s:(i+1)*delta_s],mode='full')
            corr[i,:]=(corr1+corr2+corr3)/3
            #PTT[i]=np.argmax(corr[i,delta_s:delta_s+15]/np.arange(delta_s,delta_s-15,-1))
            PTT[i]=np.argmax(corr[i,delta_s-1:delta_s+15])
            PTT_past=PTT[i]

        # Adding PTT delay to R peaks
        if i!=0:
            idx1_ori[idx1_ori>=i*delta_s]=idx1_ori[idx1_ori>=i*delta_s]+(PTT[i]-PTT[i-1])
        else:
            idx1_ori[idx1_ori>=i*delta_s]=idx1_ori[idx1_ori>=i*delta_s]+PTT[i]

    x_axis=np.arange(-(delta_s-1),delta_s)
    #corr=corr/abs(delta_s-abs(x_axis)) # division by overlapping length
    corr[-1,:]=x_axis

    #del idx1_ori;idx1_ori=df[df.Peak==1].index.values #override PTT adjustment
    # Setting up arrays
    semilnth=int((w-1)/2)
    idx1=np.array(list(idx1_ori-1)+list(idx1_ori-2)+list(idx1_ori+0)+list(idx1_ori+2)+list(idx1_ori+1))
    idx1=idx1[(idx1+semilnth+1<=len(df.Peak)) & (idx1-semilnth>=0)]
    idx0=np.array(list(idx1_ori-5)+list(idx1_ori-7)+list(idx1_ori-9)+list(idx1_ori-11)+list(idx1_ori+5)+list(idx1_ori+7)+list(idx1_ori+9)+list(idx1_ori+11))
    idx0=idx0[(idx0+semilnth+1<=len(df.Peak)) & (idx0-semilnth>=0)]
    k=0
    for idx in [idx0,idx1]:
        arr1=np.array([df.AccX[i-semilnth:i+semilnth+1].values for i in idx]).T
        arr2=np.array([df.AccY[i-semilnth:i+semilnth+1].values for i in idx]).T
        arr3=np.array([df.AccZ[i-semilnth:i+semilnth+1].values for i in idx]).T
        arr4=np.array([df.LED1[i-semilnth:i+semilnth+1].values for i in idx]).T
        arr5=np.array([df.LED2[i-semilnth:i+semilnth+1].values for i in idx]).T
        arr6=np.array([df.LED3[i-semilnth:i+semilnth+1].values for i in idx]).T

        if k==0:
            arr_0=np.array([arr1,arr2,arr3,arr4,arr5,arr6]).T
            arr_0_Y=k*np.ones(arr1.shape[1])
        else:
            arr_1=np.array([arr1,arr2,arr3,arr4,arr5,arr6]).T
            arr_1_Y=k*np.ones(arr1.shape[1])
        k=k+1
        del arr1,arr2,arr3,arr4,arr5,arr6
    times=df.time[list(idx0)+list(idx1)]

    # Visualize some data for sanity check
    s=-min(len(idx1),9); #starting sample for visualization
    # =============================================================================
    #     figure()
    #     for k in range(abs(s)):
    #         subplot(3,3,k+1)
    #         plot(arr_1[k+s,:,3],'r');hold(True);plot(arr_1[k+s,:,4],'y');hold(True);plot(arr_1[k+s,:,5],'g');grid(True);
    #     title('X('+str(s)+') to X('+str(s+8)+') for peak data');legend(['r','i','g']);
    #
    #     figure()
    #     for k in range(abs(s)):
    #         subplot(3,3,k+1)
    #         plot(arr_0[k+s,:,3],'r');hold(True);plot(arr_0[k+s,:,4],'y');hold(True);plot(arr_0[k+s,:,5],'g');grid(True);
    #     title('X('+str(s)+') to X('+str(s+8)+') for no-peak data');legend(['r','i','g']);
    #
    # =============================================================================
    # making required data arrays
    arr_X=np.concatenate([arr_0,arr_1],axis=0)
    arr_Y=np.concatenate([arr_0_Y,arr_1_Y],axis=0)
    # shuffle arrays
    p = np.random.permutation(len(arr_X))
    arr_X1=arr_X[p,:,:]
    arr_Y1=arr_Y[p]
    times1=times.values[p]
    #np.random.shuffle(arr)
    arr_X1=arr_X1.reshape(arr_X1.shape[0],1,arr_X1.shape[1],arr_X1.shape[2]) # reshape for CNN (n,h,w,c)
    return (arr_X1,arr_Y1,times1,PTT)

def data_setup_predict(arr,w):
    arr0=np.array([arr[i:i+w,0] for i in range(len(arr)-w+1)])
    arr1=np.array([arr[i:i+w,1] for i in range(len(arr)-w+1)])
    arr2=np.array([arr[i:i+w,2] for i in range(len(arr)-w+1)])
    X=np.stack((arr0.reshape(arr0.shape+(1,)),arr1.reshape(arr1.shape+(1,)),arr2.reshape(arr2.shape+(1,))),axis=2)
    X_test=X.reshape(X.shape[0],1,X.shape[1],3)
    return X_test

def visualeyes(df_ECG,df_RA,df_LA):
    figure()
    ax=subplot(211)
    plot(df_ECG.index,df_ECG.Data1,'b')
    hold(True)
    plot(df_RA.index,df_RA.LED1,'r',df_LA.index,df_LA.LED1,'r--')
    hold(True)
    plot(df_RA.index,df_RA.LED2,'y',df_LA.index,df_LA.LED2,'y--')
    hold(True)
    plot(df_RA.index,df_RA.LED3,'g',df_LA.index,df_LA.LED3,'g--')
    legend(['ECG','RA_R','LA_R','RA_I','LA_I','RA_G','LA_G'])
    title('ECG & LED Data ('+str(len(df_ECG))+' ECG Samples)')
    grid(True)
    subplot(212,sharex=ax)
    plot(df_RA.index,df_RA.AccX,'r',df_LA.index,df_LA.AccX,'r--')
    hold(True)
    plot(df_RA.index,df_RA.AccY,'g',df_LA.index,df_LA.AccY,'g--')
    hold(True)
    plot(df_RA.index,df_RA.AccZ,'b',df_LA.index,df_LA.AccZ,'b--')
    legend(['RA_X','LA_X','RA_Y','LA_Y','RA_Z','LA_Z'])
    title('Acc Data')
    grid(True)
    return None

def Data_Process(extracted_raw_data_RA,extracted_raw_data_LA,PTT_past=[4,4],normaleyes=False):

    df_RA,coeff_RA,avg_RA,seq_chk_RA,time_chk_RA,pkt_chk_RA=Preprc(raw_data=extracted_raw_data_RA,name='RA',filt=True)


    if  (seq_chk_RA): # Test 1: All Seq nos. should be 1
        print('Test 1 failed.\n')
        return 0
    if  (time_chk_RA): # Test 2: All 3 time_stamps should be consistent
        print('Test 2 failed.\n')
        return 0
    if  (pkt_chk_RA): # Test 3: No. of packet errors shouldn't be more than 10%
        print('Test 3 failed.\n')
        return 0

    mean_slope=mean([coeff_RA[0]])
    mean_intercept=mean([coeff_RA[1]])
    mean_chk_slope=(mean_slope>40.2) | (mean_slope<39.8)
    mean_chk_intercept=mean_intercept<=1.5e12
    if (mean_chk_slope) | (mean_chk_intercept):   # Test 4: check for coefficients
        print('Test 4 failed.\n');time.sleep(0.5)
        return 0

    # find RMS Value of Acc
    RA_Acc_RMS=(df_RA.AccX**2+df_RA.AccY**2+df_RA.AccZ**2)**0.5

    #Standardize Data
    if normaleyes:
        df_RA=norma(df_RA)

    # manual clipping of data for 'clean' data
    #ll=pd.Timestamp(str(df_ECG.index[0]))  # decide this value after looking the data

    # merging data with labels
    df_union=df_RA
    df_union['Peak'] = np.ones(len(df_union.iloc[:,0]))
    df_union.dropna(thresh=2,inplace=True)
    data_test_RA=df_union

    # making centered dataframe with a window w and 0's:1's=6:5
    w=11
    df_RA_X,df_RA_Y,time_RA,ptt_RA=data_setup(df_union.copy(),w,PTT_past[0],RA_Acc_RMS[:len(df_union)])
    # print(np.isnan(df_RA_X).astype('int').sum()) # finding any Nan's
    # print(np.isnan(df_RA_Y).astype('int').sum()) # finding any Nan's
    del df_union

    return data_test_RA

def load_files(path,raw_file_nos):
    lst_df=[pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
    for n in range(3):
        #path2=path+'raw'+str(raw_file_nos[n])+'\\' # for windows
        path2=path+'raw'+str(raw_file_nos[n])+'/' # for mac
        filenames = sorted(glob.glob(path2+'*.gz'))
        for fil in filenames:
            df=pd.read_csv(fil,header=None,compression='gzip')
            lst_df[n]=lst_df[n].append(df)
            #lst_df[n].sort_values(0,inplace=True)
    # Enter concatenated raw csv file names
    arr_ECG=lst_df[0].values
    arr_RA=lst_df[1].values
    arr_LA=lst_df[2].values
    del lst_df
    return arr_ECG,arr_RA,arr_LA

def ll_ul_times(arr):
    thres=1000  # change the gap threshold (in ms.) as required NOTE: Change this treshold greatly should also change no. of consecutive missing samples tolerance=50
    arr_diff=np.diff(arr[:,0])
    arr_break_idx=mlab.find(arr_diff>thres)
    arr_ll=np.append(0,arr_break_idx+1);arr_ul=np.append(arr_break_idx,len(arr_diff))
    return arr[arr_ll,0],arr[arr_ul,0]



def get_test_data(arr_RA,PTT_past=[4,4]):
    list_test_RA=0
    list_test_RA_t=0
    data_test_RA=Data_Process(arr_RA,PTT_past,normaleyes=True)
    if data_test_RA is not 0 and data_test_RA is not None:
        list_test_RA=data_test_RA.values
        list_test_RA_t = data_test_RA.index

    return list_test_RA,list_test_RA_t







