import numpy as np
import scipy.signal as ff

def CAR(clipdata):
    '''
    average to common filter
    '''
    for i in range(clipdata.shape[0]):
        mean = clipdata[i,:,:].mean() # average value of all channel  
        #mean /= clipdata.shape[0]
       # print(mean)
        for j in range(clipdata.shape[2]):
            data = clipdata[i,:,j]
            data = data - mean
            clipdata[i,:,j] = data
    #print(clipdata.shape)
    return clipdata

def bandpass(data,order,low,high,sample):
    '''
    band pass 
    '''
    b = ff.firwin(order,[low/sample*2,high/sample*2],pass_zero=False)
    a=1
    #b,a = ff.butter(order,[low/sample*2,high/sample*2],'bandpass')
    for personi in range(data.shape[0]):
        for channeli in range(data.shape[2]):
            data[personi,:,channeli] = ff.filtfilt(b,a,data[personi,:,channeli])  
    #print('pi',personi,'ci',channeli)  
    return data

if __name__ == "__main__":
    dataset = np.load('eeg_stimuli.npy')
    dataset2 = list()
    for clipi in range(18):
        dataset2.append(dict())
        dataset2[clipi]['data'] = dataset[clipi]['data'][:,-7680:,:] 
        dataset2[clipi]['label'] = dataset[clipi]['label']
        print(dataset2[clipi]['label'].shape)
        dataset2[clipi]['data'] = bandpass(dataset2[clipi]['data'],61,3,47,128)
        dataset2[clipi]['data'] = CAR(dataset2[clipi]['data'])
        print('===clip%iover!'%clipi)
    np.array(dataset2)
    np.save('eeg_processed_fir.npy',dataset2)