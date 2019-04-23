# -*- coding: utf-8 -*-
import numpy as np






def mesh_normalize(data):
    mean = data[data.nonzero()].mean()
    std = data[data.nonzero()].std()
    data[data.nonzero()] = (data[data.nonzero()]-mean)/std
    #print(data[data.nonzero()].mean(),data[data.nonzero()].std())
    return data

def allStandardScaler(data):
    '''
    @input all input (clip*person*6)*10*128*6*6
    @output mesh normalization respectively
    '''
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data[i,j,k] = mesh_normalize(data[i,j,k])
    #data[:,:,:] = mesh_normalize(data[:,:,:])
    return data
    
def eegtodata(a,CLIP_NUM,trainname,testname,shuffle=False):
    
    b = dict()
    data = a[0]['data'][:,-7680:,:]
    label = a[0]['label']
    for i in range(1,CLIP_NUM): #12345
        data = np.append(data,a[i]['data'][:,-7680:,:],axis=0)
        label = np.append(label,a[i]['label'],axis=0)
    b['data'] = data
    b['label'] = label

    c=b['data'].shape
    print(c,b['label'].shape)
    eeg=dict()
    eeg['label'] = b['label']
    eeg['data'] = np.zeros(shape=(23*CLIP_NUM,7680,6,6))
    eeg['data'][:,:,0,1] = b['data'][:,:,0]
    eeg['data'][:,:,1,0] = b['data'][:,:,1]
    eeg['data'][:,:,1,2] = b['data'][:,:,2]
    eeg['data'][:,:,2,1] = b['data'][:,:,3]
    eeg['data'][:,:,3,0] = b['data'][:,:,4]
    eeg['data'][:,:,4,1] = b['data'][:,:,5]
    eeg['data'][:,:,5,2] = b['data'][:,:,6]
    eeg['data'][:,:,5,3] = b['data'][:,:,7]
    eeg['data'][:,:,4,4] = b['data'][:,:,8]
    eeg['data'][:,:,3,5] = b['data'][:,:,9]
    eeg['data'][:,:,2,4] = b['data'][:,:,10]
    eeg['data'][:,:,1,3] = b['data'][:,:,11]
    eeg['data'][:,:,1,5] = b['data'][:,:,12]
    eeg['data'][:,:,0,4] = b['data'][:,:,13] #修正 #为了避免引入多个情感 取最后60秒  65-393

    print(eeg['data'][0,0,1,5])
    print('=====')
    print(b['data'][0,0,12])
    ##验证
    eeg['data'].resize(CLIP_NUM*23*6,10,128,6,6)
    print(eeg['data'][0,0,0,1,5],eeg['data'].shape)
    #eeg['data'] = eeg['data'].transpose(0,1,3,4,2)
    #print(eeg['data'][0,0,1,5,0],eeg['data'].shape)



    #完成处理 data格式为 （clips=6*persons=23*subsamples=6）*10s*128*6*6  =-》828*10*128*6*6
    eeg_label = np.zeros(shape=(CLIP_NUM*23,6))
    print(eeg_label.shape)
    print(eeg['label'].size)
    for i in range(eeg['label'].size):
        eeg_label[i][:] = eeg['label'][i]

    eeg_label.resize(CLIP_NUM*23*6,1)
    print(eeg_label.shape)
    #print(eeg_label)
    eeg['label'] = eeg_label
    #       labelge格式为 clips*persons*subsamples*1=-》828*1
   # print(type(eeg['data']))
    #np.array(eeg)
    if shuffle is True:
        permutation = np.random.permutation(eeg['label'].size)
        shufflez_data = eeg['data'][permutation,:,:,:,:]
        shufflez_label = eeg['label'][permutation,:]
        eeg['data'] = shufflez_data
        eeg['label'] = shufflez_label

    eeg['data'].resize(CLIP_NUM*23,6,10,128,6,6)
    eeg['label'].resize(CLIP_NUM*23,6,1)
    train_data = eeg['data'][:,:5,:,:,:,:].copy()
    test_data = eeg['data'][:,5:,:,:,:,:].copy()
    train_label = eeg['label'][:,:5,:].copy()
    test_label = eeg['label'][:,5:,:].copy() #otherwise resize error 

    print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
    #print(type(train_data))
    train_data.resize(CLIP_NUM*23*5,10,128,6,6)
    test_data.resize(CLIP_NUM*23*1,10,128,6,6)
    train_label.resize(CLIP_NUM*23*5,1)
    test_label.resize(CLIP_NUM*23*1,1)
    
    # train_data = eeg['data'][:15*23*6]
    # train_label = eeg['label'][:15*23*6]
    # test_data = eeg['data'][15*23*6:]
    # test_label = eeg['label'][15*23*6:]
    print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
    
    allStandardScaler(train_data)
    print('train_norm')
    allStandardScaler(test_data)
    print('test_norm')

    train = dict()
    test = dict()
    train['data'] = train_data
    train['label'] = train_label
    test['data'] = test_data
    test['label'] = test_label
    np.array(train)
    np.array(test)
    np.save(trainname,train)
    np.save(testname,test)
    #print(a.shape)
    #print(a[17]['data'].shape)
    #print(a[17]['label'])
    # a = np.arange(10)
    #print(a[10]['data']) 
    #print('=========')

# b = a[np.newaxis,:]
# b[1,2] = 2
# print(b)

def eegtodata2(a,CLIP_NUM,trainname,testname,shuffle=False):
    b = dict()
    data = a[0]['data'][:,-7680:,:]
    label = a[0]['label']
    for i in range(1,CLIP_NUM): #12345
        data = np.append(data,a[i]['data'][:,-7680:,:],axis=0)
        label = np.append(label,a[i]['label'],axis=0)
    b['data'] = data
    b['label'] = label

    c=b['data'].shape
    print(c,b['label'].shape)
    eeg=dict()
    eeg['label'] = b['label']
    eeg['data'] = np.zeros(shape=(23*CLIP_NUM,7680,9,9))
    eeg['data'][:,:,1,3] = b['data'][:,:,0]
    eeg['data'][:,:,2,0] = b['data'][:,:,1]
    eeg['data'][:,:,2,2] = b['data'][:,:,2]
    eeg['data'][:,:,3,1] = b['data'][:,:,3]
    eeg['data'][:,:,4,0] = b['data'][:,:,4]
    eeg['data'][:,:,6,0] = b['data'][:,:,5]
    eeg['data'][:,:,8,3] = b['data'][:,:,6]
    eeg['data'][:,:,8,5] = b['data'][:,:,7]
    eeg['data'][:,:,6,8] = b['data'][:,:,8]
    eeg['data'][:,:,4,8] = b['data'][:,:,9]
    eeg['data'][:,:,3,7] = b['data'][:,:,10]
    eeg['data'][:,:,2,6] = b['data'][:,:,11]
    eeg['data'][:,:,2,8] = b['data'][:,:,12]
    eeg['data'][:,:,1,5] = b['data'][:,:,13]  #为了避免引入多个情感 取最后60秒  65-393

    print(eeg['data'][0,0,1,5])
    print(eeg['data'][0,0,2,8])
    print('=====')
    print(b['data'][0,0,13])
    print(b['data'][0,0,12])
    ##验证
    eeg['data'].resize(CLIP_NUM*23*6,10,128,9,9)
    print(eeg['data'][0,0,0,1,5],eeg['data'].shape)
    #eeg['data'] = eeg['data'].transpose(0,1,3,4,2)
    #print(eeg['data'][0,0,1,5,0],eeg['data'].shape)



    #完成处理 data格式为 （clips=6*persons=23*subsamples=6）*10s*128*6*6  =-》828*10*128*6*6
    eeg_label = np.zeros(shape=(CLIP_NUM*23,6))
    print(eeg_label.shape)
    print(eeg['label'].size)
    for i in range(eeg['label'].size):
        eeg_label[i][:] = eeg['label'][i]

    eeg_label.resize(CLIP_NUM*23*6,1)
    print(eeg_label.shape)
    #print(eeg_label)
    eeg['label'] = eeg_label
    #       labelge格式为 clips*persons*subsamples*1=-》828*1
   # print(type(eeg['data']))
    #np.array(eeg)
    if shuffle is True:
        permutation = np.random.permutation(eeg['label'].size)
        shufflez_data = eeg['data'][permutation,:,:,:,:]
        shufflez_label = eeg['label'][permutation,:]
        eeg['data'] = shufflez_data
        eeg['label'] = shufflez_label

    eeg['data'].resize(CLIP_NUM*23,6,10,128,9,9)
    eeg['label'].resize(CLIP_NUM*23,6,1)
    train_data = eeg['data'][:,:5,:,:,:,:].copy()
    test_data = eeg['data'][:,5:,:,:,:,:].copy()
    train_label = eeg['label'][:,:5,:].copy()
    test_label = eeg['label'][:,5:,:].copy() #otherwise resize error 

    print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
    #print(type(train_data))
    train_data.resize(CLIP_NUM*23*5,10,128,9,9)
    test_data.resize(CLIP_NUM*23*1,10,128,9,9)
    train_label.resize(CLIP_NUM*23*5,1)
    test_label.resize(CLIP_NUM*23*1,1)
    
    # train_data = eeg['data'][:15*23*6]
    # train_label = eeg['label'][:15*23*6]
    # test_data = eeg['data'][15*23*6:]
    # test_label = eeg['label'][15*23*6:]
    print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
    
    allStandardScaler(train_data)
    print('train_norm')
    allStandardScaler(test_data)
    print('test_norm')

    train = dict()
    test = dict()
    train['data'] = train_data
    train['label'] = train_label
    test['data'] = test_data
    test['label'] = test_label
    np.array(train)
    np.array(test)
    np.save(trainname,train)
    np.save(testname,test)

if __name__ == "__main__":
    #a = np.load('eeg_baseline.npy')
    #eegtodata(a,18,'eeg_baseline_train.npy','eeg_baseline_test.npy')
    #eegtodata(a,18,'eeg_baseline_train_shuffle_norm.npy','eeg_baseline_test_shuffle_norm.npy',shuffle=True)
    a = np.load('eeg_stimuli.npy')
    eegtodata(a,18,'eeg_stimuli_train_shuffle_norm.npy','eeg_stimuli_test_shuffle_norm.npy',shuffle=True)
    #eegtodata2(a,18,'eeg_stimuli_train_shuffle_norm99.npy','eeg_stimuli_test_shuffle_norm99.npy',shuffle=True)
    #train (2070,10,128,6,6) (2070,1)
    #test  (414,10,128,6,6)  (414,1)