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
    data = a[0]['data']
    label = a[0]['label']
    for i in range(1,CLIP_NUM): #12345
        data = np.append(data,a[i]['data'],axis=0)
        label = np.append(label,a[i]['label'],axis=0)
    b['data'] = data
    b['label'] = label

    c=b['data'].shape
    print(c,b['label'].shape)
    eeg=dict()
    eeg['label'] = b['label']
    eeg['data'] = np.zeros(shape=(23*CLIP_NUM,7680,6,6))
    eeg['data'][:,:,0,1] = b['data'][:,128:,0]
    eeg['data'][:,:,1,0] = b['data'][:,128:,1]
    eeg['data'][:,:,1,2] = b['data'][:,128:,2]
    eeg['data'][:,:,2,1] = b['data'][:,128:,3]
    eeg['data'][:,:,3,0] = b['data'][:,128:,4]
    eeg['data'][:,:,4,1] = b['data'][:,128:,5]
    eeg['data'][:,:,5,2] = b['data'][:,128:,6]
    eeg['data'][:,:,5,3] = b['data'][:,128:,7]
    eeg['data'][:,:,4,4] = b['data'][:,128:,8]
    eeg['data'][:,:,3,5] = b['data'][:,128:,9]
    eeg['data'][:,:,2,4] = b['data'][:,128:,10]
    eeg['data'][:,:,1,3] = b['data'][:,128:,11]
    eeg['data'][:,:,1,5] = b['data'][:,128:,12]
    eeg['data'][:,:,1,4] = b['data'][:,128:,13]

    print(eeg['data'][0,0,1,5])
    print('=====')
    print(b['data'][0,128,12])
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
if __name__ == "__main__":
    a = np.load('eeg_baseline.npy')
    #eegtodata(a,18,'eeg_baseline_train.npy','eeg_baseline_test.npy')
    eegtodata(a,18,'eeg_baseline_train_shuffle_norm.npy','eeg_baseline_test_shuffle_norm.npy',shuffle=True)


    #train (2070,10,128,6,6) (2070,1)
    #test  (414,10,128,6,6)  (414,1)