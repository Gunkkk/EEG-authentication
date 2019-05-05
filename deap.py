# -*- coding: utf-8 -*-
import pickle
import numpy as np

def mesh_normalize(data):
    #print(data[data.nonzero()].mean(),data[data.nonzero()].std())
    mean = data[data.nonzero()].mean()
    std = data[data.nonzero()].std()
    data[data.nonzero()] = (data[data.nonzero()]-mean)/std
  #  print(data[data.nonzero()].mean(),data[data.nonzero()].std())
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

def deaptonpy32(data=None,personindex=0):
    ### 取后60s
    if data is None:
        return 
    data_1 = np.zeros(shape=(40,40,7680))
    data_1 = data[:,:,-7680:]
    data_transpose = data_1.transpose(0,2,1) #trial*data*channel
    print(data_transpose.shape)
    data_premesh = data_transpose.copy()
    data_premesh.resize(40*6,10,128,40)
    print(data_premesh.shape)
    data_fin = np.zeros(shape=(40*6,10,128,9,9))
    #labels
    label_fin = np.zeros(shape=data_fin.shape[0])
    label_fin[:] = personindex
    ###mesh trains
    data_fin[:,:,:,0,3] = data_premesh[:,:,:,0]
    data_fin[:,:,:,1,3] = data_premesh[:,:,:,1]
    data_fin[:,:,:,2,2] = data_premesh[:,:,:,2]
    data_fin[:,:,:,2,0] = data_premesh[:,:,:,3]
    data_fin[:,:,:,3,1] = data_premesh[:,:,:,4]
    data_fin[:,:,:,3,3] = data_premesh[:,:,:,5]
    data_fin[:,:,:,4,2] = data_premesh[:,:,:,6]
    data_fin[:,:,:,4,0] = data_premesh[:,:,:,7]
    data_fin[:,:,:,5,1] = data_premesh[:,:,:,8]
    data_fin[:,:,:,5,3] = data_premesh[:,:,:,9]
    data_fin[:,:,:,6,2] = data_premesh[:,:,:,10]
    data_fin[:,:,:,6,0] = data_premesh[:,:,:,11]
    data_fin[:,:,:,7,3] = data_premesh[:,:,:,12]
    data_fin[:,:,:,8,3] = data_premesh[:,:,:,13]
    data_fin[:,:,:,8,4] = data_premesh[:,:,:,14]
    data_fin[:,:,:,6,4] = data_premesh[:,:,:,15]
    data_fin[:,:,:,0,5] = data_premesh[:,:,:,16]
    data_fin[:,:,:,1,5] = data_premesh[:,:,:,17]
    data_fin[:,:,:,2,4] = data_premesh[:,:,:,18]
    data_fin[:,:,:,2,6] = data_premesh[:,:,:,19]
    data_fin[:,:,:,2,8] = data_premesh[:,:,:,20]
    data_fin[:,:,:,3,7] = data_premesh[:,:,:,21]
    data_fin[:,:,:,3,5] = data_premesh[:,:,:,22]
    data_fin[:,:,:,4,4] = data_premesh[:,:,:,23]
    data_fin[:,:,:,4,6] = data_premesh[:,:,:,24]
    data_fin[:,:,:,4,8] = data_premesh[:,:,:,25]
    data_fin[:,:,:,5,7] = data_premesh[:,:,:,26]
    data_fin[:,:,:,5,5] = data_premesh[:,:,:,27]
    data_fin[:,:,:,6,6] = data_premesh[:,:,:,28]
    data_fin[:,:,:,6,8] = data_premesh[:,:,:,29]
    data_fin[:,:,:,7,5] = data_premesh[:,:,:,30]
    data_fin[:,:,:,8,5] = data_premesh[:,:,:,31]
    
    return data_fin,label_fin

def deaptonpy14_99(data=None,personindex=0):
    if data is None:
        return 
     ### 取后60s
    data_1 = np.zeros(shape=(40,40,7680))
    data_1 = data[:,:,-7680:]
    data_transpose = data_1.transpose(0,2,1) #trial*data*channel
    print(data_transpose.shape)
    data_premesh = data_transpose.copy()
    data_premesh.resize(40*6,10,128,40)
    print(data_premesh.shape)
    data_fin = np.zeros(shape=(40*6,10,128,9,9))
    #labels
    label_fin = np.zeros(shape=data_fin.shape[0])
    label_fin[:] = personindex
    ###mesh trains
   # data_fin[:,:,:,0,3] = data_premesh[:,:,:,0]
    data_fin[:,:,:,1,3] = data_premesh[:,:,:,1]
    data_fin[:,:,:,2,2] = data_premesh[:,:,:,2]
    data_fin[:,:,:,2,0] = data_premesh[:,:,:,3]
    data_fin[:,:,:,3,1] = data_premesh[:,:,:,4]
  #  data_fin[:,:,:,3,3] = data_premesh[:,:,:,5]
  #  data_fin[:,:,:,4,2] = data_premesh[:,:,:,6]
    data_fin[:,:,:,4,0] = data_premesh[:,:,:,7]
   # data_fin[:,:,:,5,1] = data_premesh[:,:,:,8]
   # data_fin[:,:,:,5,3] = data_premesh[:,:,:,9]
   # data_fin[:,:,:,6,2] = data_premesh[:,:,:,10]
    data_fin[:,:,:,6,0] = data_premesh[:,:,:,11]
   # data_fin[:,:,:,7,3] = data_premesh[:,:,:,12]
    data_fin[:,:,:,8,3] = data_premesh[:,:,:,13]
   # data_fin[:,:,:,8,4] = data_premesh[:,:,:,14]
   # data_fin[:,:,:,6,4] = data_premesh[:,:,:,15]
   # data_fin[:,:,:,0,5] = data_premesh[:,:,:,16]
    data_fin[:,:,:,1,5] = data_premesh[:,:,:,17]
   # data_fin[:,:,:,2,4] = data_premesh[:,:,:,18]
    data_fin[:,:,:,2,6] = data_premesh[:,:,:,19]
    data_fin[:,:,:,2,8] = data_premesh[:,:,:,20]
    data_fin[:,:,:,3,7] = data_premesh[:,:,:,21]
   # data_fin[:,:,:,3,5] = data_premesh[:,:,:,22]
   # data_fin[:,:,:,4,4] = data_premesh[:,:,:,23]
   # data_fin[:,:,:,4,6] = data_premesh[:,:,:,24]
    data_fin[:,:,:,4,8] = data_premesh[:,:,:,25]
   # data_fin[:,:,:,5,7] = data_premesh[:,:,:,26]
   # data_fin[:,:,:,5,5] = data_premesh[:,:,:,27]
   # data_fin[:,:,:,6,6] = data_premesh[:,:,:,28]
    data_fin[:,:,:,6,8] = data_premesh[:,:,:,29]
   # data_fin[:,:,:,7,5] = data_premesh[:,:,:,30]
    data_fin[:,:,:,8,5] = data_premesh[:,:,:,31]
    
    return data_fin,label_fin

def deaptonpy14_66(data,personindex):
    if data is None:
        print('None !!!!!')
        return 
     ### 取后60s
    data_1 = np.zeros(shape=(40,40,7680))
    print(data_1[1,1,100],'....')
    data_1 = data[:,:,-7680:]
    data_transpose = data_1.transpose(0,2,1) #trial*data*channel
    print(data_transpose[1,100,1])
    data_premesh = data_transpose.copy()
    data_premesh.resize(40*6,10,128,40)
    print(data_premesh.shape)
    data_fin = np.zeros(shape=(40*6,10,128,6,6))
    #labels
    label_fin = np.zeros(shape=(data_fin.shape[0],1))
    label_fin[:] = personindex
    ###mesh trains
   # data_fin[:,:,:,0,3] = data_premesh[:,:,:,0]
    data_fin[:,:,:,0,1] = data_premesh[:,:,:,1]
    data_fin[:,:,:,1,2] = data_premesh[:,:,:,2]
    data_fin[:,:,:,1,0] = data_premesh[:,:,:,3]
    data_fin[:,:,:,2,1] = data_premesh[:,:,:,4]
  #  data_fin[:,:,:,3,3] = data_premesh[:,:,:,5]
  #  data_fin[:,:,:,4,2] = data_premesh[:,:,:,6]
    data_fin[:,:,:,3,0] = data_premesh[:,:,:,7]
   # data_fin[:,:,:,5,1] = data_premesh[:,:,:,8]
   # data_fin[:,:,:,5,3] = data_premesh[:,:,:,9]
   # data_fin[:,:,:,6,2] = data_premesh[:,:,:,10]
    data_fin[:,:,:,4,1] = data_premesh[:,:,:,11]
   # data_fin[:,:,:,7,3] = data_premesh[:,:,:,12]
    data_fin[:,:,:,5,2] = data_premesh[:,:,:,13]
   # data_fin[:,:,:,8,4] = data_premesh[:,:,:,14]
   # data_fin[:,:,:,6,4] = data_premesh[:,:,:,15]
   # data_fin[:,:,:,0,5] = data_premesh[:,:,:,16]
    data_fin[:,:,:,0,4] = data_premesh[:,:,:,17]
   # data_fin[:,:,:,2,4] = data_premesh[:,:,:,18]
    data_fin[:,:,:,1,3] = data_premesh[:,:,:,19]
    data_fin[:,:,:,1,5] = data_premesh[:,:,:,20]
    data_fin[:,:,:,2,4] = data_premesh[:,:,:,21]
   # data_fin[:,:,:,3,5] = data_premesh[:,:,:,22]
   # data_fin[:,:,:,4,4] = data_premesh[:,:,:,23]
   # data_fin[:,:,:,4,6] = data_premesh[:,:,:,24]
    data_fin[:,:,:,3,5] = data_premesh[:,:,:,25]
   # data_fin[:,:,:,5,7] = data_premesh[:,:,:,26]
   # data_fin[:,:,:,5,5] = data_premesh[:,:,:,27]
   # data_fin[:,:,:,6,6] = data_premesh[:,:,:,28]
    data_fin[:,:,:,4,4] = data_premesh[:,:,:,29]
   # data_fin[:,:,:,7,5] = data_premesh[:,:,:,30]
    data_fin[:,:,:,5,3] = data_premesh[:,:,:,31]
    
    print(data_fin[1,2,3,0,4])
  #  print(data_fin[2,2,3,4,4])
  #  print(label_fin[2])
  #  print(label_fin[1])
    print('=========================',data_transpose[0,1280+2*128+3,17])
    print(data_premesh[1,2,3,17])
  #  print(data_premesh[2,2,3,29])

    return data_fin,label_fin


def tomesh(filename,mesh_size,channel_num,shuffle=True):
    
    
    dataset = dict()
    data = None
    label = None
    for i in range(1,33):
        index = i-1
        if i<10:
            i = '0'+str(i)
        path = './data_preprocessed_python/s%s.dat'%i
        a = pickle.load(open(path,'rb'),encoding='latin1')
       # print(a.keys())
        #print(type(a['data']))
        print(a['data'].shape,a['labels'].shape)  #(40,40,8064)video/trial x channel x data (40,4)
        if channel_num == 32:
            idata,ilabel = deaptonpy32(a['data'],index)
        elif channel_num ==14 and mesh_size == 9:
            idata,ilabel = deaptonpy14_99(a['data'],index)
        elif mesh_size == 6:
            idata,ilabel = deaptonpy14_66(a['data'],index)
            print('14_66:s',index)
        else:   return 
        #idata,ilabel = func(a['data'],index)
        print(idata.shape,ilabel.shape)
        if data is None:
            data = idata
            label = ilabel
        else:
            data = np.append(data,idata,axis=0)
            label = np.append(label,ilabel,axis=0)
        print('subject%s'%i,'ok')

    if shuffle is True:
        permutation = np.random.permutation(label.size)
       # print(permutation.shape)
       # x = permutation[1]
       # aadata = data[x,1,1,0,4]
       # aalabel = label[x]
        shufflez_data = data[permutation,:,:,:,:]
        shufflez_label = label[permutation]
        data = shufflez_data
        label = shufflez_label
    print('shuffle ')
   # print(data[1,1,1,0,4],aadata)
    #print(label[1],aalabel)
    print('============',data.shape)
    dataa = allStandardScaler(data)
   # print(data[1,1,1,:,:])
    dataset['data'] = dataa
    dataset['label'] = label
    print('==========save')
    np.array(dataset)
    output = open(filename,'wb')
  #  pickle.dump(dataset,output)
    output.close()
   # np.save(filename,dataset)

if __name__ == "__main__":
    tomesh('deap_6_14_11.npy',6,14,shuffle=True) #9/6 32/14


    