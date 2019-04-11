import numpy as np
a = np.load('eeg_all.npy')

print(a.shape)
print(a[17]['data'].shape)
print(a[17]['label'])
# a = np.arange(10)
#print(a[10]['data']) 
print('=========')
b = dict()
data = a[0]['data']
label = a[0]['label']
for i in range(1,6): #12345
    data = np.append(data,a[i]['data'],axis=0)
    label = np.append(label,a[i]['label'],axis=0)
b['data'] = data
b['label'] = label

c=b['data'].shape
print(c,b['label'].shape)
eeg=dict()
eeg['label'] = b['label']
eeg['data'] = np.zeros(shape=(138,7680,6,6))
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
eeg['data'].resize(6*23*6,10,128,6,6)
print(eeg['data'][0,0,0,1,5],eeg['data'].shape)
eeg['data'] = eeg['data'].transpose(0,1,3,4,2)
print(eeg['data'][0,0,1,5,0],eeg['data'].shape)



#完成处理 data格式为 （clips=6*persons=23*subsamples=6）*10s*6*6*128  =-》828*10*6*6*128
eeg_label = np.zeros(shape=(6*23,6))
print(eeg_label.shape)
print(eeg['label'].size)
for i in range(eeg['label'].size):
    eeg_label[i][:] = eeg['label'][i]

eeg_label.resize(6*23*6,1)
print(eeg_label.shape)
print(eeg_label)
eeg['label'] = eeg_label
#       labelge格式为 clips*persons*subsamples*1=-》828*1
np.array(eeg)
np.save('eeg_data.npy',eeg)

# b = a[np.newaxis,:]
# b[1,2] = 2
# print(b)