import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
path = '/home/gunkkk/下载//DREAMER.mat'
mat = sci.loadmat(path)
print(mat.keys())
a = mat['DREAMER']


def person(p_i):
    return a['Data'][0,0][0,p_i]

def person_clip(p_i,c_i):
    return person(p_i)['EEG'][0,0]['baseline'][0,0][c_i,:][0]

#print(person_clip(1,1))
print('========')
#print(person_clip(1,3))
#b = np.squeeze(a['Data'])
#print(a['Data'][0,0][0,1]['EEG'][0,0]['baseline'][0,0][1,:])

#c= a['Data'][0,0][0,1]['EEG'][0,0]['baseline'][0,0][1,:] # first 1 => person 1 ,sec2=> file clip 1
#print(c[0].shape) # samples*channels
#print(person_clip(3,0))
#print(mat['__header__'])
#print(mat[0,0])
#print(b[0,0])
# for clip_i in range(18):
#     plt.plot(np.arange(0,14),person_clip(0,clip_i)[0,:])

# plt.show()
baseline_clip = list()
for clip_i in range(18):
    label = None
    data = None
    baseline_clip0 = dict()
    for  p_i in range(23):
        print('personddddd',p_i)
        if data is None:
                data = person_clip(p_i,clip_i)[np.newaxis,:,:]
        else:
                data = np.append(data,person_clip(p_i,clip_i)[np.newaxis,:,:],axis=0) #samples*channels
        if label is None:
                label = np.array([p_i])
        else:
                label = np.append(label,[p_i],axis=0)
    baseline_clip0['data'] = data
    baseline_clip0['label'] = label
    baseline_clip.append(baseline_clip0)
np.array(baseline_clip)
np.save('eeg_all.npy',baseline_clip)
