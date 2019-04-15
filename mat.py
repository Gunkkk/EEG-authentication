import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
path = '/home/gunkkk/下载//DREAMER.mat'
mat = sci.loadmat(path)
print(mat.keys())
a = mat['DREAMER']


def person(p_i):
    return a['Data'][0,0][0,p_i]

def baseline_person_clip(p_i,c_i):
    return person(p_i)['EEG'][0,0]['baseline'][0,0][c_i,:][0]

def stimuli_person_clip(p_i,c_i):
        return person(p_i)['EEG'][0,0]['stimuli'][0,0][c_i,:][0]

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



def baseline():
    baseline_clip = list()
    for clip_i in range(18):
        label = None
        data = None
        baseline_clip0 = dict()
        for  p_i in range(23):
                print('personddddd',p_i)
                if data is None:
                        data = baseline_person_clip(p_i,clip_i)[np.newaxis,:,:]
                else:
                        data = np.append(data,baseline_person_clip(p_i,clip_i)[np.newaxis,:,:],axis=0) #samples*channels
                if label is None:
                        label = np.array([p_i])
                else:
                        label = np.append(label,[p_i],axis=0)
        baseline_clip0['data'] = data
        baseline_clip0['label'] = label
        baseline_clip.append(baseline_clip0)
    np.array(baseline_clip)
    #print(baseline_clip.shape)
    np.save('eeg_baseline.npy',baseline_clip)


def stimuli():
    stimuli_clip = list()
    for clip_i in range(18):
        label = None
        data = None
        stimuli_clip0 = dict()
        for  p_i in range(23):
               # print('personddddd',p_i)
                if data is None:
                        data = stimuli_person_clip(p_i,clip_i)[np.newaxis,:,:]
                else:
                        data = np.append(data,stimuli_person_clip(p_i,clip_i)[np.newaxis,:,:],axis=0) #samples*channels
                if label is None:
                        label = np.array([p_i])
                else:
                        label = np.append(label,[p_i],axis=0)
        stimuli_clip0['data'] = data
        stimuli_clip0['label'] = label
        stimuli_clip.append(stimuli_clip0)
        print(clip_i)
    np.array(stimuli_clip)
    np.save('eeg_stimuli.npy',stimuli_clip) # 23*25472*14

if __name__ == "__main__":
        print(stimuli_person_clip(0,0).shape) # 25472*14
        print(baseline_person_clip(0,0).shape)
        #stimuli()