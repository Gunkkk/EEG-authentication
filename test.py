import torch
import torch.utils.data
import torch.nn as nn
import torchvision as tv
import numpy as np
from visualize import make_dot
import time 
import matplotlib.pyplot as plt
'''

!取消归一化   0.003 不收敛
    =>归一化+标准化

!取消cnndropout lr=0.003  batchsize=16 效果++   acc<>0.5 loss<>1.5
    poch:  99 | train loss: 1.3418 | test accuracy: 0.51
    Epoch:  99 | train loss: 2.0386 | test accuracy: 0.55

    =>取消cnndropout
!取消rnndropout lr=0.003  batchsize=16 ++  acc<>60 loss<>1

    =>取消rnndropout
!lr=0.01 不收敛

!取消maxpool lr=0.001 bs=16 loss <>2 
!取消maxpool lr =0.003 bs=16 loss<>1 acc<>75
    =>取消maxpool
!bs=64 lr=0.003 loss<>1.5 acc<>50 后面速度很慢

!bs=32 lr=0.003 loss<>1 acc<>70

!bs=8 lr=0.003 慢 loss<>0.8 震荡区间大 acc<>80

!bs=16 lr = 0.01 loss<>?1 acc<>65

    =>测试集分离出训练集 
!bs=16 lr = 0.03 不收敛 ??
!bs=16 lr =0.003 loss<>0.8 acc<>12


    =>更改训练集测试集集 15/3 
!bs=64 lr=0.003 test 无变化  需要选每个clip的一部分 train loss <>1.2 test loss<>4.4 acc<>0.045
    =>更改数据集 18*23*5size的train 18*23*1size的test 取1/6

!bs=64 lr=0.003 test loss <>3.5 train loss<>1.2 acc0.045

!bs=16 lr=0.003 test 照旧
？？？之前用的5/6的数据
    
    =>train /test 各自shuffle  TODO 调整超参数 全局shuffle 
!bs=16 lr=0.003 trainloss 震荡 test loss<>4 acc<>0.45

    => cnn  64 32 16 linear 64->32
!bs=16 lr=0.003 trainloss <>1.2震荡 test loss<>4  acc<>0.05
    
    =>全局shuffle
!bs=16 lr=0.003 train loss<>1.5 testloss<>3.4 acc<>0.05
!bs=256 lr=0.003 train 差不多
    =>之前的归一化标准化有问题 修正 test(): acc/step+1 修正
!bs=64 lr=0.003 不收敛? 
    =>批标准化后归一化可以收敛
!bs=128 lr=0.005 trainloss <>1.5 test loss <>4 acc<>0.12

!bs=16 lr=0.001 epoch 30 train loss<>2.8 震荡 test acc 0.13

    =>rnn 16 8
!bs=16 lr=0.001  震荡 
    =>rnn 32 16
    =>取消归一化 mesh各自标准化 nonzero
!bs=16 lr=0.001 train loss<>0.1 test loss 2 test acc<>55
!bs=128 lr=0.001 train loss<>0.8 test loss 1.7 test acc<>50
!bs=256 lr=0.001 train loss<>1.2 test loss<>1.9 test acc<>40
!bs=64 lr=0.001 train loss<>0.3 test loss <>1.6 acc<>58
!bs=64 lr=0.01 train loss <>0.1 震荡 test loss <>4.8 acc<>40
!bs=64 lr=0.005 过拟合 test loss <>3.5 test acc<> 45
    =>optim RMSprop
!bs=64 lr=0.003 
'''

train_path = 'eeg_baseline_train_shuffle_norm.npy'
test_path = 'eeg_baseline_test_shuffle_norm.npy'
BATCH_SIZE = 64
LR=0.003
EPOCH = 100
CUDA = False #TODO
def meanandvar(data):
    m = np.mean(data,axis=(0,1,2))
    n = np.std(data,axis=(0,1,2))
    #print(m,n)
    return m,n

def normalization(data):
    m = data.max()
    n = data.min() 
    data = (data-n)/(m-n)
    return data

def standardScaler(data,m,n):
    for i in range(128):
        data[:,i,:,:] = (data[:,i,:,:]-m[i])/n[i]
    return data

def tstandardScaler(data,m,n):
    for i in range(128):
        data[:,:,i,:,:] = (data[:,:,i,:,:]-m[i])/n[i]
    return data
    #print(m,n)
    
def mesh_normalize(data):
    mean = data[data.nonzero()].mean()
    std = data[data.nonzero()].std()
    data[data.nonzero()] = (data[data.nonzero()]-mean)/std
    print(data[data.nonzero()].mean(),data[data.nonzero()].std())
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

class DataSet(torch.utils.data.Dataset):
    aa =None
    def __init__(self,path,transform=None,shuffle=False):
        super(DataSet,self).__init__()
        self.aa = np.load(path).item()
       # print(self.aa['data'])
        # data = self.aa['data'][:5*23*6]
        # label = self.aa['label'][:5*23*6]
        # self.bb['data'] = data
        # self.bb['label'] = label 
        if shuffle is True:
            permutation = np.random.permutation(self.aa['label'].size)
            shufflez_data = self.aa['data'][permutation,:,:,:,:]
            shufflez_label = self.aa['label'][permutation,:]
            self.aa['data'] = shufflez_data
            self.aa['label'] = shufflez_label

        #self.meshnor = allStandardScaler(self.aa['data'])
        #self.normaa = normalization(self.aa['data'][:])
        #self.m,self.s = meanandvar(self.normaa)
      #  self.transform = tv.transforms.Compose([tv.transforms.Resize((9,9))])
    def __getitem__(self,index):
        data,label = self.aa['data'][index],self.aa['label'][index]
        #print(data.shape)
       # seqlen,features,cow,col = data.shape
        #print(batch_size,seqlen,features,cow,col)
        
       # data = torch.from_numpy(data)
        #data = torch.DoubleTensor(data)
       # dd = np.zeros(data.shape)
        
      # data => 10*128*6*6 （6*23）
       # data = standardScaler(data,self.m,self.s)
       # data = normalization(data)
        #print(data)
        #print(np.std(data,axis=(0,2,3)))
        #data = self.transform(data[:])
        #data = torch.DoubleTensor(data)
       # for i in range(np.size(data,0)):
        #data[:] = self.transform(data[:])
        #data = self.transform(data[:])
          #  dd[i] = normalization(data[i])
       # data = dd
        return data,label

    def __len__(self):
        #print(self.aa['label'].size)
        return self.aa['label'].size
    
traindata = DataSet(train_path)
testdata = DataSet(test_path)

#todo test_x 归一化标准化
#test_x = normalization(test_x[:])
#test_x = tstandardScaler(test_x[:],traindata.m,traindata.s)
trainloader = torch.utils.data.DataLoader(dataset=traindata,batch_size=BATCH_SIZE,shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testdata,batch_size=BATCH_SIZE,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
           # nn.Dropout2d(0.3),
           # nn.MaxPool2d(2)   
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
          #  nn.Dropout2d(0.3),
           # nn.MaxPool2d(2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
           # nn.Dropout2d(0.3)
            #nn.MaxPool2d(2)
        )
        self.linear = nn.Linear(64,32)
    def forward(self,x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
      #  print(x.shape)
        output = self.linear(x.view(x.size(0),-1))
        return output

class Combine(nn.Module):
    def __init__(self):
        super(Combine,self).__init__()
        self.cnn = CNN()
        self.rnn1 = nn.GRU(32,16,2,batch_first=True)
        #self.rnn2 = nn.GRU(16,8,1)
        self.linear = nn.Linear(160,23)

    def forward(self,input):
        #print(input.shape)
        batch_size,seqlen,features,cow,col = input.size()
        input = input.view(batch_size*seqlen,features,cow,col)
        coutput = self.cnn(input)
        r_input = coutput.view(batch_size,seqlen,-1)
        rout1,h1 = self.rnn1(r_input)
       # print(rout1.shape)   #[16,10,16]
       # out = self.rnn2(rout1,h1)
        
        out = self.linear(rout1.reshape(rout1.size(0),-1))  # contigious
        out = nn.functional.log_softmax(out,dim=1)
        return out


model = Combine()
model.double()
optimizer =torch.optim.RMSprop(model.parameters(),LR)
loss_fun = nn.NLLLoss()#CrossEntropyLoss()

'''
model visualization using visualize.py
'''
# xx = torch.randn(1,10,128,6,6).double()
# y = model(xx)
# g = make_dot(y)
# g.view()

#print(model)




#def test():


test_x,test_y = traindata.aa['data'][14*23*6:],traindata.aa['label'][14*6*23:]

def train(epoch):
    model.train()
    loss = 0.0
    accuracy= 0.0
    for step,(x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        y =y.long()
        #print(x.shape)
        out = model(x)
        #print(out)
        loss = loss_fun(out,y.squeeze())    
        loss.backward()
        optimizer.step()
        if step%5 ==0:
            #print(test_x.shape)
            tx = torch.DoubleTensor(test_x)
            ty = torch.LongTensor(test_y).squeeze()
            testout = model(tx)
            pred_y = testout.data.max(1)[1]
            #print(test_y.size(0))
            #print(testout.data.max(1)[1])
            #print(pred_y.shape)
            #print(test_y.shape)
            #print((pred_y == test_y).sum())
            accuracy = (pred_y == ty).sum().item()/ty.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.item(),'train acc:%.4f'%accuracy)
        
    return loss,accuracy

def test():
    model.eval()
    test_loss = 0.0
    acc = 0.0
    step = 0
    for step,(x,y) in enumerate(testloader):
        testout = model(x)
        y = y.long()
        #print(y.shape)
        pred_y = testout.data.max(1)[1]
        #print(pred_y.shape)
        acc += (pred_y == y.squeeze()).sum().item()/y.size(0)
        
        test_loss += loss_fun(testout,y.squeeze()).data.item()
        #print(acc,'nnnn',(pred_y == y.squeeze()).sum())
    #print(step)
    test_loss /= step+1 #len(testloader.dataset)
    acc /= step+1
    print('Test set average loss %.4f:'%test_loss,'accuracy %.4f'%acc)
    return test_loss,acc

if __name__ == "__main__":
    
    train_loss = np.zeros(shape=(EPOCH,2))
    test_loss = np.zeros(shape=(EPOCH,2))
    start = time.time()
    for epoch in range(EPOCH):
        train_loss[epoch,0],train_loss[epoch,1] = train(epoch)
        test_loss[epoch,0],test_loss[epoch,1]= test()
    end = time.time()
    print('time:',end-start)

    plt.subplot(2,1,1)
    plt.plot(np.arange(EPOCH),train_loss[:,0],'b-',label='train_loss')
    plt.plot(np.arange(EPOCH),test_loss[:,0],'r-',label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.arange(EPOCH),test_loss[:,1],'b-',label='test_acc')
    plt.plot(np.arange(EPOCH),train_loss[:,1],'r-',label='train_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('bs'+str(BATCH_SIZE)+'lr'+str(LR)+'time'+str(end-start)+'.png',format='png')
    plt.show()