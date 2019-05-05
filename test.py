# -*- coding: utf-8 -*-
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
!bs=64 lr=0.005  test loss <>3.5 test acc<> 45
    =>optim RMSprop
!bs=64 lr=0.003  test loss <> 2.5 test acc<>50
    =>adam
@4/17 TODO dropout l2 
    =>cnn dropout 0.3
!bs=64 lr=0.003 train loss<>0.6 acc <>70 test loss<>1.6 acc<>55
    =>rnn dropout 0.3  cancel cnn dropout 合适
!bs=64 lr=0.003 train loss<>0.3 acc<>85 test loss<>2.0 acc<>60
    =>weight_decay =0.01
!bs=64 lr=0.008 train epoch30 train acc<>15 test acc<>8 !!cancel
    =>weight decay=0.001
!bs=64 lr=0.008 train loss<>0.6 trainacc<>75 test loss<>2.0 acc<>50 
    =>weight decay=0.005 rnn dropout=0.2 =>wd过高 
!bs=64 lr=0.008 train loss<>1.8 train acc<>30 test loss<>2.0 acc<>33  =>可以继续 
!bs=256 lr=0.008 epoch30 train acc<>10 test acc<>5 !!cancel
!bs=64 lr=0.01 train loss<>1.5 acc<>40 test loss<>2.2 acc<>30
    =>weight decay = 0.005 ??rnn dropout=0.8
!bs=64 lr=0.01 !!cancel
!bs=64 lr=0.005 !!cancel
    =>weight decay=0.0005 3*cnn dropout =0.5 ??rnn dropout=0.8
!bs=64 lr=0.005 cancel
    =>weight decay=0.0005 1*cnn dropout =0.5 ??rnn dropout=0.8
!bs=64 lr=0.005 testacc>trainacc ??train loss=<>1.5 acc<>35 test train<>1.7 acc<>50
    =>weight decay = 0.0005  rnn dropout=0.8 ??should be 0.2
!bs=64 lr=0.005 train loss<>1.2 acc 45 test loss<>1.8 acc<>45
    =>weight decay = 0.0001  
!bs=64 lr=0.005 train 过拟合
    =>weigh decay = 0.0005 rnn dropout = 0.2
!bs=64 lr=0.005 train loss<>0.3 acc<>90 test loss<>2.2 acc <>55

@4/18 TODO cnn 128 64 32 dsc
    =>weight decay =0.0005 cnn dp0.2 rnn dp0.3 
!bs=64 lr=0.005 train loss <>1 acc<>60 test loss<>1.35 acc<>62 =>EPOCH 200
                            <>0.6   <>70        <>1.3      <>62
    =>weight decay =0.001 cnn dp0.2 rnn dp0.3 
!bs=64 lr=0.005 同上
    =>weight decay = 0.003 rnn dp=0.3
!bs=64 lr=0.005 过拟合 震荡train loss<>0.7 acc<>70 test loss<>1.6 acc<>55
    =>weight decay = 0.001 cnn 128 64 32 rnn dp=0.3 
!bs=64 lr=0.005 过拟合 速度慢 train loss<>0.5 acc<>80 test loss<>1.7 acc<60
    =>weight decay=0.001  cnn 128 64 32 dp=0.5 rnndp=0.3
!bs=64 lr=0.005 速度慢 acc 40
    =>weight decay=0.001  cnn 128 64 32 dp=0.3 rnndp=0.3
!bs=64 lr=0.005 train loss<>1.2 acc<>42 test oss<>1.3 acc<>55 慢
    => weight decay=0.001 DSC rnndp0.3
!bs=64 lr=0.005 train loss<>0.2 acc<>90 test loss<>1.5 acc<>60
    => weight decay=0.001 DSC l2dp0.3 rnndp0.3 及其慢
!bs=16 lr=0.005           <>0.6  <>60        <>1.5        <>60 
    =>weight decay=0.001 rnn16 8 dp0.3 

@4/19 交叉验证
    =>dsc去掉l2 weight decay=0.001 rnn16 8 dp0.3 
!bs=64 lr=0.005 过拟合  震荡   <>0.8   <>60    <>1.9     <>55
    =>交叉验证 cnn 64 32 16 weight decay=0.001 rnn32 16 dp0.3 
!bs=64 lr=0.005 
Final test loss:2.7004 acc0.3645
    => weight = 0.001 rnn 32 16 
    换dataest 和之前一一样 
    缩小dataset 效果差
    更换一段视频同一部分的数据
    1/10 test 过拟合依旧严重  
    =>更换stimuli
    =>0.3 0.3 0.001 收敛慢 
    =>0 0.3 0.001 cha
    =>dsc 0.2 0.2 0.001 交换bnrelu  收敛快 过拟合严重
    =>    0.5(无效) 0.3 .0005 收敛快 过拟合
    =>dsc + l2 0.5 0.5 0.001  过拟合收敛慢
====>之前的数据预处理shuffle无效 更正
    =>dsc + l2 0.3 0.3 0.0005  
    =>                          99  test acc <>70 loss<>1 效果提升 过拟合
    =>dsc + l2 0.5 0.3 0.001   同上             75 
    =>dsc- l2      0.3 0.001   收敛效果更好 效果差
    =>dsc- l2      0.5 0.003   过拟合 效果差 几乎同上
    =>dsc- l2      0.3 0.003 rnn 3ceng 无效果
                                 1ceng acc 70 效果好 **
                                    =>9*9        
                                    =>RELU cancel 效果差
                                    =>1*1 效果差
                                    =>sgd+momentum
                                    =>一层dropout的cnn 64-32-16 效果差
                                    =>stimuli baseline混合 提升一点点
                                    =>wd=0.0005 稳定收敛  65<>
                                    =>wd=0.05 收敛慢
                            pwdp    =>wd=0.01 66 效果不错 acc70+ <75 epoch 100未收敛完！！
                                    =>wd=0.01 66 添加高斯噪声
                                        =>128*s*s 效果差
                                        =>s*s 效果差 
                                    =>picture 01化 变差
                                    =>01+noise 不收敛
                                    =>LSTM <gru
                                    =>GRU 16 8 差
                                    =>GRU 64 32 差不多
                       pwdp dwdrop  =>差
                            maxpool => 70
                                    =>baseline  <60一般
    =>之前的dreamer标准化无效重新标准化
    =>
'''         
EXTRA_NAME=''
train_path = 'eeg_stimuli_processed_fir.npy'#clip_peron_channel.npy'#'deap_6_14.npy'
test_path = None#'eeg_stimuli_test_shuffle_norm99.npy'
MESH_SIZE = 6
TYPE_NUM = 55
BATCH_SIZE = 64
LR=0.001
EPOCH = 10
CUDA = False
RNN_DROP = 0.3
CNN_DROP = 0.5
CNN_FILTERS=[64,32,16]
DSC_FILTERS=[128,64]
RNN_FEA = [32,16]
WD=0.01
NOISE = False
ONE_ZERO =False
SEQ=10

def addGaussianNoise(data):
    '''
    add gaussian noise for 2 or 3 axis ?
    '''
    # for i in range(data.shape[0]):
    #     noise = np.random.normal(0,0.1,(data.shape[1],data.shape[2]))
    #     data[i] += noise
    noise = np.random.normal(0,0.1,(data.shape[0],data.shape[1],data.shape[2]))
    data += noise
    return data

def meanandvar(data):
    m = np.mean(data,axis=(0,1,2))
    n = np.std(data,axis=(0,1,2))
    #print(m,n)
    return m,n

def normalization(data):
    m = data.max()
    n = data[data.nonzero()].min() 
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

class CrossDataSet(torch.utils.data.Dataset):
    def __init__(self,path1,path2,k,index,cross=True):
        super(CrossDataSet,self).__init__()
        self.alltrain = np.load(path1,encoding='latin1').item()
        if path2 is not None:

            self.alltest = np.load(path2,encoding='latin1').item()

            self.alldata = np.concatenate((self.alltrain['data'],self.alltest['data']),axis=0)
            self.alllabel = np.concatenate((self.alltrain['label'],self.alltest['label']),axis=0)
        else:
            self.alldata = self.alltrain['data']
            self.alllabel = self.alltrain['label']
        self.k = k
        self.index = index
        self.cross = cross
        self.slice = self.alllabel.size/k
        self.data = np.concatenate((self.alldata[:int(self.index*self.slice)],self.alldata[int((self.index+1)*self.slice):]),axis=0)
        self.label = np.concatenate((self.alllabel[:int(self.index*self.slice)],self.alllabel[int((self.index+1)*self.slice):]),axis=0)
        self.testdata = self.alldata[int(self.index*self.slice):int((self.index+1)*self.slice)]
        self.testlabel = self.alllabel[int(self.index*self.slice):int((self.index+1)*self.slice)]
        #self.noise = NOISE
        print(self.data.shape,self.label.shape)
        #print(self.testdata.shape,self.testlabel.shape)
    def __getitem__(self,index):
        data = self.data
        label = self.label

        if self.cross is False:
            data = self.alldata
            label = self.alllabel

        if NOISE is True:
            for i in range(data.shape[1]):#seq
                data[index][i] = addGaussianNoise(data[index][i])

        if ONE_ZERO is True:
            for i in range(data.shape[1]):
                data[index][i] = normalization(data[index][i]) 

        return data[index],label[index]

    def __len__(self):
        if self.cross is True:
            return self.label.size
        else:
            return self.alllabel.size

class CrossTest(torch.utils.data.Dataset):
    def __init__(self,data,label):
        super(CrossTest,self).__init__()
        self.data = data
        self.label = label
    def __getitem__(self,index):
        data = self.data 
        label = self.label
        if NOISE is True:
            for i in range(data.shape[1]):#seq
                data[index][i] = addGaussianNoise(data[index][i])

        if ONE_ZERO is True:
            for i in range(data.shape[1]):
                data[index][i] = normalization(data[index][i]) 

        return data[index],label[index]
    def __len__(self):
        return self.label.size

'''
quit
'''
class DataSet(torch.utils.data.Dataset):
    def __init__(self,path,transform=None,shuffle=False,test=False):
        super(DataSet,self).__init__()
        self.aa = np.load(path,encoding='latin1').item()
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
        half = int(self.aa['label'].size /2)
        test = int(self.aa['label'].size /5)
        self.testdata = self.aa['data'][-test:]
        self.testlabel = self.aa['label'][-test:]
        self.data = self.aa['data'][:half]
        self.label = self.aa['label'][:half]
        if test is True:
            self.aa['data'] = self.testdata
            self.aa['label'] = self.testlabel
        else:
            self.aa['data'] = self.data
            self.aa['label'] = self.label
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
    
#traindata = DataSet(train_path)
#testdata = DataSet(test_path)

#todo test_x 归一化标准化
#test_x = normalization(test_x[:])
#test_x = tstandardScaler(test_x[:],traindata.m,traindata.s)

#trainloader = torch.utils.data.DataLoader(dataset=traindata,batch_size=BATCH_SIZE,shuffle=True)
#testloader = torch.utils.data.DataLoader(dataset=testdata,batch_size=BATCH_SIZE,shuffle=True)

class DSC(nn.Module):
    '''
    Depthwise Separable Convolution
    '''
    def __init__(self):
        super(DSC,self).__init__()
        self.depth_wise = nn.Sequential(
            nn.Conv2d(in_channels=DSC_FILTERS[0],out_channels=DSC_FILTERS[0],kernel_size=3,groups=DSC_FILTERS[0]),
            nn.ReLU(),
            nn.BatchNorm2d(DSC_FILTERS[0]),
            nn.Dropout2d(CNN_DROP)
        )
        self.point_wise = nn.Sequential(
            nn.Conv2d(in_channels=DSC_FILTERS[0],out_channels=DSC_FILTERS[1],kernel_size=1),     
            nn.ReLU(),
            nn.BatchNorm2d(DSC_FILTERS[1]),
            nn.Dropout2d(CNN_DROP),
          # nn.MaxPool2d(2)
        ) # =>64*4*4
        self.l2 = nn.Sequential(
            nn.Conv2d(64,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(CNN_DROP)
        )
        self.fc = nn.Linear(DSC_FILTERS[1]*(MESH_SIZE-2)*(MESH_SIZE-2),RNN_FEA[0])
    def forward(self,input):
        x = self.depth_wise(input)
        x = self.point_wise(x)
        #x = self.l2(x)
        out = self.fc(x.view(x.size(0),-1))
       # print(out.size())
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=CNN_FILTERS[0],kernel_size=3,padding=1),
            
            nn.ReLU(),
            nn.BatchNorm2d(CNN_FILTERS[0]),
            nn.Dropout2d(CNN_DROP),
           # nn.MaxPool2d(2)   
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=CNN_FILTERS[0],out_channels=CNN_FILTERS[1],kernel_size=3), 
            
            nn.ReLU(),
            nn.BatchNorm2d(CNN_FILTERS[1]),
            nn.Dropout2d(CNN_DROP),
           # nn.MaxPool2d(2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=CNN_FILTERS[1],out_channels=CNN_FILTERS[2],kernel_size=3),
            
            nn.ReLU(),
            nn.BatchNorm2d(CNN_FILTERS[2]),
            nn.Dropout2d(CNN_DROP)
            #nn.MaxPool2d(2)
        )
        self.linear = nn.Linear((MESH_SIZE-2)*(MESH_SIZE-2)*CNN_FILTERS[1],RNN_FEA[0])
    def forward(self,x):
        x = self.l0(x)
        x = self.l1(x)
       # x = self.l2(x)
      #  print(x.shape)
        output = self.linear(x.view(x.size(0),-1))
        return output

class Combine(nn.Module):
    def __init__(self):
        super(Combine,self).__init__()
        self.cnn = CNN()
        self.dsc = DSC()
        self.rnn1 = nn.GRU(RNN_FEA[0],int(RNN_FEA[1]/2),1,batch_first=True,dropout=RNN_DROP,bidirectional=True)
        #self.rnn2 = nn.GRU(16,8,1)
        self.linear = nn.Linear(SEQ*RNN_FEA[1],TYPE_NUM)

    def forward(self,input):
        #print(input.shape)
        batch_size,seqlen,features,cow,col = input.size()
        input = input.view(batch_size*seqlen,features,cow,col)
        coutput = self.dsc(input) #/dsc
        r_input = coutput.view(batch_size,seqlen,-1)
       # print(r_input.size())
        rout1,h1 = self.rnn1(r_input)
       # print(rout1.shape)   #[16,10,16]
       # out = self.rnn2(rout1,h1)
        
        out = self.linear(rout1.reshape(rout1.size(0),-1))  # contigious
       # print(out.size())
        out = nn.functional.log_softmax(out,dim=1)
        return out


model = None
optimizer = None
loss_fun = None
def init_model():
    global model,optimizer,loss_fun
    model = Combine()
    model.double()
    optimizer =torch.optim.Adam(model.parameters(),LR,weight_decay=WD)
    loss_fun = nn.NLLLoss()#CrossEntropyLoss()
    if CUDA is True:
        model.cuda()

'''
model visualization using visualize.py
'''
# xx = torch.randn(1,10,128,6,6).double()
# y = model(xx)
# g = make_dot(y)
# g.view()

#print(model)







def traintest(dataset):
    '''
    val shuffle
    '''
    size = dataset.label.size
    permutation = np.random.permutation(size)
    p = int(size*0.05)
    data = dataset.data[permutation[:p],:,:,:,:]
    label = dataset.label[permutation[:p],:]

    if NOISE is True:
        for index in range(p):
            for i in range(data.shape[1]):#seq
                data[index][i] = addGaussianNoise(data[index][i])

    if ONE_ZERO is True:
        for index in range(p):
            for i in range(data.shape[1]):
                data[index][i] = normalization(data[index][i]) 

    return data,label

def train(epoch,trainloader,test_x,test_y):
    model.train()
    loss = 0.0
    accuracy= 0.0
    for step,(x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        y =y.long()
        if CUDA is True:
            y = y.cuda()
            x = x.cuda()
        #print(x.shape)
        out = model(x)
        #print(out.size())
        #print('=========')
        #print(y.squeeze().size())
        loss = loss_fun(out,y.squeeze())    
        loss.backward()
        optimizer.step()
        if step%5 == 0:
            #print(test_x.shape)
            tx = torch.from_numpy(test_x)#.astype('float64')
           # print(type(tx),tx.shape)
            ty = torch.from_numpy(test_y).long().squeeze()
            if CUDA is True:
                tx = tx.cuda()
                ty = ty.cuda()
            testout = model(tx)
            #print(testout)
            #print(ty.shape)
            pred_y = testout.data.max(1)[1]
            #print(pred_y)
            #print(test_y.size(0))
            #print(testout.data.max(1)[1])
            #print(pred_y.shape)
            #print(test_y.shape)
            # print('=======')
            # print((pred_y == ty).sum())
            # print(pred_y)
            # print(ty)
            # print(ty.size(0))
            accuracy = (pred_y == ty).sum().item()/ty.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.item(),'train acc:%.4f'%accuracy)
        
    return loss,accuracy

def test(testloader):
    model.eval()
    test_loss = 0.0
    acc = 0.0
    step = 0
    for step,(x,y) in enumerate(testloader):
       
        
        if CUDA is True:
            x = x.cuda()
            y = y.cuda()
        testout = model(x)
        y = y.long()
        #print(y.shape)
        pred_y = testout.data.max(1)[1]
        #print(pred_y.shape)
        acc += (pred_y == y.squeeze()).sum().item()/y.size(0)
        #print(y.shape)
        #print(testout.shape)
        test_loss += loss_fun(testout,y.squeeze()).data.item()
        #print(acc,'nnnn',(pred_y == y.squeeze()).sum())
    #print(step)
    test_loss /= step+1 #len(testloader.dataset)
    acc /= step+1
    print('Test set average loss %.4f:'%test_loss,'accuracy %.4f'%acc)
    return test_loss,acc

def cross():
    #start = time.time()
    epoch = 0
    k=5
    train_loss = np.zeros(shape=(k,EPOCH))
    train_acc = np.zeros(shape=(k,EPOCH))
    test_loss = np.zeros(shape=k)
    test_acc = np.zeros(shape=k)
    for index in range(k):
        init_model()
        #train_loss = np.zeros(shape=(EPOCH,2))
        #test_loss = np.zeros(shape=(EPOCH,2))
        print('======index'+str(index)+'========')
        crossdata = CrossDataSet(train_path,test_path,k,index,cross=True)
        crosstest = CrossTest(crossdata.testdata,crossdata.testlabel)
        trainloader = torch.utils.data.DataLoader(dataset=crossdata,batch_size=BATCH_SIZE,shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=crosstest,batch_size=BATCH_SIZE,shuffle=True)
        test_x,test_y = traintest(crossdata)
        for epoch in range(EPOCH):
            
            #print(crosstest.label.size,len(testloader.dataset))
            #train_loss[epoch,0],train_loss[epoch,1] = 
            train_loss[index,epoch],train_acc[index,epoch] = train(epoch,trainloader,test_x,test_y)
            #test_loss[epoch,0],test_loss[epoch,1]= test(testloader)
        
    #print('time:',end-start)
        path = str(index)+'model.pkl'
        torch.save(model.state_dict(),path)
    testloss=0.0
    testacc=0.0
    for index in range(k):
        crossdata = CrossDataSet(train_path,test_path,k,index,cross=True)
        crosstest = CrossTest(crossdata.testdata,crossdata.testlabel)
        testloader = torch.utils.data.DataLoader(dataset=crosstest,batch_size=BATCH_SIZE,shuffle=True)
        #test_loss[epoch,0],test_loss[epoch,1]= test(testloader)
        path = str(index)+'model.pkl'
        init_model()
        model.load_state_dict(torch.load(path))
        loss,acc = test(testloader)
        test_loss[index] = loss
        test_acc[index] = acc
        testloss += loss
        testacc += acc        
    testloss /= k
    testacc /= k
    print('Final test loss:%.4f'%testloss,'acc%.4f'%testacc)
    #plt.show()
    # end = time.time()
    # print('time',end-start)
    plt.subplot(3,1,1)
    for i in range(k):
        plt.plot(np.arange(EPOCH),train_loss[i,:],label='train_loss%i'%i)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(3,1,2)
    for i in range(k):
        plt.plot(np.arange(EPOCH),train_acc[i,:],label='train_acc%i'%i)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(np.arange(k),test_loss[:],label='test_loss')
    plt.plot(np.arange(k),test_acc[:],label = 'test_acc')
    plt.xlabel('index')
    plt.legend()
    plt.savefig('crossval'+'bs'+str(BATCH_SIZE)+'lr'+str(LR) + EXTRA_NAME+'.png',format='png')

def final():
    start = time.time()
    init_model()
    train_loss = np.zeros(shape=(EPOCH,2))
    test_loss = np.zeros(shape=(EPOCH,2))
    # traindata = DataSet(train_path)
    # testdata = DataSet(test_path)
    # trainloader = torch.utils.data.DataLoader(dataset=traindata,batch_size=BATCH_SIZE,shuffle=True)
    # testloader = torch.utils.data.DataLoader(dataset=testdata,batch_size=BATCH_SIZE,shuffle=True)
    # test_x,test_y = traindata.aa['data'][8*23*6:9*23*6],traindata.aa['label'][8*6*23:9*23*6]

    crossdata = CrossDataSet(train_path,test_path,10,1,cross=True)
    crosstest = CrossTest(crossdata.testdata,crossdata.testlabel)
    trainloader = torch.utils.data.DataLoader(dataset=crossdata,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    testloader = torch.utils.data.DataLoader(dataset=crosstest,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    #crossdata = DataSet(train_path)
    #trainloader = torch.utils.data.DataLoader(dataset=crossdata,batch_size=BATCH_SIZE,shuffle=True)

    test_x,test_y = traintest(crossdata)
    for epoch in range(EPOCH):#TODO
            train_loss[epoch,0],train_loss[epoch,1] = train(epoch,trainloader,test_x,test_y)
            test_loss[epoch,0],test_loss[epoch,1]= test(testloader)
    
    #print('time:',end-start)
    torch.save(model.state_dict(),'testmodel.pkl')
    plt.subplot(2,1,1)
    plt.plot(np.arange(EPOCH),train_loss[:,0],'b-',label='train_loss')
    plt.plot(np.arange(EPOCH),test_loss[:,0],'r-',label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.arange(EPOCH),test_loss[:,1],'r-',label='test_acc')
    plt.plot(np.arange(EPOCH),train_loss[:,1],'b-',label='train_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    end = time.time()
    plt.savefig('epoch'+str(epoch)+'bs'+str(BATCH_SIZE)+'lr'+str(LR)+'time'+str(end-start)+EXTRA_NAME+'.png',format='png')
    plt.show()
    
def valwithmodel():
    init_model()
    # crossdata = DataSet(train_path,shuffle=True)
    # testdata = DataSet(train_path,shuffle=True,test=True)
    # trainloader = torch.utils.data.DataLoader(dataset=crossdata,batch_size=BATCH_SIZE,shuffle=True)
    # testloader = torch.utils.data.DataLoader(dataset=testdata,batch_size=BATCH_SIZE,shuffle=True)
    # test_x,test_y = traintest(crossdata)
    # for epoch in range(EPOCH):#TODO
    #        train(epoch,trainloader,test_x,test_y)
    #        test(testloader)
    
    # #print('time:',end-start)
    # torch.save(model.state_dict(),'0model.pkl')
    # return 0
    model.load_state_dict(torch.load('0model.pkl',map_location='cpu'))
    data = DataSet(train_path)
    trainloader = torch.utils.data.DataLoader(dataset=data,batch_size=BATCH_SIZE,shuffle=True)
    test_loss =0.0
    acc = 0.0
    step = 0
    print(len(trainloader.dataset))
    for step,(x,y) in enumerate(trainloader):        
        if CUDA is True:
            x = x.cuda()
            y = y.cuda()
        testout = model(x)
        y = y.long()
        #print(y.shape)
        pred_y = testout.data.max(1)[1]
        print(pred_y)
        acc = (pred_y == y.squeeze()).sum().item()/y.size(0)
        print('=========')
        print(y.squeeze())
        #print(testout.shape)
        test_loss = loss_fun(testout,y.squeeze()).data.item()
        print(acc,'nnnn',(pred_y == y.squeeze()).sum())
        print('Test set average loss %.4f:'%test_loss,'accuracy %.4f'%acc)
    #print(step)
   # test_loss /= step+1 #len(testloader.dataset)
    #acc /= step+1
    
    #return test_loss,acc
if __name__ == "__main__":
  #  cross()
    #final()
   
    valwithmodel()