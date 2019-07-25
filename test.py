# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import torch.nn as nn
import torchvision as tv
import numpy as np
from visualize import make_dot
import time 
import matplotlib.pyplot as plt
         
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
