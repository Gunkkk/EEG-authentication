import torch
import torch.utils.data
import torch.nn as nn
import torchvision as tv
import numpy as np

path = 'eeg_data.npy'
BATCH_SIZE = 16


def meanandvar(data):
    m = np.mean(data,axis=(0,1,3,4))
    n = np.std(data,axis=(0,1,3,4))
    #print(m,n)
    return m,n

def normalization(data):
    m = data[:,:,:,:].max()
    n = data[:,:,:,:].min() 
    data[:,:,:,:] = (data[:,:,:,:]-n)/(m-n)
    return data

def standardScaler(data,m,n):
    for i in range(128):
        data[:,i,:,:] = (data[:,i,:,:]-m[i])/n[i]
    return data
    #print(m,n)
    


class DataSet(torch.utils.data.Dataset):
    aa =None
    def __init__(self,transform=None):
        super(DataSet,self).__init__()
        self.aa = np.load(path).item()
        self.normaa = normalization(self.aa['data'][:])
        self.m,self.s = meanandvar(self.normaa)

    def __getitem__(self,index):
        data,label = self.aa['data'][index],self.aa['label'][index]
        #print(data.shape)
       # seqlen,features,cow,col = data.shape
        #print(batch_size,seqlen,features,cow,col)
        
       # data = torch.from_numpy(data)
        #data = torch.DoubleTensor(data)
       # dd = np.zeros(data.shape)
        
        data = normalization(data)   # data => 10*128*6*6 （6*23）
        data = standardScaler(data,self.m,self.s)
        
        #data = torch.DoubleTensor(data)
       # for i in range(np.size(data,0)):
        #data[:] = self.transform(data[:])
        #data = self.transform(data[:])
          #  dd[i] = normalization(data[i])
       # data = dd
        return data,label

    def __len__(self):
        return self.aa['label'].size

traindata = DataSet()
test_x,test_y = traindata.aa['data'][5*23*6:],traindata.aa['label'][5*6*23:]
trainloader = torch.utils.data.DataLoader(dataset=traindata,batch_size=BATCH_SIZE,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2)   
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=2,padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3)
            #nn.MaxPool2d(2)
        )
        self.linear = nn.Linear(128,32)
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
        self.rnn1 = nn.GRU(32,16,2,batch_first=True,dropout=0.3)
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
optimizer =torch.optim.Adam(model.parameters(),0.003)
loss_fun = nn.NLLLoss()#CrossEntropyLoss()


for epoch in range(100):
    for step,(x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        y =y.long()
        out = model(x)
        #print(out)
        loss = loss_fun(out,y.squeeze())    
        loss.backward()
        optimizer.step()

        if step%5 ==0:
            #print(test_x.shape)
            test_x = torch.DoubleTensor(test_x)
            test_y = torch.LongTensor(test_y).squeeze()
            testout = model(test_x)
            pred_y = testout.data.max(1)[1]
            #print(test_y.size(0))
            #print(testout.data.max(1)[1])
            #print(pred_y.shape)
            #print(test_y.shape)
            #print((pred_y == test_y).sum())
            accuracy = (pred_y == test_y).sum().item()/test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.item(), '| test accuracy: %.2f' % accuracy)


