import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

path = 'eeg_data.npy'

class DataSet(torch.utils.data.Dataset):
    aa =None
    def __init__(self):
        super(DataSet,self).__init__()
        self.aa = np.load(path).item()

    def __getitem__(self,index):
        return self.aa['data'][index],self.aa['label'][index]

    def __len__(self):
        return self.aa['label'].size

traindata = DataSet()
test_x,test_y = traindata.aa['data'][5*23*6:],traindata.aa['label'][5*6*23:]
trainloader = torch.utils.data.DataLoader(dataset=traindata,batch_size=16,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5)
            #nn.MaxPool2d(2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.5)
            #nn.MaxPool2d(2)
        )
        self.out = nn.Linear(128,32)
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        output = self.out(x)


class Combine(nn.Module):
    def __init__(self):
        super(Combine,self).__init__()
        self.cnn = CNN()
        self.rnn1 = nn.GRU(32,16,2,batch_first=True,dropout=0.3)
        #self.rnn2 = nn.GRU(16,8,1)
        self.linear = nn.Linear(160,23)

    def forward(self,input):
        batch_size,seqlen,features,cow,col = input.size()
        print(batch_size,seqlen,cow,col,features)
        input = input.view(batch_size*seqlen,cow,col,features)
        input = self.cnn(input)
        r_input = input.view(batch_size,seqlen,-1)
        rout1,h1 = self.rnn1(r_input)
       # out = self.rnn2(rout1,h1)
        out = self.linear(rout1.view(batch_size,-1))
        out = nn.Softmax(out,dim=1)
        return out


model = Combine()
optimizer =torch.optim.Adam(model.parameters())
loss_fun = nn.CrossEntropyLoss()


for epoch in range(5):
    for step,(x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fun(out,y)    
        loss.backward()
        optimizer.step()

        if step%5 ==0:
            testout = model(test_x)
            pred_y = testout.data.max(1,keep_dim=True)[1]
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)


