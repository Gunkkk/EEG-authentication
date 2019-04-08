import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

path = 'eeg.npy'

class DataSet(torch.utils.data.Dataset):
    aa =None
    def __init__(self):
        super(DataSet,self).__init__()
        self.aa = np.load(path)

    def __getitem__(self,index):
        return self.aa[0]['data'][index],self.aa[0]['label'][index]

    def __len__(self):
        return self.aa[0]['label'].size

traindata = DataSet()
test_x,test_y = traindata.aa[0]['data'][:10],traindata.aa[0]['label'][:10]
trainloader = torch.utils.data.DataLoader(dataset=traindata,batch_size=8,shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=14,out_channels=28,kernel_size=5,padding=2), 
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.MaxPool2d(2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=28,out_channels=56,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(56,23)
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        output = self.out(x)


cnn=CNN()

optimizer =torch.optim.Adam(cnn.parameters())
loss_fun = nn.CrossEntropyLoss()

for epoch in range(1):
    for step,(x,y) in enumerate(trainloader):
        out = cnn(x)
        loss = loss_fun(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 ==0:
            testout = cnn(test_x)
            pred_y = torch.max(testout, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)


