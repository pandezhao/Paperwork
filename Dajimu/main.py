import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

class Conv_DeConv(nn.Module):
    def __init__(self):
        super(Conv_DeConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.Re = nn.ReLU()
        self.pool = nn.MaxPool2d(2,stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.common_1 = nn.Linear(512,8)
        self.fc1 = nn.Linear(8,4)
        self.fc2 = nn.Linear(8,4)
        self.fc3 = nn.Linear(8,4)
        self.common_2 = nn.Linear(8, 512)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5)

    def forward(self, input_ ,flag=0):
        print(input_.size())
        input_, indices_1 = self.pool(self.Re(self.conv1(input_)))
        input_, indices_2 = self.pool(self.Re(self.conv2(input_)))
        print(input_.size())
        tmp = input_.size()
        input_ = input_.view(-1)
        input_ = self.common_1(input_)
        if flag:
            input_1 = self.fc1(input_)
            input_2 = self.fc2(input_)
        else:
            input_1 = self.fc2(input_)
            input_2 = self.fc3(input_)
        input_ = torch.cat([input_1, input_2])
        input_ = self.common_2(input_).view(tmp)
        input_ = self.unpool(input_, indices_2)
        input_ = self.deconv1(input_)
        input_ = self.unpool(input_, indices_1)
        output = self.deconv2(input_)
        return output

data_path = "CNN dataset"
save_dir = "CNN saved"
use_gpu = True
iteration = 1000
epochs = 10

train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST(root=data_path, train=False, download=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set,batch_size=1000)
test_loader = DataLoader(dataset=test_set,batch_size=2000)

def wash_data1(data,label): #将当前batch内的所有同label的数据，从0倒9分开并且分别装进一个字典里
    Origin={}
    for i in range(10):
        location = np.where(label==i)
        out = data[location]
        Origin["data_{}".format(i)]=out
    return Origin

def wash_data3(Current):
    tongbu = []
    yibu = []
    for i in range(len(Current)):
        tmp = len(Current)
        tongbu.append([Current[i],Current[i]])
        for j in range(i+1,len(Current)):
            yibu.append([Current[i],Current[j]])
            yibu.append([Current[j],Current[i]])
    random.shuffle(tongbu)
    random.shuffle(yibu)
    return {"tongbu":tongbu,"yibu":yibu}

def wash_data2(Origin): #将处理好的数据分别装进一个数组的十个元素里，每个元素又是一个数组，一个数组装着两个数组，一个是同步数据，一个是异步数据。
    shuzu = []
    for i in range(10):
        Current = Origin["data_{}".format(i)]
        result = wash_data3(Current)
        shuzu.append(result)
    return shuzu

model = Conv_DeConv()
optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.3)

for data_batch,data_label in train_loader:
    print("data_batch.size",data_batch.size())
    print("data_label.size",data_label.size())
    Origin=wash_data1(data_batch,data_label)
    print(len(Origin))
    print(Origin["data_1"].size())
    shuzu = wash_data2(Origin)

    tongbu = []
    yibu = []
    for i in range(10):
        for j in range(len(shuzu[i]["tongbu"])):
            tongbu.append(shuzu[i]["tongbu"][j])
        for k in range(len(shuzu[i]["yibu"])):
            yibu.append(shuzu[i]["yibu"][k])

    for epoch in range(epochs):
        for i in range(iteration):
            test_1 = tongbu[i][0].view(1,1,28,28)
            test_1_result = tongbu[i][0].view(1,1,28,28)
            out_1 = model(input_ = test_1, flag = 0)
            loss_1 = nn.MSELoss(out_1, test_1_result)
            loss_1.backward()
            optimizer.step()

            test_2 = yibu[i]
            test_2_result = yibu[i][0]
            out_2 = model(input_ = test_2, flag = 0)
            loss_2 = F.nll_loss(input = out_2, target = test_2_result)
