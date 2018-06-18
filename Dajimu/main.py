# 这次算法的目的是，借鉴自编码器的原理。进行同类而不同数据的还原。
# the purpose of this algorithm is, reconstruct the different sample in same class. Based on the autodecoder's principle.

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

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
        input_, indices_1 = self.pool(self.Re(self.conv1(input_)))
        input_, indices_2 = self.pool(self.Re(self.conv2(input_)))
        tmp = input_.size()
        input_ = input_.view(-1)
        input_ = self.common_1(input_)
        if flag==0: # 用于同步输出，传统的autodecoder
            input_1 = self.fc1(input_)
            input_2 = self.fc2(input_)
        if flag==1: # 用于异步输出，这次尝试的东西
            input_1 = self.fc2(input_)
            input_2 = self.fc3(input_)
        if flag==2: # 用于表征，可以用聚类算法来进行计算
            output = self.fc2(input_)
            return output
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

train_loader = DataLoader(dataset=train_set,batch_size=10000)
test_loader = DataLoader(dataset=test_set,batch_size=500)

def show(input_, output_, name1=1,name2=1): # 用于展示所生成的图片
    input = input_.view(28,28)
    output = output_.view(28,28)
    input_img = transforms.ToPILImage(input)
    output_img = transforms.ToPILImage(output)
    input_img.show()
    input_img.save("raw_epoch_{}_iter_{}.png".format(name1,name2))
    output_img.show()
    output_img.save("output_epoch_{}_iter_{}.png".format(name1,name2))


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
    return {"tongbu":tongbu,"yibu":yibu}

def wash_data2(Origin): #将处理好的数据分别装进一个数组的十个元素里，每个元素又是一个数组，一个数组装着两个数组，一个是同步数据，一个是异步数据。
    shuzu = []
    for i in range(10):
        Current = Origin["data_{}".format(i)]
        result = wash_data3(Current)
        shuzu.append(result)
    return shuzu

def cuorong(input_,iteration):
    if len(input_)>=iteration:
        tmp = input_[:iteration]
        return random.shuffle(tmp)
    else:
        cishu = int(iteration / len(input_))
        yushu = iteration % len(input_)
        tmp = []
        for i in range(cishu):
            tmp+=input_
        tmp+=input_[:yushu]
        return random.shuffle(tmp)

model = Conv_DeConv()
optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.3)
loss = nn.MSELoss()

for data_batch,data_label in train_loader:
    Origin=wash_data1(data_batch,data_label)
    shuzu = wash_data2(Origin)

    tongbu = []
    yibu = []
    # tongbu_test = []
    for i in range(10):
        for j in range(len(shuzu[i]["tongbu"])):
            tongbu.append(shuzu[i]["tongbu"][j])
        for k in range(len(shuzu[i]["yibu"])):
            yibu.append(shuzu[i]["yibu"][k])
        # for l in range(int(0.9*len(shuzu[i]["tongbu"])),len(shuzu[i]["tongbu"])):
        #     tongbu_test[i].append(shuzu[i]["tongbu"][l])

    tongbu_Loss=[]
    yibu_Loss=[]

    for epoch in range(epochs):
        for i in range(iteration):
            test_1 = tongbu[i][0].view(1,1,28,28)
            test_1_result = tongbu[i][0].view(1,1,28,28)
            out_1 = model(input_ = test_1, flag = 0)
            output_1 = loss(out_1, test_1)
            tongbu_Loss.append(output_1)
            optimizer.zero_grad()
            output_1.backward()
            optimizer.step()

            test_2 = yibu[i][0].view(1,1,28,28)
            test_2_result = yibu[i][1].view(1,1,28,28)
            out_2 = model(input_ = test_2, flag = 1)
            output_2 = loss(out_2, test_2)
            yibu_Loss.append(output_2)
            optimizer.zero_grad()
            output_2.backward()
            optimizer.step()
            if i%10==0:
                print("epoch {}, iteration {}".format(epoch,i),"tongbu loss is {}".format(output_1)," yibu loss is {}".format(output_2))

            if i+1 % int(0.2*iteration) == 0:
                torch.save(model,"epoch_{}_iteration_{}.pkl".format(epoch,i))

    plt.plot(tongbu_Loss)
    plt.title("Loss on train set")
    plt.xlabel("every 10 iterations")
    plt.ylabel("Loss Value")
    plt.show()

    plt.plot(yibu_Loss)
    plt.title("Loss on train set")
    plt.xlabel("every 10 iterations")
    plt.ylabel("Loss Value")
    plt.show()

    break

yibu_test = []

for test_batch,test_label in test_loader:
    Origin_test = wash_data1(test_batch,test_label)
    shuzu_test = wash_data2(Origin_test)
