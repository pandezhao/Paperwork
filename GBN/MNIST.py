## writer = Yiheng Zhao

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from norm import LSTM, show_data
import matplotlib.pyplot as plt
import numpy
import os
import csv

torch.cuda.set_device(0)
data_path = "CNN dataset"
save_dir = "CNN saved"
use_gpu = True
epochs = 25
batch_size = 64
hidden_size = 100
record_data = True

def transform_flatten(tensor):
    return tensor.view(-1,1).contiguous()

def transform_permute(tensor, perm):
    return tensor.index_select(0, perm)

perm = torch.randperm(784)

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

train_set = datasets.MNIST(root=data_path, train=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]), download=True)

test_set = datasets.MNIST(root=data_path, train=False,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))]), download=True)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = LSTM(cell_class='LSTMCell', input_size=1, hidden_size=100, max_length=784, num_layers=2, use_bias=True,
             dropout=0, dev_measure='SD', alpha=0.25)
fc = nn.Linear(in_features=100, out_features=10)
Loss = nn.CrossEntropyLoss()

params = list(model.parameters()) + list(fc.parameters())
optimizer = optim.Adam(params=params, lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 20], gamma=0.2)

train_Accuracy = []
train_Loss = []
test_Accuracy = []
test_Loss = []

if use_gpu:
    model.cuda()
    fc.cuda()


def compute_loss(data, label):
    hx = torch.Tensor(batch_size, hidden_size).normal_(0, 0.1)  # 这里的1的意思是input size， 比如对于这里， 由于每次输入一个像素,所以input size = 1. 所以是1
    if use_gpu:
        hx = hx.cuda()
    hx = (hx, hx)                                      # 所以input size = 1. 所以是1
    _, (out, _) = model(input_=data,hx=hx)
    output = fc(out)
    loss = Loss(input=output, target=label)
    accuracy = (output.max(1)[1] == label).float().mean()
    return loss, accuracy


for epoch in range(epochs):
    count = 0
    loss_tmp=[]
    accuracy_tmp=[]
    model.train(True)
    print('learning rate before: ', optimizer.param_groups[0]['lr'])
    scheduler.step()
    print('learning rate after: ', optimizer.param_groups[0]['lr'])
    for data, label in train_loader:
        # data = data + torch.FloatTensor(0.1 * numpy.random.randn(batch_size,784,1))
        data = data.view(batch_size, 784, 1) ## (batch_size, length, input size)
        data = Variable(data.permute(1, 0, 2))
        # data = data.permute(2, 0, 3, 1)
        # data = Variable(data.view(28, 32, 28))
        label = Variable(label)
        if use_gpu:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        loss, accuracy = compute_loss(data=data, label=label)
        loss.backward()
        clip_grad_norm(parameters=params, max_norm=1)
        optimizer.step()
        count += 1
        if count % 1 == 0:
            loss_tmp.append(loss.data)
            accuracy_tmp.append(accuracy.data)
        if count % 20 == 0:
            train_Accuracy.append(sum(accuracy_tmp)/accuracy_tmp.__len__())
            train_Loss.append(sum(loss_tmp)/loss_tmp.__len__())
            print('Epoch:{},iteration:{},train_loss:{},train_accuracy:{}'.format(epoch, count, sum(loss_tmp)/loss_tmp.__len__(), sum(accuracy_tmp)/accuracy_tmp.__len__()))

            accuracy_tmp = []
            loss_tmp = []
        if count == (int(train_set.__len__()/batch_size) - 1) or count == (int(train_set.__len__()/batch_size/2) - 1):
            with torch.no_grad():
                model.train(False)
                Loss_sum = []
                Accuracy_sum = []
                count_tmp = 0
                for test_data, test_label in test_loader:
                    test_data = test_data.view(batch_size, 784, 1)
                    test_data = Variable(test_data.permute(1, 0, 2))
                    # test_data = test_data.permute(2, 0, 3, 1)
                    # test_data = Variable(test_data.view(28, batch_size, 28))
                    test_label = Variable(test_label)
                    if use_gpu:
                        test_data = test_data.cuda()
                        test_label = test_label.cuda()
                    Tes_Loss, Tes_Accuracy = compute_loss(test_data, test_label)
                    Loss_sum.append(Tes_Loss)
                    Accuracy_sum.append(Tes_Accuracy)
                    count_tmp += 1
                    if count_tmp == int(test_set.__len__()/batch_size) - 1:
                        break
                test_Loss.append(sum(Loss_sum)/len(Loss_sum))
                test_Accuracy.append(sum(Accuracy_sum)/len(Accuracy_sum))
                print("test_loss:{},test_accuracy:{}".format(sum(Loss_sum)/len(Loss_sum), sum(Accuracy_sum)/len(Accuracy_sum)))

        if count == int(train_set.__len__()/batch_size) - 1:
            break
if record_data:
    show_data(train_Accuracy, train_Loss, test_Loss, test_Accuracy)