import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import init, functional
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from norm import show_data
import torch.nn.functional as F


class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout):
        super(rnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bias=bias, batch_first=batch_first,dropout=dropout)
        # self.linear1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=10)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, input_, hx):
        hx = torch.stack(hx, 0)
        hx = [hx, hx]
        _, (out, _) = self.LSTM(input_, hx)
        output = self.batch_norm(out[-1])
        output = self.tanh(output) # activation function can not be relu, must be tanh
        output = self.linear2(output)
        return output

data_path = "CNN dataset"
save_dir = "CNN saved"
use_gpu = True
epochs = 2
batch_size = 32
hidden_size = 100

def transform_flatten(tensor):
    return tensor.view(-1,1).contiguous()

train_set = datasets.MNIST(root=data_path, train=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))])
                             , download=True)

test_set = datasets.MNIST(root=data_path, train=False,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))])
                             , download=True)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

model = rnn(input_size=28, hidden_size=hidden_size, num_layers=2, bias=True, batch_first=False, dropout=0.0)
Loss = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(params=model.parameters(), lr=1e-3, momentum=0.9)

train_Accuracy = []
train_Loss = []
test_Accuracy = []
test_Loss = []

if use_gpu:
    model.cuda()


def compute_loss(data, label):
    hx = torch.Tensor(batch_size, hidden_size).normal_(0, 0.001)  # 这里的1的意思是input size， 比如对于这里， 由于每次输入一个像素,所以input size = 1. 所以是1
    if use_gpu:
        hx = hx.cuda()
    hx = (hx, hx)                                      # 所以input size = 1. 所以是1
    output = model(input_=data,hx=hx)
    # output = model(x=data)
    loss = Loss(output, label)
    accuracy = (output.max(1)[1] == label).float().mean()
    return loss, accuracy


for epoch in range(epochs):
    count = 0
    for data, label in train_loader:
        # data = data + torch.FloatTensor(0.0001 * numpy.random.randn(data.size(0),784,1))
        model.train(True)
        data = data.permute(2, 0, 3, 1)
        data = Variable(data.view(28, batch_size, 28))
        # print(data)
        # data = Variable(data.reshape(batch_size,1,28,28))
        # data = Variable(data)
        label = Variable(label)
        if use_gpu:
            data = data.cuda()
            label = label.cuda()

        # model.zero_grad()
        optimizer.zero_grad()

        # loss, accuracy = compute_loss(data=data, label=label)
        Train_loss, Train_accuracy = compute_loss(data, label)
        # print(output)
        # output = model(x=data)
        Train_loss.backward()
        clip_grad_norm(parameters = model.parameters(), max_norm=1)
        optimizer.step()
        count += 1
        if count % 20 == 0:
            train_Accuracy.append(Train_accuracy)
            train_Loss.append(Train_loss)
            print('Epoch:{},iteration:{},train_loss:{},train_accuracy:{},'.format(epoch, count, Train_loss, Train_accuracy))

        if count % 20 == 1:
            with torch.no_grad():
                model.train(False)
                Loss_sum = []
                Accuracy_sum = []
                count_tmp = 0
                for test_data, test_label in test_loader:
                    test_data = test_data.permute(2, 0, 3, 1)
                    test_data = Variable(test_data.view(28, batch_size, 28))
                    test_label = Variable(test_label)
                    if use_gpu:
                        test_data = test_data.cuda()
                        test_label = test_label.cuda()
                    Tes_Loss, Tes_Accuracy = compute_loss(test_data, test_label)
                    Loss_sum.append(Tes_Loss)
                    Accuracy_sum.append(Tes_Accuracy)
                    count_tmp += 1
                    if count_tmp == 100:
                        break
                test_Loss.append(sum(Loss_sum)/len(Loss_sum))
                test_Accuracy.append(sum(Accuracy_sum)/len(Accuracy_sum))


show_data(train_Accuracy, train_Loss, test_Loss, test_Accuracy, scatter=False)

