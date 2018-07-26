## writer = Yiheng Zhao

import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from norm import *

torch.cuda.set_device(0)
data_path = "CIFAR10 dataset"
save_dir = "CIFAR10 saved"
use_gpu = True
epochs = 60
global batch_size
batch_size = 64
layer_numbers = 2
hidden_size = 300
record_data = True

def transform_flatten(tensor):
    return tensor.view(-1,1).contiguous()

def transform_permute(tensor, perm):
    return tensor.index_select(0, perm)

perm = torch.randperm(1024)

train_set = datasets.CIFAR10(root=data_path, train=True,
                             transform=transforms.Compose([transforms.ToTensor()]) # transforms.Normalize((0.1307,), (0.3081,))
                             , download=True)

test_set = datasets.CIFAR10(root=data_path, train=False,
                            transform=transforms.Compose([transforms.ToTensor()]) # transforms.Normalize((0.1307,), (0.3081,))
                             , download=True)

train_length = len(train_set)
test_length = len(test_set)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = LSTM(cell_class='LSTMCell', input_size=3, hidden_size=hidden_size, max_length=1024, num_layers=layer_numbers, use_bias=True,
             dropout=0, dev_measure='SD', alpha=0.25)
fc = nn.Linear(in_features=hidden_size, out_features=10)
Loss = nn.CrossEntropyLoss()

params = list(model.parameters()) + list(fc.parameters())
optimizer = optim.Adam(params=params, lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 50], gamma=0.1)

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
    model.train(True)
    print('learning rate before: ', optimizer.param_groups[0]['lr'])
    scheduler.step()
    print('learning rate after: ', optimizer.param_groups[0]['lr'])
    for data, label in train_loader: ## total number of train set is 50000
        # data = data + torch.FloatTensor(0.1 * numpy.random.randn(batch_size,784,1))
        data = data.view(batch_size, 3, -1)
        data = Variable(data.permute(2, 0, 1)) # classical input format should be (flatten, batch size, input size)
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
            print('Epoch:{},iteration:{},train_loss:{},train_accuracy:{}'.format(epoch, count, loss, accuracy))
        if count % 20 == 0:
            train_Accuracy.append(accuracy)
            train_Loss.append(loss)
        # if count == (int(train_set.__len__() / batch_size) - 1) or count == (int(train_set.__len__() / batch_size / 2) - 1):
        #     with torch.no_grad():
        #         for i in range(1):
        #             model.train(False)
        #             Loss_sum = []
        #             Accuracy_sum = []
        #             count_tmp = 0
        #             for test_data, test_label in test_loader: # total number of test set is 10000.
        #                 test_data = test_data.view(batch_size, 3, -1)
        #                 test_data = Variable(test_data.permute(2, 0, 1))
        #                 # test_data = test_data.permute(2, 0, 3, 1)
        #                 # test_data = Variable(test_data.view(28, batch_size, 28))
        #                 test_label = Variable(test_label)
        #                 if use_gpu:
        #                     test_data = test_data.cuda()
        #                     test_label = test_label.cuda()
        #                 Tes_Loss, Tes_Accuracy = compute_loss(test_data, test_label)
        #                 Loss_sum.append(Tes_Loss)
        #                 Accuracy_sum.append(Tes_Accuracy)
        #                 count_tmp += 1
        #                 if count_tmp == int(test_length/batch_size) - 1:
        #                     break
        #             test_Loss.append(sum(Loss_sum)/len(Loss_sum))
        #             test_Accuracy.append(sum(Accuracy_sum)/len(Accuracy_sum))
        #             print("test_loss:{},test_accuracy:{}".format(sum(Loss_sum)/len(Loss_sum), sum(Accuracy_sum)/len(Accuracy_sum)))
        #     torch.save(model.state_dict(), 'model_para_{}.pkl'.format(epoch))
        if count == int(train_length/batch_size) - 1:
            break
            
if record_data:
    show_data(train_Accuracy, train_Loss, test_Loss, test_Accuracy)
