import argparse
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional, init
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import os
from datetime import datetime
import matplotlib.pyplot as plt
from utils import *
from torchvision import datasets, transforms

def transform_flatten(tensor):
    return tensor.view(-1,1).contiguous()

def transform_permute(tensor, perm):
    return tensor.index_select(0,perm)

def compute_loss_accuracy(data, label):
    hx = None
    if not pmnist:
        h0 = Variable(data.data.new(data.size(0), hidden_size).normal_(0, 0.1))
        c0 = Variable(data.data.new(data.size(0), hidden_size).normal_(0, 0.1))
        hx = (h0, c0)
    _, (h_n, _) = model(input_=data, hx=hx)
    logits = fc(h_n[0])
    loss = loss_fn(input=logits, target=label)
    accuracy = (logits.max(1)[1] == label).float().mean()
    return loss, accuracy

def main():
    data_path = args.data
    save_dir = args.save
    model_name = args.model
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    max_iter = args.max_iter
    use_gpu = args.gpu
    pmnist = args.pmnist

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if pmnist:
        perm = torch.randperm(784)
    else:
        perm = torch.arange(0, 784).long()
    
    train_dataset = datasets.MNIST(root=data_path, train=True,
            transform=transforms.Compose([transforms.ToTensor(),
            transform_flatten,partial(transform_permute, perm=perm)]),download=True)

    valid_dataset = datasets.MNIST(root=data_path, train=False,
            transform=transforms.Compose([transforms.ToTensor(),
            transform_flatten,partial(transform_permute, perm=perm)]),download=True)
    
    train_Loss = []
    train_Accuracy = []
    Loss = []
    Accuracy = []

    if model_name == 'bnlstm':
        model = LSTM(cell_class = BNLSTMCell, input_size=1,
                hidden_size = hidden_size, batch_first=True,
                max_length=784)
    elif model_name == 'lstm':
        model = LSTM(cell_class = BNLSTMCell, input_size=1,
                hidden_size = hidden_size, batch_first=True,
                max_length=784)
    elif model_name == 'gbnlstm':
        model = LSTM(cell_class = GBNLSTMCell, input_size=1,
                hidden_size = hidden_size, batch_first=True,
                max_length=784)
    else:
        raise ValueError

    fc = nn.Linear(in_features = hidden_size, out_features = 10)
    loss_fn = nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(fc.parameters())
    optimizer = optim.RMSprop(params = params,lr=1e-3,momentum=0.9)

    use_gpu=True
    if use_gpu:
        model.cuda()
        fc.cuda()

    def compute_loss_accuracy(data, label):
        hx = None
        if not pmnist:
            h0 = Variable(data.data.new(data.size(0), hidden_size).normal_(0, 0.1))
            c0 = Variable(data.data.new(data.size(0), hidden_size).normal_(0, 0.1))
            hx = (h0, c0)
        _, (h_n, _) = model(input_=data, hx=hx)
        logits = fc(h_n[0])
        loss = loss_fn(input=logits, target=label)
        accuracy = (logits.max(1)[1] == label).float().mean()
        return loss, accuracy


    iter_cnt = 0
    valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size,
                         shuffle=True, pin_memory=False)
    while iter_cnt < max_iter:
        train_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=True, pin_memory=False)
        for train_batch in train_loader:
            train_data, train_label = train_batch
            train_data = Variable(train_data)
            train_label = Variable(train_label)
            if use_gpu:
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            model.train(True)
            model.zero_grad()
            train_loss, train_accuracy = compute_loss_accuracy(
                data=train_data, label=train_label)
            train_loss.backward()
            clip_grad_norm(parameters=params, max_norm=1)
            optimizer.step()
            if iter_cnt % 10 == 0:
                train_Loss.append(train_loss)
                train_Accuracy.append(train_accuracy)
                print("iteration %i,training loss is %f, training accuracy is %f" %(iter_cnt,train_loss,train_accuracy))
            if iter_cnt % 50 == 49:
                for valid_batch in valid_loader:
                    valid_data, valid_label = valid_batch
                    break
                valid_data = Variable(valid_data, volatile=True)
                valid_label = Variable(valid_label, volatile=True)
                if use_gpu:
                    valid_data = valid_data.cuda()
                    valid_label = valid_label.cuda()
                model.train(False)
                valid_loss, valid_accuracy = compute_loss_accuracy(
                    data=valid_data, label=valid_label)
                print("Valid set loss %f and accuracy %f when iteration %i" %(valid_loss,valid_accuracy,iter_cnt))
                Loss.append(valid_loss)
                Accuracy.append(valid_accuracy)
                save_path = '{}/{}'.format(save_dir, iter_cnt)
                torch.save(model, save_path)
            iter_cnt += 1
            if iter_cnt == max_iter:
                break

    plt.plot(train_Loss)
    plt.title("Loss on train set")
    plt.xlabel("every 10 iterations")
    plt.ylabel("Loss Value")
    plt.savefig("Train loss.png")
    plt.show()

    plt.plot(train_Accuracy)
    plt.title("Accuracy on train set")
    plt.xlabel("every 10 iterations")
    plt.ylabel("Accuracy %")
    plt.savefig("Train accuracy.png")
    plt.show()

    plt.plot(Loss)
    plt.title("Loss on valid set")
    plt.xlabel("every 50 iterations")
    plt.ylabel("Loss Value")
    plt.savefig("Valid loss.png")
    plt.show()

    plt.plot(Accuracy)
    plt.title("Accuracy on valid set")
    plt.xlabel("every 50 iterations")
    plt.ylabel("Accuracy %")
    plt.savefig("Valid accuracy.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the model using MNIST dataset.')
    parser.add_argument('--data',required=True,
			help='The path to download the MNIST dataset, within the current index')
    parser.add_argument('--model',required=True,choices=['lstm','bnlstm','gbnlstm'],
                        help='Decide which model to use, three choice: lstm bnlstm gbnlstm')
    parser.add_argument('--save',required=True,help='The path to save model files')
    parser.add_argument('--hidden-size',required=True,type=int,
			help='The number of hidden units')
    parser.add_argument('--pmnist',default=False,action='store_true',
                        help='If set, it use permutated-MNIST dataset')
    parser.add_argument('--batch-size',required=True,type=int,
                        help='The maximum iteration count')
    parser.add_argument('--max-iter',required=True,type=int,
                        help='The maximum iteration count')
    parser.add_argument('--gpu',default=False,action='store_true',
                        help='The value specifying whether to use GPU')
    args = parser.parse_args()
    main()










    
