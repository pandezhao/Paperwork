import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NewBatchNorm1D(nn.Module):

    def __init__(self, num_features, max_length=1, use_bias=True, alpha=0.25, measure='SD', momentum=0.9, eps=1e-5, affine = True):
        super(NewBatchNorm1D, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.eps = eps
        self.affine = affine
        self.use_bias = use_bias
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features))
            if self.use_bias:
                self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.alpha = alpha
        for i in range(max_length):
            self.register_buffer('running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer('running_var_{}'.format(i), torch.ones(num_features))
        # self.running_mean = []
        # self.running_var = []
        # for i in range(max_length):
        #     self.running_mean.append(torch.zeros(requires_grad=False))
        #     self.running_var.append(torch.ones(requires_grad=False))
        self.measure = measure
        self.func = self.dm_func(measure)
        self.training = True
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(0.1)
            if self.use_bias:
                self.bias.data.zero_()

    def dm_func(self, measure):
        def SD(x):
            return torch.mean(x, 0), torch.std(x, 0)

        def MAD(x):
            mean = torch.mean(x, 0)
            return mean, torch.mean(torch.abs(x - mean), 0)

        def RSD(x):
            mean = torch.mean(x, 0)
            diff = x - mean
            return mean, torch.mean(diff.masked_fill_(diff.le(0.0), 0.0), 0)

        def RBD(x):
            sup, _ = torch.max(x, 0)
            inf, _ = torch.min(x, 0)
            return (sup + inf) * 0.5, (sup - inf)

        def SQD(x, alpha=0.25):
            x_num = x.size()[0]
            split_num = int(x_num * (1 - alpha))
            if split_num > x_num:
                split_num = x_num
            if split_num <= 1:
                split_num = 2
            topk, _ = torch.topk(x, split_num - 1, dim=0)
            statistic, _ = torch.min(topk, dim=0)
            mean = torch.mean(x, dim=0)
            deviation = torch.sum(topk, dim=0) / (split_num - 1) - mean
            return statistic, deviation

        def LED(x):
            return Variable(torch.log(torch.mean(torch.pow(torch.zeros(x.size()).cuda() + 2.71828, x.data), 0) + 1e-6)), Variable(torch.log(torch.mean(torch.pow(torch.zeros(x.size()).cuda() + 2.71828, (x - x.mean()).data + 1e-6), 0) + 1e-6))

        def WCD(x):
            sup, _ = torch.max(x, 0)
            mean = torch.mean(x, 0)
            return sup, (sup - mean)

        if measure == 'SD':
            return SD
        if measure == 'MAD':
            return MAD
        if measure == 'RSD':
            return RSD
        if measure == 'RBD':
            return RBD
        if measure == 'SQD':
            return SQD
        if measure == 'SQD1':
            return partial(SQD, alpha=0.25)
        if measure == 'SQD2':
            return partial(SQD, alpha=0.5)
        if measure == 'SQD3':
            return partial(SQD, alpha=0.75)
        if measure == 'LED':
            return LED
        if measure == 'WCD':
            return WCD

    def forward(self, input_, time):  # 这里的归一化应该是沿着行，进行归一化
        batch_size, length = input_.size()

        if self.training:
            running_mean = getattr(self, 'running_mean_{}'.format(time))
            running_var = getattr(self, 'running_var_{}'.format(time))
            if self.measure.__eq__('SQD'):
                statistics, dev = self.func(input_.data, alpha=self.alpha)
            else:
                statistics, dev = self.func(input_.data)

        # running_mean = self.buffer_mean[time]
        # running_var = self.buffer_var[time]
            tmp_mean = self.momentum * statistics.data + (1 -self.momentum) * running_mean
            tmp_var = self.momentum * dev.data * dev.data + (1 - self.momentum) * running_var
        # self.buffer_mean[time] = tmp_mean
        # self.buffer_var[time] = tmp_var
            setattr(self, 'running_mean_{}'.format(time), tmp_mean)
            setattr(self, 'running_var_{}'.format(time), tmp_var)
            tmp_mean = tmp_mean.unsqueeze(0).expand(batch_size, *statistics.size())
            tmp_var = tmp_var.unsqueeze(0).expand(batch_size, *dev.size())
        else:
            running_mean = getattr(self, 'running_mean_{}'.format(time))
            running_var = getattr(self, 'running_var_{}'.format(time))
            tmp_mean = running_mean.unsqueeze(0).expand(batch_size, *running_mean.size())
            tmp_var = running_var.unsqueeze(0).expand(batch_size, *running_var.size())
        if self.affine:
            if self.use_bias:
                bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
                bn_output = self.weight * (input_ - tmp_mean) / (tmp_var + self.eps) + bias_batch
            else:
                bn_output = self.weight * (input_ - tmp_mean) / (tmp_var + self.eps)
        else:
            bn_output = (input_ - tmp_mean) / (tmp_var + self.eps)
        return bn_output


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-5, measure='SD', alpha=0.25):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = eps
        self.measure = measure
        self.alpha = alpha
        self.weight_hidden = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.weight_input = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        # self.batch_norm1 = nn.BatchNorm1d(hidden_size * 4)
        # self.batch_norm2 = nn.BatchNorm1d(hidden_size * 4)
        # self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.bn_h = NewBatchNorm1D(num_features=hidden_size * 4, max_length=28, use_bias=False,  # this 32 is batch size
                                 alpha=self.alpha, measure=self.measure)
        self.bn_i = NewBatchNorm1D(num_features=hidden_size * 4, max_length=28, use_bias=False,
                                 alpha=self.alpha, measure=self.measure)
        self.bn_c = NewBatchNorm1D(num_features=hidden_size, max_length=28, use_bias=True,
                                 alpha=self.alpha, measure=self.measure)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_input.data)
        # weight_h_tmp = torch.eye(self.hidden_size) # 原文使用了这种初始化方法，我这里先尝试下正交初始化
        # weight_h_tmp = weight_h_tmp.repeat(1, 4)
        # self.weight_hidden.data.set_(weight_h_tmp)
        init.orthogonal_(self.weight_hidden.data)
        init.constant_(self.bias.data, val=0)

    def forward(self, input_, hx, time):
        batch_size = input_.size(0)
        h_0, c_0 = hx
        wh = torch.mm(h_0, self.weight_hidden)
        wx = torch.mm(input_, self.weight_input)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        # if time==0:
        #     print("bn_h: ",'time: ',time, self.bn_h.weight)
        #     print("bn_i: ",'time: ',time, self.bn_i.weight)
        #     print("bn_c: ",'time: ',time, self.bn_c.weight)
        wh = self.bn_h(wh, time=time)
        wx = self.bn_i(wx, time=time)
        # wh = self.batch_norm1(wh)
        # wx = self.batch_norm2(wx)
        # f, i, o, g = torch.split(wh + wx, split_size_or_sections=self.hidden_size, dim=1)
        f, i, o, g = torch.split(wh + wx + bias_batch, split_size_or_sections=self.hidden_size, dim=1)
        # f, i, o, g = torch.split(wh + wx + self.bias, split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        bn_c_1 = self.bn_c(c_1, time=time)
        # bn_c_1 = self.batch_norm3(c_1)
        # bn_c_1 = c_1
        h_1 = torch.sigmoid(o) * torch.tanh(bn_c_1)
        return h_1, c_1


class LSTM(nn.Module):
    def __init__(self, cell_class, input_size, hidden_size, max_length, num_layers=2, use_bias=True,
                 dropout=0, dev_measure='SD',alpha=0.25):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.dev_measure = dev_measure
        self.alpha = alpha

        #####################################
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        #####################################

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            if cell_class == 'LSTMCell':
                cell = LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size
                                , measure=self.dev_measure, alpha=self.alpha)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = getattr(self, 'cell_{}'.format(layer))
            cell.reset_parameters()

    def _forward_layer(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx): # 我这里的layer_h_n 相比标准的 H-n少一个hidden state， 我是只输出最后一层layer的hidden state的。
        # h_n = []
        # c_n = []

        layer_output = None
        max_time, batch_size, _ = input_.size()
        for layer in range(self.num_layers):
            cell = getattr(self, 'cell_{}'.format(layer))
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_layer(
                cell=cell, input_=input_, hx=hx)
            input_ = self.dropout_layer(layer_output)
            # h_n.append(layer_h_n)
            # c_n.append(layer_c_n)

        output = layer_output
        # h_n = torch.stack(h_n, 0)
        # c_n = torch.stack(c_n, 0)
        layer_h_n = self.batch_norm(layer_h_n)
        layer_h_n = self.tanh(layer_h_n)
        # layer_h_n = self.linear1(layer_h_n)
        # layer_h_n = self.relu(layer_h_n)

        return output, (layer_h_n, layer_c_n)


def show_data(train_Accuracy, train_Loss, test_Loss, test_Accuracy, scatter=False):  ## attention , we must use 'with open ' to open a txt file, if not. we won't write data in loss.
    with open('train loss.txt', 'w') as file1:
        for tra_loss in train_Loss:
            file1.write('{}\n'.format(tra_loss))
        file1.close()

    with open('train accuracy.txt', 'w') as file2:
        for tra_accu in train_Accuracy:
            file2.write('{}\n'.format(tra_accu))
        file2.close()

    with open('test loss.txt', 'w') as file3:
        for tes_loss in test_Loss:
            file3.write('{}\n'.format(tes_loss))
        file3.close()

    with open('test accuracy.txt', 'w') as file4:
        for tes_accu in test_Accuracy:
            file4.write('{}\n'.format(tes_accu))
        file4.close()

    if scatter == True:
        plt.scatter(range(len(train_Accuracy)), train_Accuracy, s=5)
        plt.title("Accuracy on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("train Accuracy")
        plt.show()

        plt.scatter(range(len(train_Loss)), train_Loss, s=5)
        plt.title("Loss on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("train Loss")
        plt.show()

        plt.scatter(range(len(test_Accuracy)), test_Accuracy, s=5)
        plt.title("Accuracy on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("test Accuracy")
        plt.show()

        plt.scatter(range(len(test_Loss)), test_Loss, s=5)
        plt.title("Loss on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("test Loss")
        plt.show()
    else:
        plt.plot(train_Accuracy)
        plt.title("Accuracy on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("train Accuracy")
        plt.show()

        plt.plot(train_Loss)
        plt.title("Loss on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("train Loss")
        plt.show()

        plt.plot(test_Accuracy)
        plt.title("Accuracy on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("test Accuracy")
        plt.show()

        plt.plot(test_Loss)
        plt.title("Loss on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("test Loss")
        plt.show()














