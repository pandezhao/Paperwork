import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(1)

def dm_SD(x):
    return torch.mean(x, 0), x.std(0, unbiased=False)


def dm_MAD(x):
    mean = torch.mean(x, 0)
    return mean, torch.mean(torch.abs(x - mean), 0)


def dm_RSD(x):
    mean = torch.mean(x, 0)
    diff = x - mean
    return mean, torch.mean(diff.masked_fill_(diff.le(0.0), 0.0), 0)


def dm_RBD(x):
    sup, _ = torch.max(x, 0)
    inf, _ = torch.min(x, 0)
    return (sup + inf)*0.5, (sup - inf)


def dm_SQD(x, alpha=0.25):
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


def dm_LED(x):
    return Variable(torch.log(torch.mean(torch.pow(torch.zeros(x.size()).cuda() + 2.71828, x.data), 0) + 1e-6)), \
           Variable(torch.log(torch.mean(torch.pow(torch.zeros(x.size()).cuda() + 2.71828, (x - x.mean()).data + 1e-6),
                                         0) + 1e-6))


def dm_WCD(x):
    sup, _ = torch.max(x, 0)
    mean = torch.mean(x, 0)
    return sup, (sup - mean)


def dm_func(dm):
    if dm.__eq__('SD'):
        return dm_SD
    if dm.__eq__('MAD'):
        return dm_MAD
    if dm.__eq__('RSD'):
        return dm_RSD
    if dm.__eq__('RBD'):
        return dm_RBD
    if dm.__eq__('SQD'):
        return dm_SQD
    if dm.__eq__('SQD1'):
        return partial(dm_SQD, alpha=0.25)
    if dm.__eq__('SQD2'):
        return partial(dm_SQD, alpha=0.5)
    if dm.__eq__('SQD3'):
        return partial(dm_SQD, alpha=0.75)
    if dm.__eq__('LED'):
        return dm_LED
    if dm.__eq__('WCD'):
        return dm_WCD


class NewBatchNorm1D(nn.Module):

    def __init__(self, num_features, use_bias=True, alpha=0.25, measure='SD', momentum=0.1, eps=1e-5, affine=True, record=True):
        super(NewBatchNorm1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.use_bias = use_bias
        self.momentum = momentum
        self.training = True
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            if self.use_bias:
                self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.track_running_stats = record
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        self.alpha = alpha
        self.measure = measure
        self.dev_calc = dm_func(measure)
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input_):  # 这里的归一化应该是沿着行，进行归一化
        if self.training:
            if self.measure.__eq__('SQD'):
                statistics, dev = self.dev_calc(input_.data, self.alpha)
            else:
                statistics, dev = self.dev_calc(input_.data)

            self.running_mean = self.momentum * statistics.data + (1-self.momentum) * self.running_mean
            self.running_var = self.momentum * dev.data + (1-self.momentum) * self.running_var
            tmp_mean = statistics
            tmp_var = dev
        else:
            tmp_mean = Variable(self.running_mean)
            tmp_var = Variable(torch.sqrt(self.running_var))

        if self.affine:
            if self.use_bias:
                # bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
                bn_output = self.weight * (input_ - tmp_mean) / (tmp_var + self.eps) + self.bias
            else:
                bn_output = self.weight * (input_ - tmp_mean) / (tmp_var + self.eps)
        else:
            bn_output = (input_ - tmp_mean) / (tmp_var + self.eps)
        return bn_output

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__))


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, max_length, eps=1e-5, measure='SD', alpha=0.25):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.eps = eps
        self.measure = measure
        self.alpha = alpha
        self.weight_hidden = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.weight_input = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))

        # for i in range(self.max_length):
        #     # tmp1 = NewBatchNorm1D(hidden_size * 4)
        #     # tmp2 = NewBatchNorm1D(hidden_size * 4)
        #     # tmp3 = NewBatchNorm1D(hidden_size)
        #
        #     tmp1 = nn.BatchNorm1d(hidden_size * 4)
        #     tmp2 = nn.BatchNorm1d(hidden_size * 4)
        #     tmp3 = nn.BatchNorm1d(hidden_size)
        #
        #     tmp1_weight = 0.02 * torch.randn(*tmp1.weight.size(), requires_grad=True) + 0.1
        #     tmp2_weight = 0.02 * torch.randn(*tmp2.weight.size(), requires_grad=True) + 0.1
        #     tmp3_weight = 0.02 * torch.randn(*tmp3.weight.size(), requires_grad=True) + 0.1
        #     tmp1.weight.data = tmp1_weight
        #     tmp2.weight.data = tmp2_weight
        #     tmp3.weight.data = tmp3_weight
        #
        #     setattr(self, 'BN_h_{}'.format(i), tmp1)
        #     setattr(self, 'BN_i_{}'.format(i), tmp2)
        #     setattr(self, 'BN_c_{}'.format(i), tmp3)


        # tmp1 = nn.BatchNorm1d(hidden_size * 4)
        # tmp2 = nn.BatchNorm1d(hidden_size * 4)
        # tmp3 = nn.BatchNorm1d(hidden_size)

        tmp1 = NewBatchNorm1D(hidden_size * 4)
        tmp2 = NewBatchNorm1D(hidden_size * 4)
        tmp3 = NewBatchNorm1D(hidden_size)

        tmp1_weight = 0.02 * torch.randn(*tmp1.weight.size(), requires_grad=True) + 0.1
        tmp2_weight = 0.02 * torch.randn(*tmp2.weight.size(), requires_grad=True) + 0.1
        tmp3_weight = 0.02 * torch.randn(*tmp3.weight.size(), requires_grad=True) + 0.1
        tmp1.weight.data = tmp1_weight
        tmp2.weight.data = tmp2_weight
        tmp3.weight.data = tmp3_weight

        setattr(self, 'BN_h', tmp1)
        setattr(self, 'BN_i', tmp2)
        setattr(self, 'BN_c', tmp3)

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_input.data)
        # weight_h_tmp = torch.eye(self.hidden_size) # 原文使用了这种初始化方法，我这里先尝试下正交初始化
        # weight_h_tmp = weight_h_tmp.repeat(1, 4)
        # self.weight_hidden.data.set_(weight_h_tmp)
        init.orthogonal_(self.weight_hidden.data)
        init.constant_(self.bias.data, val=0)


    def forward(self, input_, hx, time):
        # bn_h = getattr(self, 'BN_h_{}'.format(time))
        # bn_i = getattr(self, 'BN_i_{}'.format(time))
        # bn_c = getattr(self, 'BN_c_{}'.format(time))

        bn_h = getattr(self, 'BN_h')
        bn_i = getattr(self, 'BN_i')
        bn_c = getattr(self, 'BN_c')

        batch_size = input_.size(0)
        h_0, c_0 = hx
        wh = torch.mm(h_0, self.weight_hidden)
        wx = torch.mm(input_, self.weight_input)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        # if time==0:
        #     print("bn_h: ",'time: ',time, self.bn_h.weight)
        #     print("bn_i: ",'time: ',time, self.bn_i.weight)
        #     print("bn_c: ",'time: ',time, self.bn_c.weight)

        wh = bn_h(wh)
        wx = bn_i(wx)
        f, i, o, g = torch.split(wh + wx + bias_batch, split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        bn_c_1 = bn_c(c_1)
        # bn_c_1 = c_1
        h_1 = torch.sigmoid(o) * torch.tanh(bn_c_1)
        return h_1, c_1



class LSTM(nn.Module):
    def __init__(self, cell_class, input_size, hidden_size, max_length, num_layers, use_bias=True,
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
                                ,max_length=self.max_length, measure=self.dev_measure, alpha=self.alpha)
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

        # layer_h_n = self.batch_norm(layer_h_n)
        # layer_h_n = self.tanh(layer_h_n)

        # layer_h_n = self.linear1(layer_h_n)
        # layer_h_n = self.relu(layer_h_n)

        return output, (layer_h_n, layer_c_n)


def show_data(train_Accuracy, train_Loss, test_Loss, test_Accuracy, scatter=False, dayin=False):  ## attention , we must use 'with open ' to open a txt file, if not. we won't write data in loss.
    with open('train_loss.txt', 'w') as file1:
        for tra_loss in train_Loss:
            file1.write('{}\n'.format(tra_loss))
        file1.close()

    with open('train_accuracy.txt', 'w') as file2:
        for tra_accu in train_Accuracy:
            file2.write('{}\n'.format(tra_accu))
        file2.close()

    with open('test_loss.txt', 'w') as file3:
        for tes_loss in test_Loss:
            file3.write('{}\n'.format(tes_loss))
        file3.close()

    with open('test_accuracy.txt', 'w') as file4:
        for tes_accu in test_Accuracy:
            file4.write('{}\n'.format(tes_accu))
        file4.close()

    if scatter == True:
        plt.scatter(range(len(train_Accuracy)), train_Accuracy, s=5)
        plt.title("Accuracy on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("train_Accuracy")
        plt.close()

        plt.scatter(range(len(train_Loss)), train_Loss, s=5)
        plt.title("Loss on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("train_Loss")
        plt.close()

        plt.scatter(range(len(test_Accuracy)), test_Accuracy, s=5)
        plt.title("Accuracy on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("test_Accuracy")
        plt.close()

        plt.scatter(range(len(test_Loss)), test_Loss, s=5)
        plt.title("Loss on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("test_Loss")
        plt.close()

    else:
        plt.plot(train_Accuracy)
        plt.title("Accuracy on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("train_Accuracy")
        plt.close()

        plt.plot(train_Loss)
        plt.title("Loss on train set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("train_Loss")
        plt.close()

        plt.plot(test_Accuracy)
        plt.title("Accuracy on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Accuracy %")
        plt.savefig("test_Accuracy")
        plt.close()


        plt.plot(test_Loss)
        plt.title("Loss on test set")
        plt.xlabel("every 20 iterations")
        plt.ylabel("Loss Value")
        plt.savefig("test_Loss")
        plt.close()














