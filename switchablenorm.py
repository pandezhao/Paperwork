import torch
from torch import nn
from torch.nn import init, Parameter
from torch.autograd import Variable
from functools import partial


def dm_SQD_switch(x, alpha=[0.2, 0.4, 0.6, 0.8, 1], dimension=2):
    x_num = x.size()[0]
    x_sort, _ = torch.sort(x, dim=0, descending=True)
    mean = torch.mean(x, dim=0)
    statistic = []
    deviation = []
    for i in range(alpha.__len__()):
        split_num = int(x_num * (1 - alpha))
        if split_num > x_num:
            split_num = x_num
        if split_num <= 1:
            split_num = 2
        if dimension==2:
            topk = x_sort[:split_num, :]
        elif dimension==3:
            topk = x_sort[:split_num, :, :]
        else:
            raise ValueError("must be 2-D or 3-D Tensor")
        statistic.append(torch.min(topk, dim=0)[0])
        deviation.append(torch.sum(topk, dim=0) / (split_num - 1) - mean)
    return torch.stack(statistic), torch.stack(deviation)


def dm_func(dm):
    if dm.__eq__('SQD'):
        return dm_SQD_switch
    if dm.__eq__('SQD1'):
        return partial(dm_SQD_switch, alpha=0.25)
    if dm.__eq__('SQD2'):
        return partial(dm_SQD_switch, alpha=0.5)
    if dm.__eq__('SQD3'):
        return partial(dm_SQD_switch, alpha=0.75)


class NewBatchNorm1D(nn.Module):

    def __init__(self, num_features, use_bias=True, alpha=0.25, momentum=0.1, measure='SQD', eps=1e-5, affine=True, record=True):
        super(NewBatchNorm1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.use_bias = use_bias
        self.momentum = momentum
        self.training = True
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.switch = nn.Parameter(torch.randn(5), requires_grad=True)
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
            statistics, dev = self.dev_calc(input_.data)
            statistics = torch.sum(torch.nn.Softmax(self.switch) * statistics.data, dim=0)
            dev = torch.sum(torch.nn.Softmax(self.switch) * dev.data, dim=0)

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