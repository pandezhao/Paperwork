import torch
from torchvision import datasets,transforms
from torch import nn, optim
from torch.nn import init, functional
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import matplotlib.pyplot as plt

data_path = "CNN dataset"
save_dir = "CNN saved"
use_gpu = True
epochs = 500
iteration = 10000
global batch_size
batch_size = 32
hidden_size = 100

train_set = datasets.MNIST(data_path,train=True,transform=transforms.Compose([transforms.ToTensor()]),download=True)
test_set = datasets.MNIST(data_path,train=False,transform=transforms.Compose([transforms.ToTensor()]),download=True)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,pin_memory=True)

def reshape(x,batch_size):
    return torch.reshape(x,(batch_size,-1))

# for train_batch,label in train_loader:
#     print(len(train_batch))
#     print(len(train_batch[0]))
#     print(train_batch[0].size())
#     print(type(train_batch[0]))
#     a = reshape(train_batch,batch_size)
#     print(a.size())
#     print(label.size())
#     break
# a.t().size()
# a.t()[0].size()
#a[:,0].size()

def dm_SD(x):
    return torch.mean(x, 0), torch.std(x, 0)


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
    return (sup + inf) * 0.5, (sup - inf)


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


class SeparatedBatchNorm1d(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)


class NewBatchNorm(nn.Module):
    def __init__(self, num_features, max_length, eps=1e-5, affine=True, use_bias=True, momentum=0.5, dev_measure='SD', alpha=0.25):
        super(NewBatchNorm, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.eps = eps
        self.affine = affine
        self.use_bias = use_bias
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            if use_bias:
                self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        ###################################################
        # for i in range(max_length):
        #     self.register_buffer('running_mean_{}'.format(i), torch.zeros(batch_size,1))
        #     self.register_buffer('running_var_{}'.format(i), torch.ones(batch_size,1))
        ###################################################
        self.training = True
        self.dev_measure = dev_measure
        self.alpha = alpha
        self.dev_calc = dm_func(dev_measure)
        self.reset_parameters()

    ###################################################
    # def reset_running_state(self):
    #     running_mean_i = getattr(self, 'running_mean_{}'.format(0))
    #     running_var_i = getattr(self, 'running_var_{}'.format(0))
    #     running_mean_i.zero_()
    #     running_var_i.fill_(1)
    #     for i in range(self.max_length):
    #         setattr(self, 'running_mean_{}'.format(i), running_mean_i)
    #         setattr(self, 'running_var_{}'.format(i), running_var_i)
    ###################################################

    def reset_parameters(self):
        ###################################################
        # self.reset_running_state()
        ###################################################
        self.weight.data.uniform_()
        if self.use_bias:
            self.bias.data.zero_()

    def forward(self, input_, time):  ##目前的input是一个（32, 400）的矩阵
        ###################################################
        # running_mean_i = getattr(self, 'running_mean_{}'.format(time)) # (32)
        # running_var_i = getattr(self, 'running_var_{}'.format(time)) # (32)
        ###################################################
        if self.dev_measure.__eq__('SQD'):
            statistics, dev = self.dev_calc(input_, self.alpha)
        else:
            statistics, dev = self.dev_calc(input_)

        ###################################################
        # running_mean_i = (1 - self.momentum) * statistics + self.momentum * running_mean_i
        # running_var_i = (1 - self.momentum) * dev + self.momentum * running_var_i
        ###################################################

        running_mean_i = statistics
        running_var_i = dev

        if self.affine:
            if self.use_bias:
                bn_output = self.weight * (input_ - running_mean_i) / (running_var_i + self.eps) + self.bias
                #bn_output = (input_ - running_mean_i.view(-1, 1)) / (running_var_i + self.eps).view(-1, 1) + self.bias
            else:
                # weight_tmp = self.weight
                # input_tmp = input_
                # ###################
                # a = input_ - running_mean_i.view(-1,1)
                # weight_tmp = self.weight
                # print('test')
                # b = self.weight * a
                # c = running_var_i + self.eps
                # d = b / c.view(-1,1)
                # ###################
                bn_output = self.weight * (input_ - running_mean_i) / (running_var_i + self.eps)
                #bn_output = (input_ - running_mean_i.view(-1, 1)) / (running_var_i + self.eps).view(-1, 1)
        else:
            bn_output = (input_ - running_mean_i) / (running_var_i + self.eps)

        setattr(self, 'running_mean_{}'.format(time), running_mean_i)
        setattr(self, 'running_var_{}'.format(time), running_var_i)
        return bn_output


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, max_length, use_bias=True, momentum=0.9, dev_measure='SD', alpha=0.25):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.dev_measure = dev_measure
        self.alpha = alpha
        self.weight_hidden = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.weight_input = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.bn_h = NewBatchNorm(num_features=hidden_size * 4, max_length=784, affine=True, use_bias=False, momentum=0.5,
                                 dev_measure='SD', alpha=0.25)
        self.bn_i = NewBatchNorm(num_features=hidden_size * 4, max_length=784, affine=True, use_bias=False, momentum=0.5,
                                 dev_measure='SD', alpha=0.25)
        self.bn_c = NewBatchNorm(num_features=hidden_size, max_length=784, affine=True, use_bias=True, momentum=0.5, dev_measure='SD',
                                 alpha=0.25)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_input.data)
        ## initial of hidden weight
        weight_h_tmp = torch.eye(self.hidden_size)
        weight_h_tmp = weight_h_tmp.repeat(1, 4)
        self.weight_hidden.data.set_(weight_h_tmp)
        ## initial of bias
        if self.use_bias:
            init.constant_(self.bias.data, val=0)
        ## initialization of normalization weight
        ## 我认为这里不需要再一次进行reset parameter了，之前在实例化的时候已经算是reset过一次了
        # self.bn_h.bias.data.fill_(0)
        # self.bn_i.bias.data.fill_(0)

        self.bn_c.bias.data.fill_(0)
        self.bn_h.weight.data.fill_(0.1)
        self.bn_i.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        #if time == 0:
            #print('Cell s weight',self.weight_hidden)
            #print('Cell s weight',self.weight_input)
        h_0, c_0 = hx
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size,*self.bias.size()))
        wh = torch.addmm(bias_batch, h_0, self.weight_hidden)
        wi = torch.mm(input_, self.weight_input)
        bn_h = self.bn_h(wh, time=time)
        bn_i = self.bn_i(wi, time=time)
        # print('bn_h size:',bn_h.size())
        # print('bn_i size:',bn_i.size())
        # print('bias batch size',bias_batch.size())
        f, i, o, g = torch.split(bn_h + bn_i,split_size_or_sections=self.hidden_size, dim=1)
        # print('f',f.size())
        # print('i',i.size())
        # print('o',o.size())
        # print('g',g.size())
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
        return h_1, c_1


class LSTM(nn.Module):
    def __init__(self, cell_class, input_size, hidden_size, max_length, num_layers=2, use_bias=True, momentum=0.9,
                 dropout=0, dev_measure='SD',alpha=0.25):

        # cell_class='LSTMCell', input_size=1, hidden_size=100, max_length=784, num_layers=2, use_bias=True,
        # momentum=0.9, dropout=0, dev_measure='SD', alpha=0.25

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.momentum = momentum
        self.dropout = dropout
        self.dev_measure = dev_measure
        self.alpha = alpha

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            if cell_class == 'LSTMCell':
                # self, input_size, hidden_size, max_length, use_bias=True, momentum = 0.9,dev_measure='SD', alpha=0.25
                cell = LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size, max_length=max_length,
                                use_bias=self.use_bias, momentum=self.momentum, dev_measure = self.dev_measure, alpha = self.alpha)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = getattr(self, 'cell_{}'.format(layer))
            cell.reset_parameters()

    @staticmethod
    def _forward_layer(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            # if isinstance(cell, GBNLSTMCell):
            #   h_next, c_next = cell(input_ = input_[time],hx=hx,time = time)
            # elif isinstance(cell, LSTMCell):
            #   h
            # else:
            #   raise ValueError('must be a LSTM cell or GBNLSTM cell')
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx):
        h_n = []
        c_n = []
        layer_output = None
        max_time, batch_size = input_.size()

        # if length is None:
        #   if input_.is_cuda:
        #      device = input_.get_device()
        #     length = length.cuda(device)
        input_ = input_.view(max_time, batch_size, 1)
        for layer in range(self.num_layers):
            cell = getattr(self, 'cell_{}'.format(layer))
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_layer(
                cell=cell, input_=input_, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)


model = LSTM(cell_class='LSTMCell', input_size=1, hidden_size=100, max_length=784, num_layers=2, use_bias=True,
             momentum=0.9, dropout=0, dev_measure='SD', alpha=0.25)
fc = nn.Linear(in_features=100, out_features=10)
Loss = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(fc.parameters())
optimizer = optim.RMSprop(params=params, lr=1e-3, momentum=0.9)

train_Accuracy = []
train_Loss = []
test_Accuracy = []
test_Loss = []

if use_gpu:
    model.cuda()
    fc.cuda()


def reshape(x, batch_size):
    return torch.reshape(x, (batch_size, -1))


def compute_loss(data, label):
    hx = data.new_zeros(batch_size, hidden_size, requires_grad=False)
    hx = (hx, hx)
    type(hx)

    _, (h_n, _) = model(input_=data,hx=hx)
    #print("h_n: ",h_n.size())
    output = fc(h_n[0])
    #print(output.size())
    loss = Loss(input=output, target=label)
    accuracy = (output.max(1)[1] == label).float().mean()
    return loss, accuracy


for epoch in range(epochs):
    count = 0
    for data, label in train_loader:
        data = Variable(data.view(-1,batch_size))
        label = Variable(label)
        if use_gpu:
            data = data.cuda()
            label = label.cuda()

        model.train(True)
        model.zero_grad()
        optimizer.zero_grad()

        loss, accuracy = compute_loss(data, label)
        #print('loss:',loss)
        #print('accuracy:',accuracy)
        # if count == 0:
        #     loss.backward(retain_graph = True)
        # else:
        loss.backward()
        clip_grad_norm(parameters = params, max_norm=1)
        optimizer.step()
        count += 1
        train_Accuracy.append(accuracy)
        train_Loss.append(loss)
        if count % 1 ==0:
            print('Epoch:{},iteration:{},train_loss:{},train_accuracy:{}'.format(epoch, count, loss, accuracy))

        if count == 2000:
            break

plt.plot(train_Accuracy)
plt.title("Accuracy on train set")
plt.xlabel("every 20 iterations")
plt.ylabel("Accuracy %")
plt.show()

plt.plot(train_Loss)
plt.title("Loss on train set")
plt.xlabel("every 20 iterations")
plt.ylabel("Loss Value")
plt.show()

plt.plot(test_Accuracy)
plt.title("Accuracy on test set")
plt.xlabel("every 20 iterations")
plt.ylabel("Accuracy %")
plt.show()

plt.plot(test_Loss)
plt.title("Loss on test set")
plt.xlabel("every 20 iterations")
plt.ylabel("Loss Value")
plt.show()