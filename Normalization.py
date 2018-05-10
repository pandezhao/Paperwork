import torch
from torch.nn import functional, init
from torch import nn
from torch.autograd import Variable
import numpy as np
import fuel.datasets
import fuel.streams
import fuel.transformers
import fuel.schemes


class SeparatedBatchNorm1d(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.FloatTensor(num_features))
        self.bias = nn.Parameter(torch.FloatTensor(num_features))
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

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


def zeros(shape):
    return torch.zeros(shape)


def ones(shape):
    return torch.ones(shape)


def glorot(shape):
    d = np.sqrt(6. / sum(shape))
    return torch.from_numpy(np.random.uniform(-d, +d, size=shape))


_datasets = None


def get_dataset(which_set):
    global _datasets
    if not _datasets:
        MNIST = fuel.datasets.MNIST
        # jump through hoops to instantiate only once and only if needed
        _datasets = dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
    return _datasets[which_set]


def get_stream(which_set, batch_size, num_examples=None):
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream


class GBN:
    def __init__(self, X, gammas, betas = None, alpha = None):
        self.X = X
        self.gammas = gammas
        self.betas = betas
        self.alpha = alpha

    def quantile(self, X):
        X = torch.sort(X)
        tmp = np.floor( np.size(X[0])()[0] * self.alpha )
        return X[0][int(tmp)],int(tmp)

    def fakerelu(self, X):
        return X[np.where(X>0)]

    def SD(self):
        s = torch.mean(self.X)
        d = torch.sqrt(torch.mean(torch.pow(torch.sub(self.X, s), 2)))
        out = torch.div(torch.sub(self.X, s), torch.add(torch.pow(d, 2), 1e-5))
        if betas == None:
            return torch.mul(out, self.gammas)
        else:
            return torch.add(torch.mul(out, self.gammas), self.betas)

    def MAD(self):
        s = torch.mean(self.X)
        d = torch.mean(torch.abs(torch.sub(self.X, s)))
        out = torch.div(torch.sub(self.X, s), torch.add(torch.pow(d, 2), 1e-5))
        if betas == None:
            return torch.mul(out, self.gammas)
        else:
            return torch.add(torch.mul(out, self.gammas), self.betas)

    def RSD(self):
        s = torch.mean(self.X)
        d = torch.mean(self.fakerelu(torch.sub(self.X, s)))
        out = torch.div(torch.sub(self.X, s), torch.add(torch.pow(d, 2), 1e-5))
        if betas == None:
            return torch.mul(out, self.gammas)
        else:
            return torch.add(torch.mul(out, self.gammas), self.betas)

    def SQD(self):
        s,tmp = self.quantile(self.X)
        d = torch.mean(torch.sort(self.X)[0][tmp:-1]) - torch.mean(self.X)
        out = torch.div(torch.sub(self.X, s), torch.add(torch.pow(d, 2), 1e-5))
        if betas == None:
            return torch.mul(out, self.gammas)
        else:
            return torch.add(torch.mul(out, self.gammas), self.betas)

    def RBD(self):
        s = 0.5 * (torch.max(self.X) + torch.min(self.X))
        d = torch.max(self.X) - torch.min(self.X)
        out = torch.div(torch.sub(self.X, s), torch.add(torch.pow(d, 2), 1e-5))
        if betas == None:
            return torch.mul(out, self.gammas)
        else:
            return torch.add(torch.mul(out, self.gammas), self.betas)

    def WCD(self):
        s = torch.max(self.X)
        d = torch.max(self.X) - torch.mean(self.X)
        out = torch.div(torch.sub(self.X, s), torch.add(torch.pow(d, 2), 1e-5))
        if betas == None:
            return torch.mul(out, self.gammas)
        else:
            return torch.add(torch.mul(out, self.gammas), self.betas)


activations = dict(
    tanh=T.tanh,
    identity=lambda x: x,
    relu=lambda x: T.max(0, x))

class Empty:
    pass

class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BNLSTMCell(nn.Module):

    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(BNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        # BN parameters
        self.bn_ih = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(
            num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        # The input-to-hidden weight matrix is initialized orthogonally.
        init.orthogonal(self.weight_ih.data)
        # The hidden-to-hidden weight matrix is initialized as an identity
        # matrix.
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        init.constant(self.bias.data, val=0)
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            time: The current timestep value, which is used to
                get appropriate running statistics.
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
                                 split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
        return h_1, c_1


class GBNLSTMCell(nn.Module):

    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(GBNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        # GBN parameters
        self.bn_ih = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(
            num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        # The input-to-hidden weight matrix is initialized orthogonally.
        init.orthogonal(self.weight_ih.data)
        # The hidden-to-hidden weight matrix is initialized as an identity
        # matrix.
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        init.constant(self.bias.data, val=0)
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            time: The current timestep value, which is used to
                get appropriate running statistics.
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
                                 split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
        return h_1, c_1

'''
class GBNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, max_length):

        super(GBNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.weight_ht = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_

    def allocate_parameters(self, args):
        h0 = Variable(zeros((args.num_hidden,)), name = "h0")
        c0 = Variable(zeros((args.num_hidden,)), name = "c0")
        Wa = Variable(orthogonal((1, 4 * args.num_hidden)), requires_grad=True, name = "Wa")
        Wx = Variable(orthogonal((1, 4 * args.num_hidden)), requires_grad=True, name="Wx")
        a_gammas = Variable(args.initial_gamma * ones((4 * args.num_hidden,)), requires_grad=True, name = "a_gammas")
        b_gammas = Variable(args.initial_gamma * ones((4 * args.num_hidden,)), requires_grad=True, name = "b_gammas")
        ab_betas = Variable(args.initial_beta * ones((4 * args.num_hidden,)), requires_grad=True, name="ab_betas")

        ab_betas[args.num_hidden : 2 * args.num_hidden] = 1
        c_gammas = Variable(args.initial_gamma * ones((args.num_hidden,)), requires_grad=True, name="c_gammas")
        c_betas  = Variable(args.initial_beta  * ones((args.num_hidden,)), requires_grad=True, name="c_betas")

        parameters = [h0, c0, Wa, Wx, a_gammas, b_gammas, ab_betas, c_gammas, c_betas]
        return parameters

    def forward(self, x, hx, args):
        """
        :param x: Tensor (batch, input_size) tensor
        :param hx: A tuple (h_0, c_0), hidden and cell state. the size is both (batch, hidden_size)
        :return: h_1, c_1: the tensors containing next hidden and cell state
        """
        parameters = self.allocate_parameters(args)
        [h0, c0, Wa, Wx, a_gammas, b_gammas, ab_betas, c_gammas, c_betas] = parameters
        batch_size = h_0.size(0)
        wh = torch.mm(h_0, self.weight_ht)
        wx = torch.mm(x, self.weight_hx)
        wh_N = GBN(wh, a_gammas)
        wx_N = GBN(wx, b_gammas)
        f, i, o, g = torch.split(wh_N.SD() + wx_N.SD() + beta, split_size = self.hidden_size, dim = 1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1_N = GBN(c_1,c1_gammas)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1_N.SD())

        return h_1, c_1
'''

class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            if isinstance(cell, BNLSTMCell):
                h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            else:
                h_next, c_next = cell(input_=input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)
































