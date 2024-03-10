import torch
import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout, use_bn):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Invalid type for input_dims!'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers['fc{}'.format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(l_i)] = nn.ReLU()
            layers['drop{}'.format(l_i)] = nn.Dropout(dropout)
            if use_bn:
                layers['bn{}'.format(l_i)] = nn.BatchNorm1d(n_hidden)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)
    
    
class MLP_Discriminator(nn.Module):
    def __init__(self, embed_dim=768 ):
        super(MLP_Discriminator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_state = 500
        self.hidden_layers = 1
        self.hidden_dropout = 0.3
        self.input_dropout = 0.6
        self.use_bn = False
        self.bidirectional = True
       

        self.mlp = MLP(self.embed_dim * 5, [self.hidden_state] * self.hidden_layers,
                       1, self.hidden_dropout, self.use_bn)
        self.dropout = nn.Dropout(self.input_dropout)
        if self.bidirectional:
            self.backward_mlp = MLP(embed_dim * 5, [self.hidden_state] * self.hidden_layers,
                                    1, self.hidden_dropout, self.use_bn)
            self.backward_dropout = nn.Dropout(self.input_dropout)

    def forward(self, s1, s2):
        inputs = torch.cat([s1, s2, s1 - s2, s1 * s2, torch.abs(s1 - s2)], -1)
        scores = self.mlp(self.dropout(inputs))
        if self.bidirectional:
            backward_inputs = torch.cat(
                [s2, s1, s2 - s1, s1 * s2, torch.abs(s1 - s2)], -1)
            backward_scores = self.backward_mlp(
                self.backward_dropout(backward_inputs))
            scores = (scores + backward_scores) / 2
        return scores    

