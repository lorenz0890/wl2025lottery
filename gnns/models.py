import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP ,GINConv #GCNConv
from torch_geometric.nn import global_add_pool

from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

from gnns.convs import PGINConv, GCNConv


# from gnns.simple_baselines import GIN, GCN
# from isomorphism_tests.weisfeiler_leman import weisfeiler_leman_1wl
class PGIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4):
        super(PGIN, self).__init__()
        self.convs = torch.nn.ModuleList()  # Use ModuleList to hold GINConv layers

        # Dynamically create the GINConv layers based on num_layers
        for _ in range(num_layers):
            mlp = MLP([num_features, num_features*2, num_features*4], batch_norm=False, bias=False, plain_last=False) # plain_last = False for logging!
            self.convs.append(PGINConv(mlp))

        self.fc1 = torch.nn.Linear(num_features, num_classes) #*num_layers
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, graph):
        x, edge_index, batch = graph.x.cuda(), graph.edge_index.cuda(), graph.batch

        # Apply each GINConv layer in a loop
        #xlist = []
        for conv in self.convs:
            x = conv(x, edge_index)#.relu()
            #xlist.append(x)
        #x = torch.cat(xlist, dim=1)
        x = global_add_pool(x, batch)  # Pooling for graph-level classification
        x = self.fc1(x)
        x = self.drop(x)
        return F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, activation = F.sigmoid):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()  # Use ModuleList to hold GINConv layers

        # Dynamically create the GINConv layers based on num_layers
        #def act(x):
        #    return F.relu(F.sigmoid(x))-F.sigmoid(x*0)

        for _ in range(num_layers):
            mlp = MLP([num_features, num_features, num_features], batch_norm=False, bias=False, plain_last=False, act=activation)# plain_last = False for logging!
            self.convs.append(GINConv(mlp))

        self.fc1 = torch.nn.Linear(num_features, num_classes) #*num_layers
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, graph):
        x, edge_index, batch = graph.x.cuda(), graph.edge_index.cuda(), graph.batch

        '''
        for name, param in self.named_parameters():
            if 'conv' in name:
                with torch.no_grad():
                    #param.requires_grad = True
                    param.copy_((param.abs()+1e-6) * torch.eye(param.shape[0]).cuda())
        '''
        # Apply each GINConv layer in a loop
        #xlist = []
        for conv in self.convs:
            x = conv(x, edge_index)#.relu()
            #x = self.projector(x)
            #xlist.append(x)
        #x = torch.cat(xlist, dim=1)
        xr = global_add_pool(x, batch)  # Pooling for graph-level classification
        x = self.fc1(xr)
        x = self.drop(x)
        return F.log_softmax(x, dim=1), xr


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, activation=F.relu):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Add GCN layers
        for _ in range(num_layers):
            self.convs.append(GCNConv(num_features, num_features, activation=activation, bias=False))

        self.fc = torch.nn.Linear(num_features, num_classes)
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x.cuda(), data.edge_index.cuda()

        for conv in self.convs:
            x = conv(x, edge_index) #Relu included in layer for metrics
        xr = global_add_pool(x, data.batch)  # Pooling for graph-level classification
        x = F.dropout(xr, p=0.5, training=self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), xr