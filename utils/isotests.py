import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_networkx
from gnns.utils import binary_to_int



def partition_complexity(dataset, iterations=1):
    total_complexity = {i: [] for i in range(iterations)}

    for data in dataset:
        colors = binary_to_int(data.x)
        old = len(set(list(colors.numpy())))
        wl = torch_geometric.nn.conv.WLConv()
        for i in range(iterations):
            colors = wl(colors, data.edge_index)
            new = len(set(list(colors.numpy())))

            total_complexity[i].append(new / old if old > 0 else 0)
            old=new

    for key in total_complexity:
        total_complexity[key] = np.array(total_complexity[key]).mean()
    return total_complexity


