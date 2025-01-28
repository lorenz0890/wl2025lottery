import math
import random

import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected, add_self_loops, degree, one_hot
from gnns.utils import binary_to_int

class AddOneHotEncodedDegree(BaseTransform):
    def __init__(self, max_degree=None):
        self.max_degree = max_degree

    def __call__(self, data):
        # Compute the degree of each node
        d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)

        # If max_degree is not set, use the max degree in this graph
        max_degree = self.max_degree if self.max_degree is not None else d.max().item()

        # One-hot encode the degrees
        d_one_hot = one_hot(d, num_classes=max_degree + 1).to(torch.float)

        # Concatenate with existing node features, if they exist
        data.x = torch.cat([data.x, d_one_hot], dim=1) if data.x is not None else d_one_hot
        #data.num_node_features = max_degree
        return data

    def __repr__(self):
        return '{}(max_degree={})'.format(self.__class__.__name__, self.max_degree)


class AddBinaryEncodedDegree(BaseTransform):
    def __init__(self, max_degree=None):
        self.max_degree = max_degree

    def degree_to_binary(self, degree, max_length):
        """Converts a degree value to a binary tensor with padding."""
        binary = torch.tensor([int(x) for x in bin(degree)[2:]], dtype=torch.float)
        # Pad the binary representation to have the same length
        if len(binary) < max_length:
            padding = torch.zeros(max_length - len(binary), dtype=torch.float)
            binary = torch.cat((padding, binary), dim=0)
        return binary

    def __call__(self, data):
        # Compute the degree of each node
        d = degree(data.edge_index[0], dtype=torch.long)

        # Determine the max length for binary representation
        if self.max_degree is None:
            self.max_degree = d.max().item()
        max_length = len(bin(self.max_degree)[2:])

        # Convert and pad each degree to binary
        binary_degrees = torch.stack([self.degree_to_binary(deg, max_length) for deg in d])

        # Concatenate with existing node features, if they exist
        data.x = torch.cat([data.x, binary_degrees], dim=1) if data.x is not None else binary_degrees
        return data

    def __repr__(self):
        return '{}(max_degree={})'.format(self.__class__.__name__, self.max_degree)


class AddRandomBinaryFeatures(BaseTransform):
    def __init__(self, generator_length, feature_length):
        self.feature_length = feature_length
        self.generator_length = generator_length

    def __stochastic_rounding__(self, number):

        if isinstance(number, float) and number.is_integer():
            return int(number)
        elif isinstance(number, int):
            return number

        floor_number = math.floor(number)
        ceiling_number = math.ceil(number)

        # Proximity to the floor and ceiling
        proximity_to_floor = number - floor_number
        proximity_to_ceiling = ceiling_number - number

        # The probability of rounding to the ceiling is the proximity to the ceiling
        # and vice versa for the floor.
        probabilities = [proximity_to_ceiling, proximity_to_floor]

        # Choose whether to round down or up based on the proximity
        rounded_number = random.choices([floor_number, ceiling_number], weights=probabilities, k=1)[0]

        return rounded_number

    def __call__(self, data):
        stochastic_generator_length = self.__stochastic_rounding__(self.generator_length)
        # print(a, flush=True)
        num_nodes = data.num_nodes
        # Generate random binary features for each node and replace the original features
        data.x = torch.randint(0, 2, (num_nodes, stochastic_generator_length)).float()
        data.x = torch.cat([data.x, torch.ones((num_nodes, self.feature_length - stochastic_generator_length)).float()],
                           dim=1)
        return data

    def __repr__(self):
        return '{}(feature_length={})'.format(self.__class__.__name__, self.feature_length)

    def __repr__(self):
        return '{}(feature_length={}, homophily_ratio={})'.format(self.__class__.__name__, self.feature_length,
                                                                  self.homophily_ratio)

# Factory function to create the transform
def create_assign_unique_label_pre_transform():
    # Counter to keep track of the current index
    class AssignUniqueLabel(BaseTransform):
        def __init__(self):
            super(AssignUniqueLabel, self).__init__()
            self.index = 0  # Initialize the index counter

        def __call__(self, data):
            # Assign the current index as the label
            data.y = torch.tensor([self.index], dtype=torch.long)
            self.index += 1  # Increment the index for the next call
            return data

    return AssignUniqueLabel()
