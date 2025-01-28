import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import Aggregation, MeanAggregation, SumAggregation

def binary_to_int(binary_vector):
    # Assuming binary_vectors is a 2D tensor with shape [num_nodes, feature_size]
    # Calculate powers of 2 for each bit position
    num_bits = binary_vector.size(1)
    powers_of_two = 2 ** torch.arange(num_bits - 1, -1, -1).to(binary_vector.device).to(torch.float32)

    # Use matmul to perform the binary to integer conversion for the whole batch
    integer_values = torch.matmul(binary_vector.to(torch.float32), powers_of_two).to(torch.long)
    return integer_values


def compute_sparsity_masks(model, active_masks):
    """
    Compute a sparsity mask for each weight parameter in the model.

    Args:
        model (nn.Module): The model for which to compute sparsity masks.

    Returns:
        masks (dict): A dictionary containing the masks for each parameter.
    """
    masks = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'conv' in name:
                if 'weight' in name:  # We are only interested in weight parameters
                    if active_masks:
                        mask = (param != 0).float()  # Create a mask: 1 where weight is non-zero, 0 where it is zero
                        masks[name] = mask
                    else:
                        masks[name] = torch.ones_like(param).float()
    return masks


def apply_sparsity_masks(model, masks):
    """
    Apply sparsity masks to the model parameters to maintain sparsity after optimizer steps.

    Args:
        model (nn.Module): The model whose parameters will be masked.
        masks (dict): A dictionary containing the masks for each parameter.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'conv' in name:
                if name in masks:  # Apply mask if it exists for this parameter
                    param.data = param.data * masks[name]


def compute_gradient_sparsity(model, masks, eps=1e-6):
    """
    Compute the average gradient sparsity for each layer, only considering the non-masked weight elements.
    Sparsity is computed using torch.clamp with an epsilon margin to avoid numerical precision issues.

    Args:
        model (nn.Module): The model whose gradients will be analyzed.
        masks (dict): A dictionary containing the sparsity masks for each parameter.

    Returns:
        grad_sparsity (dict): A dictionary containing the average gradient sparsity per layer.
    """
    grad_sparsity = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'conv' in name:
                if name in masks and param.grad is not None:
                    mask = masks[name]
                    grad = param.grad

                    # Only consider the non-zero (non-masked) elements
                    active_grad = grad * mask


                    # Calculate sparsity: 1 - (non-zero elements / total elements)
                    non_zero_count = torch.count_nonzero(active_grad)
                    total_count = active_grad.numel()
                    sparsity = (1 - (non_zero_count / total_count)).detach().cpu().item()

                    # Store sparsity for the current layer
                    grad_sparsity[name] = sparsity
    return grad_sparsity


def compute_dead_neurons(model, masks, eps=1e-6):
    """
    Compute the percentage of dead neurons per layer based on the gradients, considering the sparsity masks
    and accounting for numerical precision using torch.clamp.

    Args:
        model (nn.Module): The model whose gradients will be analyzed.
        masks (dict): A dictionary containing the sparsity masks for each parameter.
    Returns:
        dead_neurons (dict): A dictionary containing the percentage of dead neurons per layer.
    """
    dead_neurons = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None and name in masks:
                if 'conv' in name:
                    # Apply the sparsity mask to the gradients
                    mask = masks[name]
                    active_grad = param.grad * mask  # Zero out the gradients for masked (pruned) elements

                    active_neuron_gradients = active_grad.abs().sum(dim=1)  

                    dead_count = (active_neuron_gradients == 0).sum().item()  # Count neurons with zero gradients
                    total_neurons = active_neuron_gradients.numel()  # Total number of neurons (rows)
                    dead_neuron_percentage = dead_count / total_neurons  # Percentage of dead neurons
                    dead_neurons[name] = dead_neuron_percentage
    return dead_neurons


def train(model, optimizer, device, loss_fn, epochs, train_data, active_masks=True):
    total_loss = []
    all_grad_sparsities = {}
    all_dead_neurons = {}
    sparsity_masks = compute_sparsity_masks(model, active_masks=active_masks)
    for epoch in range(epochs):
        epoch_loss = []
        model.train()
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(data)

            if isinstance(out, tuple):
                out = out[0]  

            out = out.unsqueeze(0) if out.dim() == 1 else out
            data.y = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y  

            loss = loss_fn(out, data.y)
            loss.backward()
            grad_sparsity = compute_gradient_sparsity(model, sparsity_masks, torch.finfo(torch.float32).eps)
            dead_neurons = compute_dead_neurons(model, sparsity_masks, torch.finfo(torch.float32).eps)
            for layer, sparsity in grad_sparsity.items():
                if layer not in all_grad_sparsities:
                    all_grad_sparsities[layer] = []
                all_grad_sparsities[layer].append(sparsity)
            for layer, neurons in dead_neurons.items():
                if layer not in all_dead_neurons:
                    all_dead_neurons[layer] = []
                all_dead_neurons[layer].append(neurons) 

            optimizer.step()
            apply_sparsity_masks(model, sparsity_masks)
            epoch_loss.append(loss.item())

        # Append average loss for the epoch
        total_loss.append(np.array(epoch_loss).mean())

    for layer, sparsities in all_grad_sparsities.items():
        all_grad_sparsities[layer] = np.mean(sparsities)
    for layer, dead_percentages in all_dead_neurons.items():
        all_dead_neurons[layer] = np.mean(dead_percentages)
    return total_loss, all_grad_sparsities, all_dead_neurons


class LayerMetrics:
    def __init__(self):
        self.metrics = {'o_rank' : [], 'io_refinement': [], 'o_var' : []}

    def __call__(self, module, input, output):
        with torch.no_grad():
            
            out_rank = torch.linalg.matrix_rank(output).detach().cpu().item()
            
            unique_rows, counts = torch.unique(output, dim=0, return_counts=True)
            num_unique_out_rows = unique_rows.shape[0]

            unique_rows, counts = torch.unique(input[0], dim=0, return_counts=True)
            num_unique_in_rows = unique_rows.shape[0]

           
            out_var = torch.var(output).detach().cpu().item()

           
            self.metrics['io_refinement'].append(num_unique_out_rows/ num_unique_in_rows)
            ratio = out_var
            self.metrics['o_var'].append(ratio)
            ratio = out_rank
            self.metrics['o_rank'].append(ratio)

    def average_metric(self):
        # Calculate the average rank
        for key in self.metrics:
            self.metrics[key] = np.array(self.metrics[key]).mean()
        return self.metrics


def encode_features_to_labels(node_features):
    unique_features, labels = np.unique(node_features, axis=0, return_inverse=True)
    return torch.tensor(labels, dtype=torch.long)


def get_feature_based_label_homophily_score(dataset):
    homophily_scores = []
    for data in dataset:
        node_features = data.x.cpu().numpy()
        labels = encode_features_to_labels(node_features)
        edge_index = data.edge_index.cpu()
        source_labels = labels[edge_index[0]]
        target_labels = labels[edge_index[1]]

        similarity_scores = (source_labels == target_labels).float()
        homophily_score = similarity_scores.mean().item()
        homophily_scores.append(homophily_score)

    return np.mean(homophily_scores), np.std(homophily_scores)

# Both methods below measure expressivity, the reported correlations should work with both.
'''
def count_distinguishable_tensors(tensor_list):
    epsilon = torch.finfo(tensor_list[0].dtype).eps  # Machine epsilon for the tensor's dtype
    n = len(tensor_list)
    distinguishable = [True] * n  # Initially assume all tensors are distinguishable

    for i in range(n):
        for j in range(i + 1, n):
            if distinguishable[i] or distinguishable[j]:
                # Check if the absolute difference exceeds epsilon for any element
                if torch.any(torch.abs(tensor_list[i] - tensor_list[j]) > 0): #We dont do this right now bc influence of perturbation is not clear
                    continue
                else:
                    # Mark tensors as indistinguishable if no element differs by more than epsilon
                    distinguishable[i] = distinguishable[j] = False

    # Count the number of true entries in distinguishable
    return sum(distinguishable) #durch n mal n-1 und durch diagnale gleich 0bei der matrix
'''
def count_distinguishable_tensors(tensor_list):
    epsilon = torch.finfo(tensor_list[0].dtype).eps  # Machine epsilon for the tensor's dtype
    n = len(tensor_list)
    distinguishable = np.ones((n,n))#[True] * n  # Initially assume all tensors are distinguishable
    distinguishable -= np.eye(n)

    for i in range(n):
        for j in range(n):
            #if distinguishable[i] or distinguishable[j]:
            #    # Check if the absolute difference exceeds epsilon for any element
            if torch.any(torch.abs(tensor_list[i] - tensor_list[j]) > 0):
                continue
            else:
                # Mark tensors as indistinguishable if no element differs by more than epsilon
                distinguishable[i, j] = 0#= distinguishable[j] = False

    # Count the number of true entries in distinguishable

    return sum(sum(distinguishable))/(n*(n-1))
def eval(model, device, loss_fn, eval_data, depth=5):
    hooks = {}
    hook_handles = []
    for name, module in model.named_modules():
        #print(name)
        is_conv = name in ['convs.{}'.format(i) for i in range(0, depth)]
        is_mlp = name in ['convs.{}.nn'.format(i) for i in range(0, depth)]
        is_linear = name in ['convs.{}.lin'.format(i) for i in range(0, depth)]
        is_aggr = name in ['convs.{}.aggr_module'.format(i) for i in range(0, depth)]

        if is_conv or is_mlp or is_linear or is_aggr:
            hook = LayerMetrics()
            hook_handles.append(module.register_forward_hook(hook))
            hooks['{}_{}{}'.format(name, int(is_mlp or is_linear), int(is_conv))] = hook

    model.eval()

    # Evaluating
    total_loss = []
    correct_predictions = 0
    total_graphs = 0
    embeddings, homophily = [],[]
    for data in eval_data:
        data = data.to(device)

        # Homophily (feature based - double check the meaning of this!)
        #edge_index = data.edge_index.cpu()
        #node_features = data.x.cpu()
        #source_features = node_features[edge_index[0]].long()
        #target_features = node_features[edge_index[1]].long()
        #similarity_scores = (source_features & target_features).float().sum(dim=1)
        #homophily_score = (similarity_scores > 0).float().mean().item()

        node_features = data.x.cpu().numpy()
        labels = encode_features_to_labels(node_features)
        edge_index = data.edge_index.cpu()
        source_labels = labels[edge_index[0]]
        target_labels = labels[edge_index[1]]
        similarity_scores = (source_labels == target_labels).float()
        homophily_score = similarity_scores.mean().item()
        homophily.append(homophily_score)

        # Forward pass
        with torch.no_grad():
            out, xr = model(data)  
        out = out.unsqueeze(0) if out.dim() == 1 else out  
        data.y = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y 

        loss = loss_fn(out, data.y)
        total_loss.append(loss.item())

        embeddings.append(xr)
        # Compute accuracy
        _, predicted = torch.max(out, dim=1)
        correct_predictions += (predicted == data.y).sum().item()
        total_graphs += data.y.size(0)
    total_loss = np.array(total_loss).mean()

    tensor = torch.stack(embeddings) # Graph embeddings
    tensor_cpu = tensor.cpu()

    num_unique_vectors = count_distinguishable_tensors([t.squeeze() for t in tensor_cpu])

    test_accuracy = correct_predictions / total_graphs

    metrics = {name: hook.average_metric() for name, hook in hooks.items()}
    for hook in hook_handles: hook.remove()

    return test_accuracy, total_loss, metrics, num_unique_vectors, sum(homophily)/len(homophily)
