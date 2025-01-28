import torch
import torch.nn as nn
import random

def prune_weights(model, sparsity=0.5):
    """
    Randomly prunes weights of the model by setting a fraction of the weights to zero.
    
    Args:
        model (nn.Module): The GNN model whose weights are to be pruned.
        sparsity (float): The fraction of weights to prune (set to zero).
    
    Returns:
        None
    """
    with torch.no_grad():
        # Iterate over all layers in the model
        for name, param in model.named_parameters():
            if 'conv' in name:
                # Check if the parameter is trainable and has gradients
                if 'weight' in name and param.requires_grad:
                    #print(f"Pruning {name} with sparsity {sparsity}")

                    # Flatten the weights for easier random sampling
                    weight_flat = param.view(-1)
                    num_weights = weight_flat.size(0)

                    # Determine how many weights to prune
                    num_prune = int(sparsity * num_weights)

                    # Randomly select indices to prune
                    prune_indices = random.sample(range(num_weights), num_prune)

                    # Set selected weights to zero
                    weight_flat[prune_indices] = 0

                    # Reshape the flattened weights back to their original shape
                    param.copy_(weight_flat.view(param.size()))

