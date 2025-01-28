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


def enforce_positive_percentage(tensor, positive_percentage=0.5):
    """
    Adjusts a tensor to ensure that a specified percentage of its elements are positive.

    Args:
        tensor (torch.Tensor): The tensor to modify.
        positive_percentage (float): The desired fraction of elements that should be positive (0 to 1).

    Returns:
        torch.Tensor: The modified tensor with the specified percentage of positive elements.
    """
    # Flatten the tensor for easier manipulation
    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.numel()

    # Calculate the number of elements that should be positive
    num_positive = int(positive_percentage * num_elements)

    # Sort indices by value (ascending order)
    sorted_indices = torch.argsort(flat_tensor)

    # Set the largest `num_positive` elements to be positive (if not already)
    for i in range(num_positive):
        index = sorted_indices[-(i+1)]
        if flat_tensor[index] <= 0:
            flat_tensor[index] = torch.abs(flat_tensor[index]) + 1e-6  # Make it slightly positive

    # Reshape back to the original shape
    return flat_tensor.view(tensor.shape)

def generate_full_rank_tensor(tensor, rank_type='column'):
    """
    Generates a tensor of the specified shape that has full column or row rank.

    Args:
        shape (tuple): The shape of the tensor (n, m).
        rank_type (str): Either 'column' or 'row' to indicate which rank should be full.

    Returns:
        torch.Tensor: A tensor with full column or row rank and random elements.
    """
    n, m = tensor.shape
    assert rank_type in ['column', 'row'], "rank_type must be either 'column' or 'row'"

    # Generate a random matrix
    #tensor = torch.randn(shape)

    # If full column rank is required
    if rank_type == 'column':
        # Ensure n >= m for full column rank (more rows than columns)
        if n < m:
            raise ValueError("Shape must have more rows than columns for full column rank")

        # Apply QR decomposition to ensure full column rank
        q, _ = torch.qr(tensor)
        tensor = q[:, :m]  # Truncate to the required number of columns

    # If full row rank is required
    elif rank_type == 'row':
        # Ensure m >= n for full row rank (more columns than rows)
        if m < n:
            raise ValueError("Shape must have more columns than rows for full row rank")

        # Apply QR decomposition to ensure full row rank
        q, _ = torch.qr(tensor.t())
        tensor = q[:, :n].t()  # Truncate and transpose to get full row rank

    return tensor

def init_and_prune_weights_maintain_full_rank_positive(model, sparsity=0.5, num_attempts=100, positive_percentage=0.75):
    """
    Prunes weights of the model by randomly setting a fraction of the weights to zero
    multiple times, and selects the pruning pattern that maintains the highest rank.

    Args:
        model (nn.Module): The MLP model whose weights are to be pruned.
        sparsity (float): The fraction of weights to prune (set to zero).
        num_attempts (int): Number of random pruning attempts to maximize rank.

    Returns:
        None
    """
    with torch.no_grad():
        # Iterate over all layers in the model
        for name, param in model.named_parameters():
            if 'conv' in name:
                # Check if the parameter is trainable and has gradients
                if 'weight' in name and param.requires_grad:
                    # Get the original shape of the weight matrix
                    orig_shape = param.shape
                    weight_matrix = param.view(orig_shape[0], -1)
                    weight_matrix = generate_full_rank_tensor(weight_matrix, 'row')
                    if positive_percentage > 0:
                        weight_matrix = enforce_positive_percentage(weight_matrix, positive_percentage)
                    num_weights = weight_matrix.numel()

                    # Calculate the number of weights to prune
                    num_prune = int(sparsity * num_weights)

                    best_rank = 0
                    best_mask = None

                    # Try multiple random pruning patterns
                    for _ in range(num_attempts):
                        # Generate a random pruning mask
                        mask_flat = torch.ones(num_weights).to(weight_matrix.get_device())
                        prune_indices = random.sample(range(num_weights), num_prune)
                        mask_flat[prune_indices] = 0
                        mask = mask_flat.view_as(weight_matrix)

                        # Apply the mask to the weight matrix and calculate its rank
                        pruned_matrix = weight_matrix * mask
                        current_rank = torch.linalg.matrix_rank(pruned_matrix)

                        # Keep track of the mask that maintains the highest rank
                        if current_rank > best_rank:
                            best_rank = current_rank
                            best_mask = mask

                    # Apply the best mask found
                    param.copy_((weight_matrix * best_mask).view(orig_shape))