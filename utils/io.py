
def generate_filename(args):
    """
    Generate a filename based on argparse arguments.

    Parameters:
    - args: Namespace object containing command line arguments.

    Returns:
    - filename: A string representing the filename constructed from the arguments.
    """
    filename_parts = [
        f"model_{args.model}",
        f"depth_{args.depth}",
        f"activation_{args.activation}",
        f"numClean_{args.num_clean}",
        f"numDirty_{args.num_dirty}",
        f"prunePerc_{int(args.prune_percentage * 100)}",  # Convert to percentage
        f"dataset_{args.dataset}"
    ]
    if hasattr(args, 'feature_dim'): filename_parts.append(f"featureDim_{args.feature_dim}")
    if hasattr(args, 'homophily_modifier'): filename_parts.append(f"homophilyModifier_{args.homophily_modifier}")
    filename = "_".join(filename_parts) + ".json"
    return filename