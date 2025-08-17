"""
Utility functions for loading models, setting random seeds, and multi-GPU support.
"""
import numpy as np
import torch
import os

def load_model(model: torch.nn.Module, model_path: str, verbose: bool = True):
    """
    Load the model state from the specified path.
    If the model is not found, it will print an error message.

    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str): The path to the model file.
        verbose (bool): If True, prints messages about the loading process.
    Returns:
        torch.nn.Module: The model with loaded state.
    """

    # Check if the model path exists
    if not os.path.exists(model_path):
        if verbose:
            print("\033[91m" + f"[UTILS] Model file not found at {model_path}" + "\033[0m")  # RED
        return model, 0.0

    # Attempt to load the model state
    try:
        loaded_data = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded_data['model_state_dict'])  # Load the model state
        HighestAccuracy = loaded_data.get('highest_accuracy', 0.0)  # Default to 0.0 if not found
        if verbose:
            print("\033[92m" + f"[UTILS] Model loaded successfully from {model_path} with accuracy {HighestAccuracy:.2f}%" + "\033[0m")  # GREEN

    except Exception as e:
        if verbose:
            print("\033[91m" + f"[UTILS] Error loading model: {e}" + "\033[0m")  # RED
        HighestAccuracy = 0.0  # Default to 0.0 if loading fails.

    return model, HighestAccuracy

def model_to_multi_gpu(model: torch.nn.Module, verbose: bool = True):
    """
    Convert the model to use multiple GPUs if available.
    This function uses DataParallel instead of DistributedDataParallel for simplicity.

    Args:
        model (torch.nn.Module): The model to convert.
        verbose (bool): If True, prints messages about the conversion process.
    Returns:
        torch.nn.Module: The model wrapped for multi-GPU training.
    """
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        if verbose:
            print("\033[92m" + f"[UTILS] Model set to use {torch.cuda.device_count()} GPUs with DataParallel." + "\033[0m")  # GREEN

    return model

def set_random_seed(Seed: int):
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed)
    print("\033[95m" + f"[UTILS] Random seed set to {Seed}" + "\033[0m")  # PINK

if __name__ == "__main__":
    print("This is a utility module and is not meant to be run directly.")
