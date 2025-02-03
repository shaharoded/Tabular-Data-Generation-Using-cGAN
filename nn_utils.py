
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_network(config):
    """
    Build a network based on the provided configuration.

    Args:
        config (list[dict]): Layer-wise network configuration. Each layer can specify:
            - input_dim (int): Input dimension of the layer.
            - output_dim (int): Output dimension of the layer.
            - activation (nn.Module, optional): Activation function for the layer.
            - batch_norm (bool or str, optional): If "batch", applies BatchNorm1d; if "layer", applies LayerNorm.
            - dropout (float, optional): Dropout rate (0.0 to 1.0).

    Returns:
        nn.Sequential: Sequential model based on the configuration.
    """
    layers = []
    for layer in config:
        in_dim, out_dim = layer["input_dim"], layer["output_dim"]
        layers.append(nn.Linear(in_dim, out_dim))

        # Add activation function if specified
        if "activation" in layer and layer["activation"]:
            layers.append(layer["activation"])

        # Add normalization if specified
        if "norm" in layer:
            if layer["norm"] == "batch":
                layers.append(nn.BatchNorm1d(out_dim))
            elif layer["norm"] == "layer":
                layers.append(nn.LayerNorm(out_dim))

        # Add dropout if specified
        if "dropout" in layer and layer["dropout"] > 0:
            layers.append(nn.Dropout(layer["dropout"]))

    return nn.Sequential(*layers)


def correlation_loss(real_data, fake_data):
    '''
    Calculates a penalty on the generator loss if the real data and fake data 
    does not share the same correlations.
    
        The function computes the Pearson correlation matrices of the real and fake datasets
    and penalizes the generator based on the absolute difference between them.
    
    Args:
        real_data (torch.Tensor): A tensor of real data samples, shape `(batch_size, num_features)`.
        fake_data (torch.Tensor): A tensor of fake (generated) data samples, shape `(batch_size, num_features)`.
    
    Returns:
        torch.Tensor: A scalar tensor representing the mean absolute difference 
                      between the correlation matrices of real and fake data.
    '''
        # Calculate means and standard deviations
    real_mean = torch.mean(real_data, dim=0)
    fake_mean = torch.mean(fake_data, dim=0)
    
    real_std = torch.std(real_data, dim=0) + 1e-6  # Add small epsilon to avoid division by zero
    fake_std = torch.std(fake_data, dim=0) + 1e-6
    
    # Center and normalize the data
    real_normalized = (real_data - real_mean) / real_std
    fake_normalized = (fake_data - fake_mean) / fake_std
    
    # Calculate correlation matrices manually
    real_corr = torch.matmul(real_normalized.T, real_normalized) / (real_data.size(0) - 1)
    fake_corr = torch.matmul(fake_normalized.T, fake_normalized) / (fake_data.size(0) - 1)
    
    # Replace NaN values with 0 using masks
    real_corr = torch.where(torch.isnan(real_corr), torch.zeros_like(real_corr), real_corr)
    fake_corr = torch.where(torch.isnan(fake_corr), torch.zeros_like(fake_corr), fake_corr)
    
    # Calculate absolute difference
    corr_diff = torch.abs(real_corr - fake_corr)
    
    return torch.mean(corr_diff)


def adaptive_correlation_loss(real_data, fake_data, running_diff=None, alpha=0.9):
   """
   Correlation loss that adaptively weights problematic correlations more heavily.
   
   Args:
       real_data: Real batch data tensor [batch_size x n_features]
       fake_data: Generated batch data tensor [batch_size x n_features]
       running_diff: Running average of correlation differences matrix [n_features x n_features]
       alpha: Exponential moving average factor (default 0.9)
       
   Returns:
       loss: Weighted correlation loss (scalar)
       running_diff: Updated running difference matrix
   """
   # Calculate means and standard deviations
   real_mean = torch.mean(real_data, dim=0)
   fake_mean = torch.mean(fake_data, dim=0)
   
   real_std = torch.std(real_data, dim=0) + 1e-6  # Add small epsilon to avoid division by zero
   fake_std = torch.std(fake_data, dim=0) + 1e-6
   
   # Center and normalize the data
   real_normalized = (real_data - real_mean) / real_std
   fake_normalized = (fake_data - fake_mean) / fake_std
   
   # Calculate correlation matrices 
   real_corr = torch.matmul(real_normalized.T, real_normalized) / (real_data.size(0) - 1)
   fake_corr = torch.matmul(fake_normalized.T, fake_normalized) / (fake_data.size(0) - 1)
   
   # Replace NaN values with 1 using masks
   real_corr = torch.where(torch.isnan(real_corr), torch.ones_like(real_corr), real_corr)
   fake_corr = torch.where(torch.isnan(fake_corr), torch.ones_like(fake_corr), fake_corr)
   
   # Calculate current correlation differences
   current_diff = torch.abs(real_corr - fake_corr)
   
   # Update running difference matrix with exponential moving average
   if running_diff is None:
       running_diff = current_diff
   else:
       running_diff = alpha * running_diff + (1 - alpha) * current_diff
   
   # Create weights based on historical difficulty
   weights = 1.0 + torch.sigmoid(running_diff)  # Higher weights for historically difficult pairs
   
   # Apply weights to current differences
   weighted_diff = weights * current_diff
   
   # Return mean loss and updated running differences
   return torch.mean(weighted_diff), running_diff