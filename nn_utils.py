
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


def composite_loss(reconstructed, original, num_features=6, beta=0.5):
    """
    Composite loss for AE with separate weighting for numerical and categorical features,
    using SmoothL1Loss for both but potentially different beta values.
    
    Args:
        reconstructed: The reconstructed data
        original: The original data
        num_features: Number of numerical features
        beta: Weight balance between numerical and categorical loss
    Returns:
        total_loss - The total loss term, for propagation.
        num_loss - The numeric loss term, for monitoring.
        cat_loss - The categorical loss term, for monitoring.
    """
    # SmoothL1Loss for numerical features (possibly with different parameters)
    num_criterion = nn.SmoothL1Loss(beta=0.3)  # Smaller beta for numerical features
    num_loss = num_criterion(
        reconstructed[:, :num_features], 
        original[:, :num_features]
    )
    
    # Categorical features - add thresholding loss
    cat_criterion = nn.SmoothL1Loss(beta=1.0)
    cat_output = reconstructed[:, num_features:]
    cat_loss = cat_criterion(cat_output, original[:, num_features:]) + \
               0.1 * torch.mean(torch.abs(torch.abs(cat_output) - 1.0))  # Push values towards ±1
    
    total_loss = beta * num_loss + (1 - beta) * cat_loss
    
    return total_loss, num_loss, cat_loss