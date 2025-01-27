
import torch.nn as nn


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