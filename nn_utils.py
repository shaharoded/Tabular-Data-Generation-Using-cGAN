
import torch.nn as nn


def build_network(config):
    """
    Build a network based on the provided configuration.

    Args:
        config (list[dict]): Layer-wise network configuration.

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

        # Add batch normalization if specified
        if "batch_norm" in layer and layer["batch_norm"]:
            layers.append(nn.BatchNorm1d(out_dim))

        # Add dropout if specified
        if "dropout" in layer and layer["dropout"] > 0:
            layers.append(nn.Dropout(layer["dropout"]))

    return nn.Sequential(*layers)