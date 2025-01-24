import os
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths:
DATA_DIR_PATH = 'Data'
FULL_DATA_PATH = os.path.join(DATA_DIR_PATH, 'adult.arff')
TARGET_COLUMN = 'income'
TRAINED_MODELS_DIR_PATH = 'Trained Models'
MODEL_NAME = 'cgan'  # Change between model types ['gan', 'cgan', 'ae']
SAVE_PATH = os.path.join(TRAINED_MODELS_DIR_PATH, MODEL_NAME)
PRETRAIN_PATH = os.path.join(SAVE_PATH, 'best_model.pth')

# Data Config:
NUM_CLASSES = 2     # Number of classes in the dataset, for cGAN
LABEL_RATIO = {0: 0.76, 1: 0.24}    # The ratio of the labels, manually derived from dataset, for generation purposes.
APPLY_AUGMENTATION = True # Apply augmentation on minority classes when stratified split is called.
BATCH_SIZE = 128
VAL_RATIO = 0.0     # Ratio out of the training dataset
TEST_RATIO = 0.2    # Ratio out of the full dataset
SEED = 42   # Change the seed and check influence on the model

# Model Config
DATA_DIM = 108 # The size of the feature vector
NOISE_DIM = 8  # The size of the initial noise vector
LATENT_DIM = 16 # The dimension of the latent (encoding) dimension

## Generator Configuration
GENERATOR_CONFIG = [
    {"input_dim": NOISE_DIM, "output_dim": 32, "activation": nn.LeakyReLU(0.2), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 32, "output_dim": 64, "activation": nn.LeakyReLU(0.2), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 64, "output_dim": 128, "activation": nn.LeakyReLU(0.2), "batch_norm": True, "dropout": 0.3},
    {"input_dim": 128, "output_dim": 64, "activation": nn.LeakyReLU(0.2), "batch_norm": True, "dropout": 0.3},
    {"input_dim": 64, "output_dim": 32, "activation": nn.LeakyReLU(0.2), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 32, "output_dim": LATENT_DIM, "activation": nn.Tanh(), "batch_norm": False, "dropout": 0.0},  # No dropout on the last layer
]

## Discriminator Configuration
DISCRIMINATOR_CONFIG = [
    {"input_dim": LATENT_DIM, "output_dim": 8, "activation": nn.LeakyReLU(0.2), "batch_norm": False, "dropout": 0.5},
    {"input_dim": 8, "output_dim": 1, "activation": nn.Sigmoid(), "batch_norm": False, "dropout": 0.0},  # No dropout on the output layer
]

# Encoder Configuration
ENCODER_CONFIG = [
    {"input_dim": DATA_DIM, "output_dim": 512, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 512, "output_dim": 256, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 256, "output_dim": 128, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 128, "output_dim": 64, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 64, "output_dim": 32, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 32, "output_dim": LATENT_DIM, "activation": None, "batch_norm": False, "dropout": 0.0},  # Latent space, no activation
]

# Decoder Configuration
DECODER_CONFIG = [
    {"input_dim": LATENT_DIM, "output_dim": 32, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 32, "output_dim": 64, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 64, "output_dim": 128, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 128, "output_dim": 256, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 256, "output_dim": 512, "activation": nn.ReLU(), "batch_norm": True, "dropout": 0.2},
    {"input_dim": 512, "output_dim": DATA_DIM, "activation": nn.Tanh(), "batch_norm": False, "dropout": 0.0},  # Reconstructed output
]

# AE Training Config
# GAN\AE Training Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_LEARNING_RATE = 1e-3    
WEIGHT_DECAY = 1e-4
GAN_EARLY_STOP = 25     # Stop after |EARLY_STOP| epochs with no improvement in the total loss
AE_EARLY_STOP = 5     # Stop after |EARLY_STOP| epochs with no improvement in the total loss
WARMUP_EPOCHS = 50  # Define a number of GAN warmup iterations in which the model won't count towards an early stop.
EPOCHS = 500    #   A high number of epochs, hoping for an early stopping 
GENERATOR_UPDATE_FREQ = 1   # Number of G updates per D updates, to balance their losses.
