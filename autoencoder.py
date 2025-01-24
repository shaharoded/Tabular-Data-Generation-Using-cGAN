import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Local Code
from config import *
from dataset import *
from nn_utils import build_network


class Autoencoder(nn.Module):
    def __init__(self, encoder_config, decoder_config, pretrained_path=None):
        """
        Autoencoder with configurable encoder and decoder layers.

        Args:
            encoder_config (list[dict]): Configuration for the encoder layers.
            decoder_config (list[dict]): Configuration for the decoder layers.
            pretrained_path (str, optional): Path to a pre-trained model to load.
        """
        super(Autoencoder, self).__init__()
        print(f'[Model Status]: Building {type(self)} Model...')
        self.encoder = build_network(encoder_config)
        self.decoder = build_network(decoder_config)
        self.apply(self.__init_weights)

        # Automatically load weights if a pretrained file exists
        if pretrained_path:
            try:
                self.__load_weights(pretrained_path)
            except Exception as e:
                print(f"[Model Status]: Could not load existing weights: {e}")
                print("[Model Status]: Starting a new model...")
                
    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor, torch.Tensor: Latent representation and reconstructed output.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def train_model(self, train_loader, val_loader, epochs, lr, 
                    device, early_stop=5, save_path=None):
        """
        Train the autoencoder with early stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Number of epochs to train.
            lr (float): Learning rate.
            device (torch.device): Device to train on.
            early_stop (int, optional): Early stopping patience. Defaults to 5.
            save_path (str, optional): Path to save the best model. Defaults to None.

        Returns:
            list: Training and validation losses per epoch.
        """
        # Loss function and optimizer
        criterion = nn.SmoothL1Loss()   # Recommended for such tasks
        optimizer = torch.optim.Adam(self.parameters(), lr=lr*10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float("inf")
        stop_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):          
            # Training phase
            self.train()
            train_loss = 0.0
            with tqdm(train_loader, desc="Training", leave=False) as train_bar:
                for batch in train_bar:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]  # Extract features if dataset returns (features, labels)
                    batch = batch.to(device)
                    _, reconstructed = self(batch)
                    loss = criterion(reconstructed, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_bar.set_postfix(loss=loss.item())

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(val_loader, desc="Validation", leave=False) as val_bar:
                    for batch in val_bar:
                        if isinstance(batch, (list, tuple)):
                            batch = batch[0]
                        batch = batch.to(device)
                        _, reconstructed = self(batch)
                        loss = criterion(reconstructed, batch)
                        val_loss += loss.item()
                        val_bar.set_postfix(loss=loss.item())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Adjust learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_counter = 0
                if save_path:
                    self.save_weights(save_path)
            else:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    print(f"[Training Status]: Autoencoder saved to {save_path}")
                    break
        self.__plot_losses(train_losses, val_losses)
    
    
    def __plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __load_weights(self, pretrained_path):
        """
        Load weights from a pre-trained model checkpoint.

        Args:
            pretrained_path (str): Path to the pre-trained model checkpoint.
        """
        if os.path.isfile(pretrained_path):
            print(f"[Model Status]: Loading weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, weights_only=True)
            self.load_state_dict(checkpoint)
        else:
            print(f"[Model Status]: Pre-trained model not found at {pretrained_path}, starting from scratch.")

    def save_weights(self, save_path):
        """
        Save the weights of the autoencoder.

        Args:
            save_path (str): Full path where to save the model weights.
        """
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        torch.save(self.state_dict(), save_path)
        

if __name__ == "__main__":
    # Usage example    
    print("[Main]: Initializing dataset...")
    dataset = TabularDataset(
        file_path=FULL_DATA_PATH,
        target_column=TARGET_COLUMN,
        augment=APPLY_AUGMENTATION,
        info=False  # Print dataset info
    )
    
    # Perform stratified split
    print("[Main]: Performing stratified train-val-test split...")
    train_set, val_set, test_set = dataset.stratified_split(
        val_size=VAL_RATIO, test_size=TEST_RATIO, random_state=SEED
    )
    print("[Main]: Creating Dataloader...")
    train_loader = get_dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
        # Initialize Autoencoder
    print("[Main]: Initializing Autoencoder...")
    autoencoder = Autoencoder(
        encoder_config=ENCODER_CONFIG,
        decoder_config=DECODER_CONFIG,
        pretrained_path=PRETRAIN_PATH  # Load pretrained weights if they exist
    ).to(DEVICE)

    # Train the Autoencoder
    print("[Main]: Training Autoencoder...")
    autoencoder.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=BASE_LEARNING_RATE,
        device=DEVICE,
        early_stop=AE_EARLY_STOP,
        save_path=PRETRAIN_PATH
    )

    print("[Main]: Evaluating Autoencoder...")
    autoencoder.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", leave=False) as test_bar:
            for batch in test_bar:
                # Extract features if the batch is a list or tuple
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Extract the features

                batch = batch.to(DEVICE)
                _, reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                test_loss += loss.item()
                test_bar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    print(f"[Main]: Test Loss: {test_loss:.4f}")

    # Example usage of the trained autoencoder
    print("[Main]: Example latent representation...")
    with torch.no_grad():
        batch = next(iter(test_loader))[0].to(DEVICE)  # Extract features
        _, reconstructed = autoencoder(batch)
        print("Original:", batch[0])
        print("Reconstructed:", reconstructed[0])
    