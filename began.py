import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Local Code
from config import *
from dataset import *
from nn_utils import build_network, correlation_loss
from autoencoder import Autoencoder


class Generator(nn.Module):
    def __init__(self, config):
        """
        Initialize the Generator.
        Args:
            config (list[dict]): Configuration for generator layers.
        """
        super(Generator, self).__init__()
        self.model = build_network(config)
        self.noise_dim = config[0].get("input_dim")  # Input noise dimension

    def forward(self, z):
        return self.model(z)
    

class BEGAN(nn.Module):
    def __init__(self, gen_config, pretrained_autoencoder, pretrained_path=None, gamma=0.75, lambda_corr=0.1, lambda_k=0.001):
        """
        Initialize the BEGAN with generator and an autoencoder critic.
        A trained AE is inputed for manipulation on original data, and it's decoder is used as
        a critic within the GAN.
        Args:
            gen_config (list[dict]): Configuration for generator layers.
            pretrained_autoencoder (Autoencoder): A trained AE object with correct dimensions.
            pretrained_path (str): Path to pretrained models directory.
            gamma (float): BEGAN balance factor (default: 0.75).
            lambda_k (float): Learning rate for `k_t` balance term (default: 0.001).
            lambda_corr (float): Learning rate for `corr_loss` term (default: 0.1).
        """
        super(BEGAN, self).__init__()
        print(f'[Model Status]: Building {type(self)} Model...')
        self.generator = Generator(gen_config)
        self.decoder = pretrained_autoencoder.decoder
        self.encoder = pretrained_autoencoder.encoder
        for param in self.encoder.parameters():  # Freeze encoder weights
            param.requires_grad = False
        self.noise_dim = gen_config[0].get("input_dim")
        self.cat_column_indices = []    # For generation function of cat values
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.lambda_corr = lambda_corr
        self.k_t = 0.0  # Balance variable

        if pretrained_path:
            try:
                self.__load_weights(pretrained_path)
            except Exception as e:
                print(f'[Model Status]: Could not load existing weights from the directory: {e}')
                print('[Model Status]: Starting a new model...')
                
    
    def __load_weights(self, pretrained_path):
        """
        Load weights from a pre-trained model checkpoint.
        
        Args:
            pretrained_path (str): Path to the pre-trained model checkpoint.
        """
        if os.path.isfile(pretrained_path):
            print(f"[Model Status]: Loading weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, weights_only=True)
            
            # Check dimensions
            current_input_dim = self.generator.state_dict()['model.0.weight'].shape[1]
            checkpoint_input_dim = checkpoint['generator_state_dict']['model.0.weight'].shape[1]
            
            # For cBEGAN, we need to account for the conditional input
            if hasattr(self, 'num_classes'):
                expected_input_dim = self.noise_dim + self.num_classes
                if current_input_dim != expected_input_dim:
                    raise ValueError(f"Input dimension mismatch for cBEGAN: Expected {expected_input_dim} (noise_dim={self.noise_dim} + num_classes={self.num_classes}), but current model has {current_input_dim}")
                if checkpoint_input_dim != expected_input_dim:
                    raise ValueError(f"Checkpoint dimension mismatch for cBEGAN: Expected {expected_input_dim}, but checkpoint has {checkpoint_input_dim}")
            # For regular BEGAN, just check if dimensions match
            else:
                if current_input_dim != checkpoint_input_dim:
                    raise ValueError(f"Input dimension mismatch: Current model has {current_input_dim}, but checkpoint has {checkpoint_input_dim}")
            
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.cat_column_indices = checkpoint.get('cat_column_indices',[])
            
        else:
            print(f"[Model Status]: Pre-trained model not found at {pretrained_path}, starting from scratch.")

            
    def save_weights(self, epoch, gen_loss, disc_loss, save_path):
        """
        Save the weights of the generator and ae critic.

        Args:
            epoch (int): Current epoch number.
            gen_loss (float): Generator loss at the end of this epoch.
            disc_loss (float): Discriminator loss at the end of this epoch.
            save_path (str): Full path where to save the model weights, which is identical to pretrain_path.
        """
        save_dir = os.path.dirname(save_path)  # Get directory from the provided path
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'k_t': self.k_t,
            'gen_loss': gen_loss,
            'critic_loss': disc_loss,
            'cat_column_indices': self.cat_column_indices
        }, save_path)
        
        print(f"Model saved to {save_path}")


    def forward(self, z):
        """
        Generate samples from noise.
        This method ensures the generated output matches the original tabular data format.
        
        Args:
            z (torch.Tensor): Random noise vector of shape (batch_size, noise_dim).
        
        Returns:
            torch.Tensor: Generated tabular samples.
        """
        latent_space = self.generator(z)  # Generate **latent representations**
        generated_data = self.decoder(latent_space)  # Decode to tabular format
        return generated_data
       
    
    def train_model(self, train_loader, epochs, warm_up, lr, weight_decay, device, early_stop=5, save_path=None):
        """
        Train the BEGAN model.
        Best model weights are saved, and the improvement is decided by Generator loss.
        Adds correlation difference between real and fake data as loss term to the generator.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train.
            warm_up (int): Number of warm-up epochs which will not count toward early stopping.
            lr (float): Learning rate.
            weight_decay (float): W2 penalty, added only to critic to slow it down.
            device (torch.device): Device to train on.
            early_stop (int, optional): Early stopping patience. Defaults to 5.
            save_path (str): Full path where to save the model weights, which is identical to pretrained_path.
        """
        # Initialization update
        self.cat_column_indices = train_loader.cat_column_indices
        
        # Optimizers
        optim_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_critic = optim.Adam(self.decoder.parameters(), lr=lr/2, betas=(0.5, 0.999), weight_decay=weight_decay)
        
        # Schedulers - exponential decay with small gamma
        scheduler_gen = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.995)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(optim_critic, gamma=0.995)
        
        # Early stopping variables
        best_loss = float("inf")
        stop_counter = 0
        best_epoch = 0

        # Loss tracking for plotting
        gen_losses, crit_losses, k_values = [], [], []       

        self.train()
        for epoch in range(epochs):
            gen_loss_epoch, crit_loss_epoch = 0, 0

            for real_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)

                # Encode real data into latent space
                with torch.no_grad():
                    real_latent = self.encoder(real_data)
                
                # === Critic Step ===
                optim_critic.zero_grad()
                    
                # Generate fake data for critic
                z = torch.randn(batch_size, self.noise_dim).to(device)
                fake_latent_critic = self.generator(z).detach()
                fake_data_critic = self.decoder(fake_latent_critic)

                # Train critic
                real_reconstructed = self.decoder(real_latent)
                fake_reconstructed = self.decoder(fake_latent_critic)

                real_loss = torch.mean(torch.abs(real_data - real_reconstructed))
                fake_loss = torch.mean(torch.abs(fake_data_critic - fake_reconstructed))

                critic_loss = real_loss - self.k_t * fake_loss
                critic_loss.backward(retain_graph=True)  # Add retain_graph=True here
                torch_utils.clip_grad_norm_(self.generator.parameters(), 1.0)  # Apply gradient clipping
                optim_critic.step()
                
                # === Generator Step ===
                optim_gen.zero_grad()

                # Generate fresh fake data for generator
                z = torch.randn(batch_size, self.noise_dim).to(device)
                fake_latent_gen = self.generator(z)
                fake_data_gen = self.decoder(fake_latent_gen)
                fake_reconstructed_gen = self.decoder(fake_latent_gen)

                # Train generator
                corr_loss = correlation_loss(real_data, fake_data_gen) # Calculate correlation loss
                recon_loss = torch.mean(torch.abs(fake_data_gen - fake_reconstructed_gen))
                generator_loss = recon_loss + (self.lambda_corr * (1 - self.k_t)) * corr_loss   # Increase corr loss when k_t is small
                generator_loss.backward()  # No need for retain_graph here as it's the last backward pass
                torch_utils.clip_grad_norm_(self.generator.parameters(), 1.0)  # Apply gradient clipping
                optim_gen.step()

                # Update k_t
                with torch.no_grad():  # Add no_grad here for k_t update
                    self.k_t = max(0, min(1, self.k_t + self.lambda_k * (self.gamma * real_loss.item() - fake_loss.item())))

                gen_loss_epoch += generator_loss.item()
                crit_loss_epoch += critic_loss.item()

            # Step the schedulers at the end of each epoch
            scheduler_gen.step()
            scheduler_critic.step()
            
            # Log the losses
            gen_loss_epoch /= len(train_loader)
            crit_loss_epoch /= len(train_loader)
            gen_losses.append(gen_loss_epoch)
            crit_losses.append(crit_loss_epoch)
            k_values.append(self.k_t)

            print(f"[Training Status]: Epoch {epoch+1}: G Loss: {gen_loss_epoch:.4f}, "
                  f"C Loss: {crit_loss_epoch:.4f}, Corr Loss: {corr_loss.item():.4f}, k_t: {self.k_t:.4f}")

            if gen_loss_epoch < best_loss and epoch >= warm_up:
                best_loss = gen_loss_epoch
                best_epoch = epoch
                stop_counter = 0
                self.save_weights(epoch, gen_loss_epoch, crit_loss_epoch, save_path)
            elif epoch >= warm_up:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    break

        # Plot losses at the end of training
        self._plot_losses(gen_losses, crit_losses, k_values, best_epoch)
        
    def _post_processing(self, generated_data):
        '''
        Performs post process on the data, ensuring it fits an expected format 
        based on the train dataset, for example, ensuring cat columns have only 1 True value.
        A helper function to the generator to create better logits.
        
        Args:
            generated_data: The output from self.generator(noise)
        '''
        # Post-process categorical columns
        for group in self.cat_column_indices:
            # Iterate over each group of categorical columns
            for row in range(generated_data.size(0)):  # Iterate over rows
                row_data = generated_data[row, group]  # Extract the group of columns for the current row

                # Find the index of the max value in the row for this group
                max_val_idx = row_data.argmax().item()  # Max value in the group for this row (get the scalar index)

                # Map the relative index (within the group) to the global index in the full row
                global_idx = group[max_val_idx]  # This gives the actual column index in the full row

                # Set all values in the group to -1, then set the max value to 1
                generated_data[row, group] = -1  # Set all values in this group to -1
                generated_data[row, global_idx] = 1  # Mark the highest value as 1 in the global index
        return generated_data
        
    def _plot_losses(self, gen_losses, critic_losses, k_values, best_epoch=None):
        """
        Plot Generator and Critic losses on the primary y-axis,
        and k_t values on a secondary y-axis to ensure proper scaling.
        
        Args:
            gen_losses (list): List of generator losses for each epoch.
            critic_losses (list): List of critic losses for each epoch.
            k_values (list, optional): Monitored K values from the training process.
            best_epoch (int, optional): The index of the best epoch to highlight. Defaults to None.
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Primary y-axis (Generator & Critic losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="black")
        ax1.plot(gen_losses, label="Generator Loss", color="blue")
        ax1.plot(critic_losses, label="Critic Loss", color="red")
        ax1.tick_params(axis="y", labelcolor="black")
        
        # Secondary y-axis (k_t values)
        ax2 = ax1.twinx()
        ax2.set_ylabel("k_t Value", color="green")
        ax2.plot(k_values, label="k_t", color="green", linestyle="--")
        ax2.tick_params(axis="y", labelcolor="green")

        # Highlight the best epoch
        if best_epoch is not None:
            ax1.axvline(best_epoch, color="purple", linestyle="--", label=f"Best Epoch ({best_epoch})")

        # Legends
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.title("BEGAN Training Metrics")
        plt.grid(True)
        plt.show()
                    
    def generate(self, num_samples, noise_dim, device, labels=None, num_classes=None):
        """
        Generate synthetic samples using the trained generator.

        Args:
            num_samples (int): Number of samples to generate.
            noise_dim (int): Dimensionality of the noise vector.
            device (torch.device): Device for generation.
            labels (torch.Tensor, optional): Conditional labels (for cGAN). Defaults to None.
            num_classes (int, optional): Number of classes for one-hot encoding (for cGAN). Required if labels are provided.

        Returns:
            torch.Tensor: Generated samples.
        """
        self.generator.eval()
        self.decoder.eval()
        
        # Raise an exception if labels are provided but the model is not cGAN
        if labels is not None:
            if not hasattr(self, 'num_classes'):
                raise ValueError("Labels were provided, but the model is not a cGAN. Remove labels or use a cGAN model.")
            assert num_classes is not None, "num_classes must be specified if labels are provided."
        
        # Generate noise
        z = torch.randn(num_samples, noise_dim).to(device)
        
        # If labels are provided, concatenate them (for cGAN)
        if labels is not None:
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
            z = torch.cat([z, labels_one_hot], dim=1)
        
        # Generate synthetic data
        with torch.no_grad():
            latent_data = self.generator(z)
            synthetic_data = self.decoder(latent_data)
        
        # Post-process categorical columns
        synthetic_data = self._post_processing(synthetic_data)
        
        return synthetic_data
    

class cBEGAN(BEGAN):
    '''
    A conditional BEGAN architecture inheriting from the original BEGAN
    for modularity.
    '''
    def __init__(self, gen_config, pretrained_autoencoder, num_classes, pretrained_path=None, gamma=0.75, lambda_corr=0.1, lambda_k=0.001):
        """
        Initialize the conditional BEGAN with generator and autoencoder critic.
        
        Args:
            gen_config (list[dict]): Configuration for generator layers.
            pretrained_autoencoder (Autoencoder): A trained AE object with correct dimensions.
            num_classes (int): Dimension of the one-hot encoded labels.
            pretrained_path (str): Path to pretrained models directory.
            gamma (float): BEGAN balance factor (default: 0.75).
            lambda_corr (float): Learning rate for `corr_loss` term (default: 0.1).
            lambda_k (float): Learning rate for k_t balance term (default: 0.001).
        """
        # Dynamically adjust input dimensions for conditional input
        gen_config[0]["input_dim"] += num_classes  # Adjust generator's input layer
        
        super(cBEGAN, self).__init__(gen_config, pretrained_autoencoder, pretrained_path, gamma, lambda_corr, lambda_k)
        self.num_classes = num_classes
        self.noise_dim = gen_config[0].get("input_dim") - num_classes  # Noise dimension without label
        if self.noise_dim <= 0:
            raise ValueError("Input dimension of generator must be greater than num_classes.")

    def forward(self, z, labels):
        """
        Generate samples from noise and labels.
        
        Args:
            z (torch.Tensor): Random noise vector of shape (batch_size, noise_dim).
            labels (torch.Tensor): One-hot encoded labels.
        
        Returns:
            torch.Tensor: Generated tabular samples.
        """
        # Concatenate noise and labels for generator input
        z = torch.cat([z, labels], dim=1)
        latent_space = self.generator(z)
        generated_data = self.decoder(latent_space)
        return generated_data

    def train_model(self, train_loader, epochs, warm_up, lr, weight_decay, device, early_stop=5, save_path=None):
        """
        Train the conditional BEGAN model, with the addition of the labels.
        """
        self.cat_column_indices = train_loader.cat_column_indices
        
        optim_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_critic = optim.Adam(self.decoder.parameters(), lr=lr/2, betas=(0.5, 0.999), weight_decay=weight_decay)

        # Schedulers - exponential decay with small gamma
        scheduler_gen = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.995)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(optim_critic, gamma=0.995)
        
        best_loss = float("inf")
        stop_counter = 0
        best_epoch = 0
        gen_losses, crit_losses, k_values = [], [], []

        self.train()
        for epoch in range(epochs):
            gen_loss_epoch, crit_loss_epoch = 0, 0

            for real_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data, labels = real_data.to(device), labels.to(device)
                batch_size = real_data.size(0)

                # One-hot encode labels if not already
                if labels.dim() == 1:
                    labels = F.one_hot(labels, num_classes=self.num_classes).float()

                # Encode real data into latent space
                with torch.no_grad():
                    real_latent = self.encoder(real_data)
                
                # === Critic Step ===
                optim_critic.zero_grad()
                    
                # Generate fake data for critic (with labels)
                z = torch.randn(batch_size, self.noise_dim).to(device)
                z = torch.cat([z, labels], dim=1)
                fake_latent_critic = self.generator(z).detach()
                fake_data_critic = self.decoder(fake_latent_critic)

                real_reconstructed = self.decoder(real_latent)
                fake_reconstructed = self.decoder(fake_latent_critic)

                real_loss = torch.mean(torch.abs(real_data - real_reconstructed))
                fake_loss = torch.mean(torch.abs(fake_data_critic - fake_reconstructed))

                critic_loss = real_loss - self.k_t * fake_loss
                critic_loss.backward(retain_graph=True)
                torch_utils.clip_grad_norm_(self.generator.parameters(), 1.0)  # Apply gradient clipping
                optim_critic.step()
                
                # === Generator Step ===
                optim_gen.zero_grad()

                # Generate fresh fake data for generator (with labels)
                z = torch.randn(batch_size, self.noise_dim).to(device)
                z = torch.cat([z, labels], dim=1)
                fake_latent_gen = self.generator(z)
                fake_data_gen = self.decoder(fake_latent_gen)
                fake_reconstructed_gen = self.decoder(fake_latent_gen)
                
                # Train generator
                corr_loss = correlation_loss(real_data, fake_data_gen) # Calculate correlation loss
                recon_loss = torch.mean(torch.abs(fake_data_gen - fake_reconstructed_gen))
                generator_loss = recon_loss + (self.lambda_corr * (1 - self.k_t)) * corr_loss   # Increase corr loss when k_t is small
                generator_loss.backward()  # No need for retain_graph here as it's the last backward pass
                torch_utils.clip_grad_norm_(self.generator.parameters(), 1.0)  # Apply gradient clipping
                optim_gen.step()

                # Update k_t
                with torch.no_grad():
                    self.k_t = max(0, min(1, self.k_t + self.lambda_k * (self.gamma * real_loss.item() - fake_loss.item())))

                gen_loss_epoch += generator_loss.item()
                crit_loss_epoch += critic_loss.item()
            
            # Step the schedulers at the end of each epoch
            scheduler_gen.step()
            scheduler_critic.step()
            
            # Log the losses
            gen_loss_epoch /= len(train_loader)
            crit_loss_epoch /= len(train_loader)
            gen_losses.append(gen_loss_epoch)
            crit_losses.append(crit_loss_epoch)
            k_values.append(self.k_t)

            print(f"[Training Status]: Epoch {epoch+1}: G Loss: {gen_loss_epoch:.4f}, "
                  f"C Loss: {crit_loss_epoch:.4f}, Corr Loss: {corr_loss.item():.4f}, k_t: {self.k_t:.4f}")

            if gen_loss_epoch < best_loss and epoch >= warm_up:
                best_loss = gen_loss_epoch
                best_epoch = epoch
                stop_counter = 0
                self.save_weights(epoch, gen_loss_epoch, crit_loss_epoch, save_path)
            elif epoch >= warm_up:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    break

        self._plot_losses(gen_losses, crit_losses, k_values, best_epoch)
    
    
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
    train_set, _, test_set = dataset.stratified_split(
        val_size=VAL_RATIO, test_size=TEST_RATIO, random_state=SEED
    )
    print("[Main]: Creating Dataloader...")
    train_loader = get_dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    print("[Main]: Initializing Autoencoder...")
    autoencoder = Autoencoder(
        encoder_config=ENCODER_CONFIG,
        decoder_config=DECODER_CONFIG,
        pretrained_path=os.path.join(TRAINED_MODELS_DIR_PATH, 'ae', 'best_model.pth')
    ).to(DEVICE)
        
    if MODEL_NAME == 'gan':
        print("[Main]: Training a BEGAN model...")
        model = BEGAN(gen_config=GENERATOR_CONFIG,
                    pretrained_autoencoder=autoencoder,
                    pretrained_path=PRETRAIN_PATH,
                    gamma=GAMMA,
                    lambda_k=LAMBDA_K)
        
        model.train_model(
            train_loader=train_loader, 
            epochs=EPOCHS,
            warm_up=WARMUP_EPOCHS, 
            lr=BASE_LEARNING_RATE, 
            weight_decay = WEIGHT_DECAY,
            device=DEVICE, 
            early_stop=GAN_EARLY_STOP,
            save_path=PRETRAIN_PATH
        )
        
        synthetic = model.generate(
            num_samples=1, 
            noise_dim=NOISE_DIM, 
            device=DEVICE
        )
        print('Synthetic Data:')
        print(synthetic)
    
    elif MODEL_NAME == 'cgan':
        print("[Main]: Training a WcGAN model...")
        model = cBEGAN(gen_config=GENERATOR_CONFIG,
                    pretrained_autoencoder=autoencoder,
                    gamma=GAMMA,
                    lambda_k=LAMBDA_K,
                    num_classes=NUM_CLASSES,
                    pretrained_path=PRETRAIN_PATH)
        
        model.train_model(train_loader=train_loader, 
            epochs=EPOCHS,
            warm_up=WARMUP_EPOCHS, 
            lr=BASE_LEARNING_RATE, 
            weight_decay = WEIGHT_DECAY,
            device=DEVICE, 
            early_stop=GAN_EARLY_STOP,
            save_path=PRETRAIN_PATH)
        
        synthetic = model.generate(
                num_samples=1, 
                noise_dim=NOISE_DIM, 
                device=DEVICE, 
                labels=torch.tensor([1]).to(DEVICE), 
                num_classes=NUM_CLASSES
            )
    else:
        raise ValueError('A model name has to be picked (gan / cgan)')

    print('Synthetic Data:')
    print(synthetic)