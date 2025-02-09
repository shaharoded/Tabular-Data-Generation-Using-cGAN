import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Local Code
from config import *
from dataset import *
from nn_utils import build_network, correlation_loss

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


class Discriminator(nn.Module):
    def __init__(self, config):
        """
        Initialize the Discriminator.
        Args:
            config (list[dict]): Configuration for discriminator layers.
        """
        super(Discriminator, self).__init__()
        self.model = build_network(config)

    def forward(self, x):
        return self.model(x)
    
    
class GAN(nn.Module):
    def __init__(self, gen_config, disc_config, pretrained_path=None, lambda_corr=0.1):
        """
        Initialize the GAN with generator and discriminator architectures.
        
        Args:
            gen_config (list[dict]): Configuration for generator layers.
            disc_config (list[dict]): Configuration for discriminator layers.
            pretrained_path (str): Path to pretrained models directory.
            lambda_corr (float): Learning rate for `corr_loss` term (default: 0.1).
        """
        super(GAN, self).__init__()
        print('[Model Status]: Building Model...')
        self.generator = Generator(gen_config)
        self.discriminator = Discriminator(disc_config)
        self.gen_config = gen_config
        self.disc_config = disc_config
        self.noise_dim = gen_config[0].get("input_dim") # Noise dimention is equal to the input
        self.cat_column_indices = []    # For generation function of cat values
        self.lambda_corr = lambda_corr
        
        if pretrained_path:
            try:
                self.__load_weights(pretrained_path)
            except Exception as e:
                print(f'[Model Status]: Could not load existing weights from the directory: {e}')
                print('[Model Status]: Starting a new model...')
                pass
    
    def __load_weights(self, pretrained_path):
        """
        Load weights from a pre-trained model checkpoint.
        """
        if os.path.isfile(pretrained_path):
            print(f"[Model Status]: Loading weights from {pretrained_path}")
            try:
                # Try loading without weights_only first
                checkpoint = torch.load(pretrained_path, weights_only=True)
                
                if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator_state_dict'])
                    self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                    self.cat_column_indices = checkpoint.get('cat_column_indices', [])
                    print("[Model Status]: Successfully loaded model weights")
                else:
                    raise ValueError("Checkpoint does not contain expected keys")
                    
            except Exception as e:
                print(f"[Model Status]: Error loading checkpoint: {str(e)}")
                print("[Model Status]: Starting from scratch...")
                
        else:
            print(f"[Model Status]: Pre-trained model not found at {pretrained_path}, starting from scratch.")
            
    def save_weights(self, epoch, gen_loss, disc_loss, save_path):
        """
        Save the weights of the generator and discriminator.

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
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'cat_column_indices': self.cat_column_indices
        }, save_path)
        
        print(f"Model saved to {save_path}")

    def forward(self, z, real_data=None):
        """
        Forward pass for the GAN.
        
        Args:
            z (torch.Tensor): Noise input to the generator.
            real_data (torch.Tensor, optional): Real data for the discriminator.
        
        Returns:
            torch.Tensor, torch.Tensor: Outputs of generator and discriminator.
        """
        generated_data = self.generator(z)
        if real_data is not None:
            disc_real = self.discriminator(real_data)
            disc_fake = self.discriminator(generated_data)
            return generated_data, disc_real, disc_fake
        return generated_data
    
    def train_model(self, train_loader, epochs, warm_up, lr, weight_decay, device, early_stop=5, gen_update_freq=5, save_path=None):
        """
        Enhanced GAN training with correlation loss, gradient clipping, and learning rate scheduling.
        
        Args:
            (previous args remain the same)
            
        Additional features:
            - Correlation loss between real and fake data distributions
            - Gradient clipping to prevent exploding gradients
            - Learning rate scheduling with warm-up and cosine annealing
            - Separate schedulers for generator and discriminator
        """
        self.cat_column_indices = train_loader.cat_column_indices
        
        # Loss functions
        criterion = nn.BCELoss(reduction='mean')
        
        # Optimizers
        optim_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_disc = optim.Adam(self.discriminator.parameters(), lr=lr/4, betas=(0.5, 0.999), weight_decay=weight_decay)
        
        # Learning rate schedulers
        scheduler_gen = optim.lr_scheduler.OneCycleLR(
            optim_gen,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader) * gen_update_freq,
            pct_start=0.3,  # Warm-up phase
            anneal_strategy='cos'
        )
        
        scheduler_disc = optim.lr_scheduler.OneCycleLR(
            optim_disc,
            max_lr=lr/4,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # For early stopping
        architecture_loss = float("inf")
        best_epoch = 0
        stop_counter = 0
        
        # Track losses
        gen_losses, disc_losses, corr_losses = [], [], []
        self.train()
        
        for epoch in range(epochs):
            gen_loss_epoch, disc_loss_epoch, corr_loss_epoch = 0, 0, 0
            
            for real_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)
                
                # Labels
                real_labels = torch.full((batch_size, 1), 0.9, device=device)
                fake_labels = torch.full((batch_size, 1), 0.1, device=device)
                
                # Train Discriminator
                z = torch.randn(batch_size, self.noise_dim).to(device)
                fake_data = self.generator(z)
                
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data.detach())
                
                loss_real = criterion(real_output, real_labels)
                loss_fake = criterion(fake_output, fake_labels)
                disc_loss = (loss_real + loss_fake)/2 # Aspire to log(2) loss.
                
                optim_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                optim_disc.step()
                scheduler_disc.step()
                
                disc_loss_epoch += disc_loss.item()
                
                # Train Generator
                for _ in range(gen_update_freq):
                    z = torch.randn(batch_size, self.noise_dim).to(device)
                    fake_data = self.generator(z)
                    fake_output = self.discriminator(fake_data)
                    
                    # Combine adversarial and correlation losses
                    gen_adv_loss = criterion(fake_output, real_labels)
                    corr_loss = correlation_loss(real_data, fake_data)
                    gen_loss = gen_adv_loss + self.lambda_corr * corr_loss # Weighted combination
                    
                    optim_gen.zero_grad()
                    gen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    optim_gen.step()
                    scheduler_gen.step()
                    
                    gen_loss_epoch += gen_loss.item()
                    corr_loss_epoch += corr_loss.item()
            
            # Average losses
            gen_loss_epoch /= (len(train_loader) * gen_update_freq)
            disc_loss_epoch /= len(train_loader)
            corr_loss_epoch /= (len(train_loader) * gen_update_freq)
            
            # Track losses
            gen_losses.append(gen_loss_epoch)
            disc_losses.append(disc_loss_epoch)
            corr_losses.append(corr_loss_epoch)
            
            print(f"[Training Status]: Epoch {epoch+1}: G Loss: {gen_loss_epoch:.4f}, "
                f"D Loss: {disc_loss_epoch:.4f}, Correlation Loss: {corr_loss_epoch:.4f}")
            
            # Early stopping based on generator loss
            if gen_loss_epoch < architecture_loss and epoch >= warm_up:
                architecture_loss = gen_loss_epoch
                best_epoch = epoch
                self.save_weights(epoch, gen_loss_epoch, disc_loss_epoch, save_path)
                stop_counter = 0
            elif gen_loss_epoch >= architecture_loss:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    break
        
        # Plot losses
        self._plot_losses(gen_losses, disc_losses, best_epoch)
        
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
        
    def _plot_losses(self, gen_losses, disc_losses, best_epoch=None):
        """
        Plot Generator and Discriminator losses on the same axis and highlight the best epoch.
        
        Args:
            gen_losses (list): List of generator losses for each epoch.
            disc_losses (list): List of discriminator losses for each epoch.
            best_epoch (int, optional): The index of the best epoch to highlight. Defaults to None.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(gen_losses, label='Generator Loss', color='blue')
        plt.plot(disc_losses, label='Discriminator Loss', color='red')
        
        # Add a red vertical line for the best epoch
        if best_epoch is not None:
            plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
        
        plt.title('Generator and Discriminator Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)  # Optional: Adds a grid for better visualization
        plt.tight_layout()
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
            synthetic_data = self.generator(z)
        
        # Post-process categorical columns
        synthetic_data = self._post_processing(synthetic_data)
        
        return synthetic_data
    

class cGAN(GAN):
    '''
    A conditional GAN architecture inheriting from the original architecture for modularity.
    '''
    def __init__(self, gen_config, disc_config, num_classes, pretrained_path=None, lambda_corr=0.1):
        """
        Initialize the conditional GAN with generator and discriminator architectures.
        
        Args:
            gen_config (list[dict]): Configuration for generator layers.
            disc_config (list[dict]): Configuration for discriminator layers.
            num_classes (int): Dimension of the one-hot encoded labels.
            pretrained_path (str): Path to pretrained models directory.
            lambda_corr (float): Learning rate for `corr_loss` term (default: 0.1).
        """
        # Dynamically adjust input dimensions for conditional input
        gen_config[0]["input_dim"] += num_classes  # Adjust generator's input layer
        disc_config[0]["input_dim"] += num_classes  # Adjust discriminator's input layer
        
        super(cGAN, self).__init__(gen_config, disc_config, pretrained_path, lambda_corr)
        self.num_classes = num_classes
        self.noise_dim = gen_config[0].get("input_dim") - num_classes  # Noise dimension without label
        if self.noise_dim <= 0:
            raise ValueError("Input dimension of generator must be greater than num_classes.")

    def forward(self, z, labels, real_data=None):
        """
        Forward pass for the cGAN.
        
        Args:
            z (torch.Tensor): Noise input to the generator.
            labels (torch.Tensor): One-hot encoded labels.
            real_data (torch.Tensor, optional): Real data for the discriminator.
        
        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Outputs of generator and discriminator.
        """
        # Concatenate noise and labels for generator input
        z = torch.cat([z, labels], dim=1)
        generated_data = self.generator(z)

        if real_data is not None:
            # Concatenate labels with real and generated data for discriminator input
            disc_real_input = torch.cat([real_data, labels], dim=1)
            disc_fake_input = torch.cat([generated_data, labels], dim=1)
            
            disc_real = self.discriminator(disc_real_input)
            disc_fake = self.discriminator(disc_fake_input)
            return generated_data, disc_real, disc_fake
        
        return generated_data

    def train_model(self, train_loader, epochs, warm_up, lr, weight_decay, device, early_stop=5, gen_update_freq=5, save_path=None):
        """
        Enhanced conditional GAN training with improved training techniques.
        """
        self.cat_column_indices = train_loader.cat_column_indices
        
        # Loss functions with mean reduction
        criterion = nn.BCELoss(reduction='mean')
        
        # Optimizers with different learning rates
        optim_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_disc = optim.Adam(self.discriminator.parameters(), lr=lr/4, betas=(0.5, 0.999), weight_decay=weight_decay)
        
        # Learning rate schedulers
        scheduler_gen = optim.lr_scheduler.OneCycleLR(
            optim_gen,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader) * gen_update_freq,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        scheduler_disc = optim.lr_scheduler.OneCycleLR(
            optim_disc,
            max_lr=lr/4,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # For early stopping
        architecture_loss = float("inf")
        best_epoch = 0
        stop_counter = 0
        
        # Track losses
        gen_losses, disc_losses, corr_losses = [], [], []
        self.train()
        
        for epoch in range(epochs):
            gen_loss_epoch, disc_loss_epoch, corr_loss_epoch = 0, 0, 0
            
            for real_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data, labels = real_data.to(device), labels.to(device)
                batch_size = real_data.size(0)
                
                # One-hot encode labels if not already
                if labels.dim() == 1:
                    labels = F.one_hot(labels, num_classes=self.num_classes).float()
                
                # Labels with smoothing
                real_labels = torch.full((batch_size, 1), 0.9, device=device)
                fake_labels = torch.full((batch_size, 1), 0.1, device=device)
                
                # Train Discriminator
                z = torch.cat([torch.randn(batch_size, self.noise_dim).to(device), labels], dim=1)
                fake_data = self.generator(z)
                
                disc_real_input = torch.cat([real_data, labels], dim=1)
                disc_fake_input = torch.cat([fake_data.detach(), labels], dim=1)
                
                real_output = self.discriminator(disc_real_input)
                fake_output = self.discriminator(disc_fake_input)
                
                loss_real = criterion(real_output, real_labels)
                loss_fake = criterion(fake_output, fake_labels)
                disc_loss = (loss_real + loss_fake)/2  # Average for log(2) target
                
                optim_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                optim_disc.step()
                scheduler_disc.step()
                
                disc_loss_epoch += disc_loss.item()
                
                # Train Generator
                for _ in range(gen_update_freq):
                    z = torch.cat([torch.randn(batch_size, self.noise_dim).to(device), labels], dim=1)
                    fake_data = self.generator(z)
                    disc_fake_input = torch.cat([fake_data, labels], dim=1)
                    fake_output = self.discriminator(disc_fake_input)
                    
                    # Combine adversarial and correlation losses
                    gen_adv_loss = criterion(fake_output, real_labels)
                    corr_loss = correlation_loss(real_data, fake_data)
                    gen_loss = gen_adv_loss + self.lambda_corr * corr_loss # Weighted combination
                                        
                    optim_gen.zero_grad()
                    gen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    optim_gen.step()
                    scheduler_gen.step()
                    
                    gen_loss_epoch += gen_loss.item()
                    corr_loss_epoch += corr_loss.item()
            
            # Average losses
            gen_loss_epoch /= (len(train_loader) * gen_update_freq)
            disc_loss_epoch /= len(train_loader)
            corr_loss_epoch /= (len(train_loader) * gen_update_freq)
            
            # Track losses
            gen_losses.append(gen_loss_epoch)
            disc_losses.append(disc_loss_epoch)
            corr_losses.append(corr_loss_epoch)
            
            print(f"[Training Status]: Epoch {epoch+1}: G Loss: {gen_loss_epoch:.4f}, "
                f"D Loss: {disc_loss_epoch:.4f}, Correlation Loss: {corr_loss_epoch:.4f}")
            
            # Early stopping based on generator loss
            if gen_loss_epoch < architecture_loss and epoch >= warm_up:
                architecture_loss = gen_loss_epoch
                best_epoch = epoch
                self.save_weights(epoch, gen_loss_epoch, disc_loss_epoch, save_path)
                stop_counter = 0
            elif gen_loss_epoch >= architecture_loss:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    break
        
        self._plot_losses(gen_losses, disc_losses, best_epoch)
    

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
    
    if MODEL_NAME == 'gan':
        print("[Main]: Training a GAN model...")
        model = GAN(gen_config=GENERATOR_CONFIG,
                    disc_config=DISCRIMINATOR_CONFIG,
                    pretrained_path=PRETRAIN_PATH,
                    lambda_corr=LAMBDA_CORR)
        
        model.train_model(train_loader=train_loader, 
            epochs=EPOCHS,
            warm_up=WARMUP_EPOCHS, 
            lr=BASE_LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY, 
            device=DEVICE, 
            early_stop=GAN_EARLY_STOP,
            gen_update_freq=GENERATOR_UPDATE_FREQ, 
            save_path=PRETRAIN_PATH)
        
        synthetic = model.generate(
            num_samples=1, 
            noise_dim=NOISE_DIM, 
            device=DEVICE
        )
    
    elif MODEL_NAME == 'cgan':
        print("[Main]: Training a cGAN model...")
        model = cGAN(gen_config=GENERATOR_CONFIG,
                    disc_config=DISCRIMINATOR_CONFIG,
                    num_classes=NUM_CLASSES,
                    pretrained_path=PRETRAIN_PATH,
                    lambda_corr=LAMBDA_CORR)
        
        model.train_model(train_loader=train_loader, 
            epochs=EPOCHS,
            warm_up=WARMUP_EPOCHS, 
            lr=BASE_LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY, 
            device=DEVICE, 
            early_stop=GAN_EARLY_STOP,
            gen_update_freq=GENERATOR_UPDATE_FREQ, 
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
    
    
    
    
    