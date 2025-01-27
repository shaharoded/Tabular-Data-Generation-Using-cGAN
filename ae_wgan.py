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
from nn_utils import build_network
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


class Critic(nn.Module):
    def __init__(self, config):
        """
        Initialize the Critic for WGAN.
        Args:
            config (list[dict]): Configuration for critic layers.
        """
        super(Critic, self).__init__()
        self.model = build_network(config)

    def forward(self, x):
        return self.model(x)  # Real-valued output
    

class WGAN(nn.Module):
    def __init__(self, gen_config, critic_config, autoencoder, pretrained_path=None, lambda_gp=10.0):
        """
        Initialize the WGAN with generator and critic.
        Args:
            gen_config (list[dict]): Configuration for generator layers.
            critic_config (list[dict]): Configuration for critic layers.
            autoencoder (Autoencoder): A trained AE object with correct dimensions.
            pretrained_path (str): Path to pretrained models directory.
            lambda_gp (float): Gradient penalty weight.
        """
        super(WGAN, self).__init__()
        print(f'[Model Status]: Building {type(self)} Model...')
        self.generator = Generator(gen_config)
        self.critic = Critic(critic_config)
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():  # Freeze encoder weights
            param.requires_grad = False
        self.noise_dim = gen_config[0].get("input_dim")
        self.cat_column_indices = []    # For generation function of cat values
        self.lambda_gp = lambda_gp

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
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.cat_column_indices = checkpoint.get('cat_column_indices',[])
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
            'critic_state_dict': self.critic.state_dict(),
            'gen_loss': gen_loss,
            'critic_loss': disc_loss,
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
            disc_real = self.critic(real_data)
            disc_fake = self.critic(generated_data)
            return generated_data, disc_real, disc_fake
        return generated_data
       
    def _compute_gradient_penalty(self, real_samples, fake_samples):
        """
        Compute gradient penalty for WGAN-GP.
        Args:
            real_samples (torch.Tensor): Real samples.
            fake_samples (torch.Tensor): Fake samples generated by the generator.
        Returns:
            torch.Tensor: Gradient penalty value.
        """
        batch_size = real_samples.size(0)

        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, device=real_samples.device)
        alpha = alpha.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)  # Ensure gradients can be computed

        # Get critic scores for interpolated samples
        interpolates_validity = self.critic(interpolates)

        # Compute gradients w.r.t. interpolates
        grad_outputs = torch.ones_like(interpolates_validity, device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=interpolates_validity,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Reshape gradients and compute their norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Compute gradient penalty
        gradient_penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty
    
    def train_model(self, train_loader, epochs, warm_up, lr, device, early_stop=5, critic_update_freq=5, save_path=None):
        """
        Train the WGAN model with gradient penalty.
        Best model weights are saved, and the improvement is decided by Generator loss.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train.
            warm_up (int): Number of warm-up epochs which will not count toward early stopping.
            lr (float): Learning rate.
            device (torch.device): Device to train on.
            early_stop (int, optional): Early stopping patience. Defaults to 5.
            critic_update_freq (int): Number of Critic updates per Generator update.
            save_path (str): Full path where to save the model weights, which is identical to pretrained_path.
        """
        # Initialization update
        self.cat_column_indices = train_loader.cat_column_indices
        
        # Optimizers
        optim_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_critic = optim.Adam(self.critic.parameters(), lr=lr, betas=(0.5, 0.999))

        # Early stopping variables
        best_loss = float("inf")
        stop_counter = 0
        best_epoch = 0

        # Loss tracking for plotting
        gen_losses = []
        critic_losses = []
        gp_values = []

        for epoch in range(epochs):
            gen_loss_epoch, critic_loss_epoch, gp_epoch  = 0, 0, 0
            self.train()

            for real_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)

                # Encode real data into latent space
                with torch.no_grad():
                    real_data = self.autoencoder.encoder(real_data)

                # === Train Critic ===
                for _ in range(critic_update_freq):  # More frequent Critic updates
                    z = torch.randn(batch_size, self.noise_dim).to(device)
                    fake_data = self.generator(z)
                    fake_data = fake_data.detach()  # Detach fake_data from computation graph

                    # Real and fake outputs
                    real_validity = self.critic(real_data)
                    fake_validity = self.critic(fake_data)

                    # Compute gradient penalty
                    gp = self._compute_gradient_penalty(real_data, fake_data)

                    # Critic loss
                    critic_loss = fake_validity.mean() - real_validity.mean() + gp

                    # Update Critic
                    optim_critic.zero_grad()
                    critic_loss.backward()
                    optim_critic.step()

                    critic_loss_epoch += critic_loss.item()
                    gp_epoch += gp.item()  # Add GP to the running total

                # === Train Generator ===
                z = torch.randn(batch_size, self.noise_dim).to(device)
                fake_data = self.generator(z)
                fake_validity = self.critic(fake_data)

                # Generator loss
                gen_loss = -fake_validity.mean()

                # Update Generator
                optim_gen.zero_grad()
                gen_loss.backward()
                optim_gen.step()

                gen_loss_epoch += gen_loss.item()

            # Average losses for the epoch
            gen_loss_epoch /= len(train_loader)
            critic_loss_epoch /= len(train_loader)
            gp_epoch /= (len(train_loader) * critic_update_freq)

            # Append losses for plotting
            gen_losses.append(gen_loss_epoch)
            critic_losses.append(critic_loss_epoch)
            gp_values.append(gp_epoch)

            # Print progress
            print(f"[Training Status]: Epoch {epoch+1}: Generator Loss: {gen_loss_epoch:.4f}, Critic Loss: {critic_loss_epoch:.4f}, GP: {gp_epoch:.4f}")

            # Early stopping logic
            if gen_loss_epoch < best_loss and epoch >= warm_up:
                best_loss = gen_loss_epoch
                best_epoch = epoch
                stop_counter = 0
                self.save_weights(epoch, gen_loss_epoch, critic_loss_epoch, save_path)
            elif epoch >= warm_up:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    break

        # Plot losses at the end of training
        self._plot_losses(gen_losses, critic_losses, best_epoch, gp_values)
        
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
        
    def _plot_losses(self, gen_losses, critic_losses, best_epoch=None, gp_values=None):
        """
        Plot Generator and Critic losses, and optionally Gradient Penalty, on the same axis.
        
        Args:
            gen_losses (list): List of generator losses for each epoch.
            critic_losses (list): List of critic losses for each epoch.
            best_epoch (int, optional): The index of the best epoch to highlight. Defaults to None.
            gp_values (list, optional): List of gradient penalties for each epoch. Defaults to None.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(gen_losses, label='Generator Loss', color='blue')
        plt.plot(critic_losses, label='Critic Loss', color='red')
        
        # Plot gradient penalties if provided
        if gp_values is not None:
            plt.plot(gp_values, label='Gradient Penalty', color='green', linestyle='--')
        
        # Highlight the best epoch
        if best_epoch is not None:
            plt.axvline(best_epoch, color='purple', linestyle='--', label=f'Best Epoch ({best_epoch})')

        plt.title('Training Losses and Gradient Penalty')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
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
        self.autoencoder.decoder.eval()
        
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
            synthetic_data = self.autoencoder.decoder(latent_data)
        
        # Post-process categorical columns
        synthetic_data = self._post_processing(synthetic_data)
        
        return synthetic_data

    
class WcGAN(WGAN):
    """
    A conditional GAN architecture inheriting from WGAN.
    """
    def __init__(self, gen_config, critic_config, autoencoder, num_classes, pretrained_path=None, lambda_gp=10.0):
        """
        Initialize the conditional GAN with generator and critic architectures.
        
        Args:
            gen_config (list[dict]): Configuration for generator layers.
            critic_config (list[dict]): Configuration for critic layers.
            autoencoder (Autoencoder): A trained AE object with correct dimensions.
            num_classes (int): Dimension of the one-hot encoded labels.
            pretrained_path (str, optional): Path to pretrained models directory.
            lambda_gp (float): Gradient penalty weight. Defaults to 10.0.
        """
        # Adjust input dimensions for conditional inputs
        gen_config[0]["input_dim"] += num_classes  # Add num_classes to generator input
        critic_config[0]["input_dim"] += num_classes  # Add num_classes to critic input
        
        super(WcGAN, self).__init__(gen_config, critic_config, autoencoder, pretrained_path, lambda_gp)
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
            real_data (torch.Tensor, optional): Real data for the critic.
        
        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: Outputs of generator and critic.
        """
        # Concatenate noise and labels for generator input
        z = torch.cat([z, labels], dim=1)
        generated_data = self.generator(z)

        if real_data is not None:
            # Concatenate labels with real and generated data for critic input
            critic_real_input = torch.cat([real_data, labels], dim=1)
            critic_fake_input = torch.cat([generated_data, labels], dim=1)
            
            critic_real = self.critic(critic_real_input)
            critic_fake = self.critic(critic_fake_input)
            return generated_data, critic_real, critic_fake
        
        return generated_data

    def train_model(self, train_loader, epochs, warm_up, lr, device, early_stop=5, critic_update_freq=5, save_path=None):
        """
        Train the conditional GAN with one-hot encoded labels.
        """
        # Initialization update
        self.cat_column_indices = train_loader.cat_column_indices
        
        # Optimizers
        optim_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optim_critic = optim.Adam(self.critic.parameters(), lr=lr, betas=(0.5, 0.999))

        # Early stopping variables
        best_loss = float("inf")
        stop_counter = 0
        best_epoch = 0

        # Loss tracking for plotting
        gen_losses = []
        critic_losses = []
        gp_values = []

        for epoch in range(epochs):
            gen_loss_epoch, critic_loss_epoch, gp_epoch = 0, 0, 0
            self.train()

            for real_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                real_data, labels = real_data.to(device), labels.to(device)
                batch_size = real_data.size(0)

                # Encode real data into latent space
                with torch.no_grad():
                    real_data = self.autoencoder.encoder(real_data)

                # One-hot encode labels if necessary
                if labels.dim() == 1:
                    labels = F.one_hot(labels, num_classes=self.num_classes).float()

                # === Train Critic ===
                for _ in range(critic_update_freq):
                    z = torch.randn(batch_size, self.noise_dim).to(device)
                    z = torch.cat([z, labels], dim=1)  # Concatenate noise with labels
                    fake_data = self.generator(z).detach()  # Detach to avoid backprop through generator

                    # Real and fake inputs for critic
                    critic_real_input = torch.cat([real_data, labels], dim=1)
                    critic_fake_input = torch.cat([fake_data, labels], dim=1)

                    # Critic outputs
                    critic_real = self.critic(critic_real_input)
                    critic_fake = self.critic(critic_fake_input)

                    # Compute gradient penalty
                    gp = self._compute_gradient_penalty(critic_real_input, critic_fake_input)

                    # Critic loss
                    critic_loss = critic_fake.mean() - critic_real.mean() + gp

                    # Update Critic
                    optim_critic.zero_grad()
                    critic_loss.backward()
                    optim_critic.step()

                    critic_loss_epoch += critic_loss.item()
                    gp_epoch += gp.item()

                # === Train Generator ===
                z = torch.randn(batch_size, self.noise_dim).to(device)
                z = torch.cat([z, labels], dim=1)
                fake_data = self.generator(z)
                critic_fake_input = torch.cat([fake_data, labels], dim=1)
                critic_fake = self.critic(critic_fake_input)

                # Generator loss
                gen_loss = -critic_fake.mean()

                # Update Generator
                optim_gen.zero_grad()
                gen_loss.backward()
                optim_gen.step()

                gen_loss_epoch += gen_loss.item()

            # Average losses for the epoch
            gen_loss_epoch /= len(train_loader)
            critic_loss_epoch /= len(train_loader)
            gp_epoch /= (len(train_loader) * critic_update_freq)

            # Append losses for plotting
            gen_losses.append(gen_loss_epoch)
            critic_losses.append(critic_loss_epoch)
            gp_values.append(gp_epoch)

            # Print progress
            print(f"[Training Status]: Epoch {epoch+1}: Generator Loss: {gen_loss_epoch:.4f}, Critic Loss: {critic_loss_epoch:.4f}, GP: {gp_epoch:.4f}")

            # Early stopping logic
            if gen_loss_epoch < best_loss and epoch >= warm_up:
                best_loss = gen_loss_epoch
                best_epoch = epoch
                stop_counter = 0
                self.save_weights(epoch, gen_loss_epoch, critic_loss_epoch, save_path)
            elif epoch >= warm_up:
                stop_counter += 1
                if stop_counter >= early_stop:
                    print("[Training Status]: Early stopping triggered!")
                    break

        # Plot losses and gradient penalty at the end of training
        self._plot_losses(gen_losses, critic_losses, best_epoch, gp_values)
                
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
        print("[Main]: Training a WGAN model...")
        model = WGAN(gen_config=GENERATOR_CONFIG,
                    critic_config=CRITIC_CONFIG,  # Use the updated CRITIC_CONFIG
                    autoencoder=autoencoder,
                    pretrained_path=PRETRAIN_PATH,
                    lambda_gp=LAMBDA_GP)  # Set gradient penalty weight
        
        model.train_model(
            train_loader=train_loader, 
            epochs=EPOCHS,
            warm_up=WARMUP_EPOCHS, 
            lr=BASE_LEARNING_RATE, 
            device=DEVICE, 
            early_stop=GAN_EARLY_STOP,
            critic_update_freq=CRITIC_UPDATE_FREQ, 
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
        model = WcGAN(gen_config=GENERATOR_CONFIG,
                    critic_config=CRITIC_CONFIG,
                    autoencoder=autoencoder,
                    lambda_gp=LAMBDA_GP,
                    num_classes=NUM_CLASSES,
                    pretrained_path=PRETRAIN_PATH)
        
        model.train_model(train_loader=train_loader, 
            epochs=EPOCHS,
            warm_up=WARMUP_EPOCHS, 
            lr=BASE_LEARNING_RATE, 
            device=DEVICE, 
            early_stop=GAN_EARLY_STOP,
            critic_update_freq=CRITIC_UPDATE_FREQ, 
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