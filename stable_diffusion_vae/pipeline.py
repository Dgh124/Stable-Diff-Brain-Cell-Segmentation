import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
import os

WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    input_image=None,
    strength=0.8,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
):
    with torch.no_grad():
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
  
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler value {sampler_name}")

        latents_shape = (1, 32, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image is not None:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = np.array(input_image, dtype=np.float32)
            input_image_tensor = torch.tensor(input_image_tensor, device=device)
            print(f'\n\nINPUT.SHAPE DADDY: {input_image_tensor}\n\n')
            input_image_tensor = input_image_tensor.unsqueeze(0)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents, mu, logvar, encoded_features = encoder(input_image_tensor, encoder_noise)

            # Here we only add initial noise once using add_noise(...)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])[0]
            to_idle(encoder)

        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
            encoded_features = None

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for timestep in timesteps:
            time_embedding = get_time_embedding(timestep).to(device)

            # 1) Predict noise with your diffusion model
            model_output = diffusion(latents, time_embedding)

            # 2) Use sampler.step(...) to remove noise from latents
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents, encoded_features)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        return images[0]



def rescale(x, old_range, new_range, clamp=False):
    """
    Rescales tensor x from old_range to new_range. If clamp=True, x is clipped
    to [new_min, new_max] after rescaling.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = (x - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timesteps):
    """
    Basic positional embedding for timesteps, returning shape (B, 320).
    """
    # Ensure timesteps is at least 1D
    timesteps = timesteps.unsqueeze(0) if timesteps.dim() == 0 else timesteps
    device = timesteps.device

    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32, device=device) / 160)
    args = timesteps[:, None].float() * freqs[None, :]
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 320)
    return embeddings


import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

def kl_divergence(mu, logvar):
    # KL from N(mu, var) to N(0, 1)
    # shape: (batch_size, latent_dim)
    # typically sum or mean across dims
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()


def train_model(
    models,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    n_epochs,
    num_timesteps,
    device,
    accumulation_steps,
    mini_patience,
    full_patience
):
    """
    Trains a diffusion-based model that uses a VAE encoder requiring an extra `noise` argument.
    """
    encoder = models['encoder']
    decoder = models['decoder']
    diffusion = models['diffusion']
    beta = 0.5
    
    # Move all models to the specified device
    encoder.to(device)
    decoder.to(device)
    diffusion.to(device)
    
    # Sampler used for adding noise at various timesteps
    sampler = DDPMSampler(generator=torch.Generator(device=device), num_training_steps=num_timesteps)

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Track learning rate
    init_lr = optimizer.param_groups[0]['lr']
    cur_lr = prev_lr = init_lr

    # Early stopping parameters
    early_stopping = 0
    mini_patience = mini_patience  
    full_patience = full_patience

    # Best model tracking
    best_loss = float('inf')
    # best_encoder_weights = copy.deepcopy(encoder.state_dict())
    # best_decoder_weights = copy.deepcopy(decoder.state_dict())
    best_diffusion_weights = copy.deepcopy(diffusion.state_dict())

    global_step = 0

    for epoch in range(n_epochs):
        # -----------------------------
        #          TRAINING
        # -----------------------------
        encoder.eval()
        decoder.eval()
        diffusion.train()
  
        epoch_loss = 0.0
        optimizer.zero_grad()

        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Training]"):
            images = images.to(device)
            batch_size = images.size(0)
            
            global_step += 1
            
            # 1) Prepare shape for encoder_noise
            latents_height = images.size(2) // 8
            latents_width = images.size(3) // 8
            noise_shape = (batch_size, 32, latents_height, latents_width)

            # 2) Create random noise for the VAE_Encoder
            encoder_noise = torch.randn(noise_shape, device=device)

            # 3) Encode images into latents, passing noise as well
            latents, mu, logvar, encoded_features = encoder(images, encoder_noise)

            # 4) Randomly sample timesteps
            # timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            # e.g. T=1000
            max_t = num_timesteps - 1  # e.g. 999
            timesteps = max_t * torch.ones(batch_size, device=device).long()


            # 5) Compute time embeddings
            time_embeddings = get_time_embedding(timesteps).to(device)

            # 6) Add noise to latents at these timesteps
            noisy_latents, noise = sampler.add_noise(latents, timesteps)

            # 7) Model predicts the noise
            model_output = diffusion(noisy_latents, time_embeddings)

            # 8) Compute loss
            # MSE_loss = criterion(model_output, noise)
            # kl_loss = kl_divergence(mu, logvar)
            # loss = MSE_loss + (beta * kl_loss)
            
            loss = criterion(model_output, noise)
            
            loss = loss / accumulation_steps

            # Backprop
            loss.backward()

            epoch_loss += loss.item() * accumulation_steps  # scale back up to store the true total

            # Only step optimizer after "accumulation_steps" mini-batches
            if (global_step + 1) % accumulation_steps == 0 or (global_step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        
        # If the last batch didn't align with accumulation_steps exactly,
        # there's no final step() call after the loop. If you want to handle that,
        # you could do an additional check here.

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Training Loss: {avg_epoch_loss:.4f}")
        

        # VALIDATION
        encoder.eval()
        decoder.eval()
        diffusion.eval()

        val_loss = 0.0
        with torch.no_grad():
            for images in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Validation]"):
                images = images.to(device)
                batch_size = images.size(0)

                # 1) Prepare shape for encoder_noise
                latents_height = images.size(2) // 8
                latents_width = images.size(3) // 8
                noise_shape = (batch_size, 32, latents_height, latents_width)

                # 2) Create random noise for encoder
                encoder_noise = torch.randn(noise_shape, device=device)

                # 3) Encode images
                latents, mu, logvar, encoded_features = encoder(images, encoder_noise)

                # 4) Randomly sample timesteps
                timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device).long()

                # 5) Time embeddings
                time_embeddings = get_time_embedding(timesteps).to(device)

                # 6) Add noise
                noisy_latents, noise = sampler.add_noise(latents, timesteps)

                # 7) Model predicts noise
                model_output = diffusion(noisy_latents, time_embeddings)

                # 8) Compute loss
                loss = criterion(model_output, noise).type(torch.float64)
                val_loss += loss.item() * batch_size

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
        # --- STEP LR SCHEDULER WITH AVERAGE LOSS ---
        scheduler.step(avg_val_loss)
        cur_lr = scheduler.optimizer.param_groups[0]['lr']
        if abs(cur_lr - prev_lr) > 1e-12:
            print(f"Learning rate updated: {cur_lr}")
            prev_lr = cur_lr

        # -----------------------------
        #    SAVE BEST MODEL
        # -----------------------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stopping = 0
            best_diffusion_weights = copy.deepcopy(diffusion.state_dict())
            
            torch.save(diffusion.state_dict(), f"/content/drive/MyDrive/Stable_diff/models/sd_models/best_diff_model-epoch:{epoch+1}-loss:{best_loss:.6f}.pth")
            print(f"new best model(s) saved with best_loss={best_loss:.6f}")
            
        else:
            early_stopping += 1
            if early_stopping >= full_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            if early_stopping >= mini_patience:
                print(f"Reverting weights at epoch {epoch+1} to best model so far.")
                diffusion.load_state_dict(best_diffusion_weights)

        # if (global_step + 1) % 100 == 0:
        #     torch.save(diffusion.state_dict(), 
        #               f"/content/drive/MyDrive/Stable_diff/complete_tdata/overfitt_debug/best_diff_model-epoch:{epoch+1}-loss:{best_loss:.6f}.pth")
        #     print(f"New best model saved with best_loss={best_loss:.6f}")

                
    diffusion.load_state_dict(best_diffusion_weights)
    torch.save(diffusion.state_dict(), f"/content/drive/MyDrive/Stable_diff/models/sd_models/best_diff_model-epoch:{epoch+1}-loss:{best_loss:.6f}.pth")
    print(f"Final diff model saved with best_loss={best_loss:.6f}")
  