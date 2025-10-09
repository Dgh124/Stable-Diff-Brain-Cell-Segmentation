 ## Stable-Diff-Brain-Cell-Segmentation
 This project aims to build a pipeline that feeds generated training data (brain cell scans) into an image segmentation model that can output cell shape, size and positioning. The two main goals for this project are: <br><br>
 (1) Train a modified Stable Diffusion model (based on DDPM) to create realistic cell scans. <br>
 (2) Optimize a U-net based segmentation model to predict grayscale cell body masks. <br> 

## Whats in the repo?

Essentially all model/training files and some samples from the training dataset. Unfortunately, the model files are too large to upload to Git (25MB file size limit), but I am looking into workarounds for this (potentially Dropbox).

## Where does the data come from?
* All images used are originally in .tif format and come from a publicly available Electron Microscopy Dataset, linked below:
    * https://www.epfl.ch/labs/cvlab/data/data-em/
* Tif images were originally 1024x768, broken into 12 256x256 chunks for training (per image)
* This expansion generated 12,780 unique chunks for training/validation
* Images are kept grayscale and normalized with a mean=0.5 and std=0.5 for enhanced performance

# How were the models trained?

Stable diffusion training was done in PyTorch. For image segmentation, model.fit() was used in Tensorflow. 
To make up for low GPU ram, training was often performed with __gradient accumulation__ and larger batch sizes. Both sets incorporated a learning rate scheduler, and used validation loss for optimization. They also used checkpoint callbacks that saved the best models so far, and reverted to earlier ones after x epochs of no improvement. More specific details are listed under each section.

I initially trained the img. segmentation model and VAE on my Macbook MPS. When the intensity of the project increased, I moved to Google Colab and trained on Nvidia GPUs. I currently use Georgia Tech's PACE-ICE cluster and train on the CUDA enabled GPUs they provide (likewise Nvidia). My favorite go to has been 2-3 H100's with 50g of ram.

![Screenshot 2025-01-31 at 6 50 36 PM](https://github.com/user-attachments/assets/a8e13e03-a701-4a18-8527-8cd2c0d73bff)

# Stable Diffusion - Variational Autoencoder portion

The development process went as follows:
1. Train and optimize weights for VAE (Variational Autoencoder)
    - **Loss function**: MSE + beta * KL Divergence
    - **MSE** is used to find mean squared error of the difference between each pixel in input image & output
    - **KL** divergence term calculates distance from one distribution to another, in this case I used normal distr. (mean=0, std=1.0)
    - __Optimizer__: SDG with momentum + lr-scheduler 
 2. Loss reached a minimum of ~0.003076

Below is a progression of model improvement in reconstructing images over time.
    
![WhatsApp Image 2024-11-01 at 14 44 32](https://github.com/user-attachments/assets/89fc4aaf-f90e-4e88-910e-3c60ce8dfb35)

And here are two samples with the highest performing model. A slight amount of noise sampled from a normal distr. is utilized in the encoding process and the model learns to discern this noise and reconstruct from the latent vectors.
![Screenshot 2025-01-28 at 4 04 54 PM](https://github.com/user-attachments/assets/cc747ac2-a54b-4656-875a-37c025150158)

![Screenshot 2025-01-28 at 4 07 24 PM](https://github.com/user-attachments/assets/e70053b3-c2ea-40f1-a2ba-e6f01f200e7b)

 
# Stable Diffusion - Diffusion portion

During stable diffusion, noise is first added according to a beta-scheduler. This algorithm also encodes a timestep vector at which the noise was added (positional encoding), s.t the model is given context to infer how much noise is present based on the time embedding. Below is a visual overview of the diffusion process this project follows, without the use of CLIP text embeddings (no need to train on a prompt - all images are of brain cells).

* All training done through __pipeline.py__ train_model() func!
* __Loss function__: MSE 

<img width="1033" alt="Screenshot 2025-01-28 at 5 09 33 PM" src="https://github.com/user-attachments/assets/808c2cc6-5470-4618-906a-0ba3b6da21b3" />
Courtesy of Umar Jamil <br>


Once more, the math behind this model is very complex and based on the 2020 paper "Denoising and Diffusing Probabalistic Models" by Jonathan Ho, Ajay Jain, Pieter Abbeel. It is a bit too lengthy to go into in the README, but the pdf of this paper can be found at:

https://arxiv.org/abs/2006.11239

After hundreds of epochs of training and tweaking the diffusion process (removing unecessary residual and attention blocks, increasing the number of neural net layers for time embeddings (so time embeddings are added to the latent vector in steps of (1, 320) -> (320, 640) -> (640, 1024) rather than jumping from (1,320) -> (320,1024)). Training was performed with 1000 timesteps, and sampling from the model works with just 50. Below are some sample images generated at runtime.

**Attention batch 1**

<img width="256" height="256" alt="sample_0" src="https://github.com/user-attachments/assets/00f651f7-15dd-4487-ae85-27199118ff3c" />
<img width="256" height="256" alt="sample_2" src="https://github.com/user-attachments/assets/4f7565fd-c552-42c5-9478-159cef8f8205" />
<img width="256" height="256" alt="sample_5" src="https://github.com/user-attachments/assets/e0ba31d9-6eec-4ef4-b28f-af05cdb51f77" />

**Attention batch 2**

<img width="256" height="256" alt="sample_5" src="https://github.com/user-attachments/assets/108714a4-356b-4fcf-b6c5-a913a15f4c10" />
<img width="256" height="256" alt="sample_4" src="https://github.com/user-attachments/assets/b2bee1e8-908b-4a22-b90a-0a5a5a3a57c4" />
<img width="256" height="256" alt="sample_3" src="https://github.com/user-attachments/assets/a05483e6-8605-42c5-bbf6-a8e4dd047b73" />

**Attention batch 3** 

<img width="256" height="256" alt="sample_7" src="https://github.com/user-attachments/assets/c6963ac4-5877-43e7-b9f2-6aeb44cc3983" />
<img width="256" height="256" alt="sample_2" src="https://github.com/user-attachments/assets/bfca1814-f3a9-4d7b-b787-f9f42f39d100" />
<img width="256" height="256" alt="sample_1" src="https://github.com/user-attachments/assets/1b4a19ac-5010-4210-ad9b-96eee90769bf" />

These images are clearly un-usable as synthetic training data, and I will discuss some of the flaws of my approach that led to these results under "What I learned about generating synthetic data for training".

## Image Segmentation

This model follows the standard U-net architecture. It utilizes encode/decode blocks + residual to capture relevant details about cell shapes. <br>
* __Loss function__: binary cross entropy + iou (intersection over union)
* __Optimizer__: Adam + lr-scheduler <br> 

The model was trained in a supervised manner and optimized with ground truth image masks. Below is an example of a scan and its respective mask:

<img width="662" alt="Screenshot 2025-01-28 at 7 09 48 PM" src="https://github.com/user-attachments/assets/4b776cc9-496f-414b-9a5a-86b7fe0ce522" /> <br>

The model outputs a 256x256 prediction mask, where each pixel has a probability on the scale 0-1.0. By adjusting the threshold for this probability, one can choose their strength of likelihood for what the mask looks like. A good cutoff has centered around ~0.05. Below is an example output from a model with loss ~0.21. <br>
 
![Screenshot 2025-01-28 at 3 45 37 PM](https://github.com/user-attachments/assets/6c120037-6209-436b-a6c8-a8140d811a0d)

As you can see, it is able to roughly identify the shape, size and positioning of cell bodies, and if I were to lower the likelihood, noise would be sacrificed along with detail. 

## Why use AI generated data in training?

While it is true that, in respect to LLMs, training models on synthetic data leads to model collapse - this is mostly due to the model learning weights of the model it's data was produced by. The immediate outcome is a loss of diversity in training data, and a model that performs poorly on real data. The goal of training a stable diffusion model on brain cell scans is not to replace real data - it is to supplement real data with rarer cell types that are underrepresented in publicly available datasets. This can help the model learn to identify signs associated with rare diseases. Another reason is simply that, as generated content continues to be consumed more and more, it is expected that one day they will be used to train themselves. As such, building the infrastructure that experiments with this process is important, and its effect can be measured in a meaningful way.

# What I learned about generating synthetic data for training

I will first critique my approach on a technical code-level (look at flaws and fixes), then a general design choice level that reflects the then "current" limitations of diffusion models. 

## Technical critiques

unet_diffusion.py · UNet forward path lacks timestep conditioning
Issue: The forward(x, t) accepts t but it is not embedded or injected into blocks. The denoiser learns a single average denoise → blurred samples across noise levels.
Fix: Add sinusoidal timestep embeddings → MLP → inject via FiLM/affine modulation into every ResBlock (both down/bridge/up paths) and into attention blocks.
Outcome: Noise-level–aware denoising, sharper late-step details, reduced over-smoothing.

unet_diffusion.py · UpBlock merge under-parameterized
Issue: In UpBlock, features are upsampled and then merged with skip features via only one residual stage; use_attention=False by default. The concatenated tensor (in_ch + skip_ch) is compressed too quickly.
Fix: Keep use_attention=True; insert two ResBlocks post-merge (merge → Res → Attn → Res). Use GroupNorm; widen channels before projecting to out_ch.
Outcome: Better skip fusion and spatial alignment; preserves high-freq membranes after upsampling.

unet_diffusion.py · DownBlock lacks anti-aliasing before strided conv
Issue: Strided convs downsample without anti-aliasing, causing ringing and loss of fine boundaries.
Fix: Add blur/avg-pool anti-aliasing (e.g., 3×3 blur) before each stride-2 downsample.
Outcome: Cleaner feature pyramids; less aliasing of thin EM structures.

unet_diffusion.py · Attention missing at highest and lowest resolutions
Issue: Self-attention is absent or disabled at 16×16/32×32 and 128×128/256×256 stages; only mid-res sees global context.
Fix: Enable attention at lowest (bridge) and one high-res stage; use relative positional encodings.
Outcome: Global morphology consistency + high-res texture anchoring.

vae_encoder.py · Excessive compression without skip pathways
Issue: 256×16×16 (65,536) features collapse straight to a small latent_dim via a single FC; no residual/attention; no skip connections to decoder. High-freq detail lost before diffusion.
Fix: Increase latent_dim (e.g., 512–1024); replace FC with conv bottleneck; add ResBlocks + GroupNorm; optionally self-attention at 32×32/16×16.
Outcome: Richer latent preserving edges/textures that the decoder and diffusion can recover.

vae_decoder.py · Output activation mismatched with dataset normalization
Issue: Decoder ends with sigmoid (0–1), while training pipeline normalizes images to mean=0.5, std=0.5 (≈[-1,1]). This compresses contrast or forces ad-hoc re-scaling.
Fix: If the dataset is standardized to [-1,1], end with tanh and compute losses in standardized space; otherwise keep data in [0,1] end-to-end and remove standardization.
Outcome: Correct dynamic range; better micro-contrast and less desaturation.

vae_* + training loop · MSE-only objective encourages smoothness
Issue: Both VAE recon and diffusion training use MSE only; pixelwise loss averages out fine texture.
Fix: Add perceptual loss (LPIPS/SSIM) to VAE; optionally a light GAN loss (patch-GAN) at decoder output; for diffusion, add edge-aware auxiliary loss on late steps.
Outcome: Preserves membrane/nuclear boundaries while keeping stability.

unet_diffusion.py · Sampler and β-schedule are vanilla linear
Issue: Linear β with few sampling steps (~50) limits micro-detail; late steps under-refine.
Fix: Use cosine β schedule; implement DDIM / DPM-Solver samplers; allocate more steps to the last 20–30% of the trajectory; maintain EMA weights for sampling.
Outcome: Sharper reconstructions at equal wall-clock; better edge recovery.

unet_diffusion.py · Image-space diffusion only, no structure conditioning
Issue: Unconditional image-space DDPM drifts; lacks constraints to preserve EM morphology.
Fix: Add a ControlNet-style branch conditioned on Canny/Sobel edges or coarse masks (downsampled) and inject into UNet at each scale.
Outcome: Structure-anchored synthesis; edges align with plausible anatomy.

vae_* + unet_diffusion.py · Single-stage 256×256 training limits fidelity
Issue: Trying to get both global shape and micro-texture at 256 in one go forces trade-offs.
Fix: Move to latent-space diffusion (diffuse at 16×16 with VAE latents) or adopt a two-stage cascade: base 256 → super-resolution diffusion to 512/1024 with perceptual loss.
Outcome: Base model captures semantics; SR stage restores fine EM texture critical for segmentation.

What these fixes accomplish overall:
Make the denoiser noise-aware, stop throwing away detail in the VAE, align activation ranges with preprocessing, impose perceptual/edge constraints, and add multiscale + conditioning. Together, these changes directly target the observed blur and produce synthetic EM images that hold up under downstream Dice/IoU checks.

## Issues with choosing stable diffusion over traditional data augmentation

In 2024, latent diffusion was excellent for natural images, but it was a poor first tool for my goal: increasing *rare* cellular examples for segmentation. Classic augmentations are cheap, label-preserving, auditable, and run inline with training. Stable Diffusion requires training a large generator and then sampling full images. That is far more compute, more code, and more moving parts. My compute was not the limiting factor. Time-to-useful-data and operational complexity were.

Diffusion learns the empirical data distribution. Rare phenomena live in the tails. With few rare exemplars, the model underfits those tails and gravitates to the mean. Generated images look plausible but trend toward average morphologies. That is the opposite of what I needed. Without strong structure conditioning and carefully curated rare cases, the model interpolates common modes and hallucinates the rest.

There is also an objective mismatch. The diffusion loss predicts noise under an MSE-style objective. That optimizes for per-pixel agreement, not morphological correctness. Coupled with a VAE bottleneck, high-frequency boundaries get smoothed. Segmentation depends on those boundaries. Classic augmentations preserve geometry while diversifying appearance in controlled ways (e.g., elastic, intensity, noise). They expand coverage along axes that matter and keep labels exact.

Labeling is a hard blocker. Synthetic images are not useful for supervised segmentation unless masks are correct. I did not have a reliable mask-to-image or simulation-first pipeline. Post-hoc masks from my own segmenter would introduce label noise. Classic augmentations keep labels by construction or transform them deterministically. That keeps supervision clean.

Validation was another gap. I did not have ongoing access to domain experts to confirm that generated structures were anatomically consistent with EM biology. Visual plausibility is not enough. Without expert review and a pre-registered downstream test (train on synthetic + small real, evaluate on held-out real), it is easy to drift the distribution. Augmentations are simpler to validate. They are transparent and bounded.

Finally, sample efficiency matters. Generating a new image is expensive. Augmentations reuse scarce real images and multiply them with simple transforms. That improves class balance fast. It also integrates directly into training loops without new infrastructure.

Net: Stable Diffusion can help, but only after I have domain-aware constraints and guardrails. That means a sharper microscopy-tuned autoencoder, structure conditioning (edges or masks), cascaded super-resolution, and a utility-first gate with expert review. Starting with diffusion before I had those pieces—and before I had rigorous validation—was the wrong order. For rare-class scarcity, the right baseline is targeted sampling plus classic augmentation, with strict class balancing and physics-aware noise/PSF simulation as needed. Generative models come later, and only if they prove downstream utility.


