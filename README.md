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

I initially trained the img. segmentation model and VAE on my Macbook MPS. When the intensity of the project increased, I moved to Google Colab and trained on Nvidia GPUs. I currently use Georgia Tech's PACE-ICE cluster and train on the CUDA enabled GPUs they provide (likewise Nvidia). My favorite go to has been the L40 with around ~25g of ram.

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

After hundreds of epochs of training and tweaking the diffusion process (removing unecessary residual and attention blocks, increasing the number of neural net layers for time embeddings (so time embeddings are added to the latent vector in steps of (1, 320) -> (320, 640) -> (640, 1024) rather than jumping from (1,320) -> (320,1024)). Training was performed with 1000 timesteps, and sampling from the model works quite well with just 50.

![WhatsApp Image 2025-01-09 at 16 31 17](https://github.com/user-attachments/assets/4e635200-0af4-47d1-88df-bf7077e8c5fe)

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
