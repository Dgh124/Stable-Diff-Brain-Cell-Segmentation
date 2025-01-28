 # Stable-Diff-Brain-Cell-Segmentation
 This project aims to build a pipeline that feeds generated training data (brain cell scans) into an image segmentation model that can output cell shape, size and positioning. There are two main goals for this project. <br><br>
 (1) Train a modified Stable Diffusion model (based on DDPM) to create realistic cell scans. <br>
 (2) Optimize a U-net based segmentation model to predict grayscale cell body masks. <br> 

A couple outlining features about the project:
* All training images come are originally in tif format and come from a publicly available Electron Microscopy Dataset, linked below:
    * https://www.epfl.ch/labs/cvlab/data/data-em/
* Tif images were originally 1024x768, broken into 12 256x256 chunks for training
* Images are kept grayscale and normalized with a mean=0.5 and std=0.5 for enhanced performance

# Stable Diffusion - Variational Autoencoder portion

The development process went as follows:
1. Train and optimize weights for VAE (Variational Autoencoder)
    - **Loss function**: MSE + beta * KL Divergence
    - **MSE** is used to find mean squared error of the difference between each pixel in input image & output
    - **KL** divergence term calculates distance from one distribution to another, in this case I used normal distr. (mean=0, std=1.0)
 2. Loss reached a minimum of ~0.003076

Below is a progression of model improvement in reconstructing images over time.
    
![WhatsApp Image 2024-11-01 at 14 44 32](https://github.com/user-attachments/assets/89fc4aaf-f90e-4e88-910e-3c60ce8dfb35)

And here are two samples with the highest performing model. A slight amount of noise sampled from a normal distr. is utilized in the encoding process and the model learns to discern this noise and reconstruct from the latent vectors.
![Screenshot 2025-01-28 at 4 04 54 PM](https://github.com/user-attachments/assets/cc747ac2-a54b-4656-875a-37c025150158)

![Screenshot 2025-01-28 at 4 07 24 PM](https://github.com/user-attachments/assets/e70053b3-c2ea-40f1-a2ba-e6f01f200e7b)

 
 What has been accomplished thus far, is a modified Stable Diffusion model that produces 
 
![Screenshot 2025-01-28 at 3 45 37 PM](https://github.com/user-attachments/assets/6c120037-6209-436b-a6c8-a8140d811a0d)
