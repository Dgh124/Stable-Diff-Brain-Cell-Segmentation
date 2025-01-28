 # Stable-Diff-Brain-Cell-Segmentation
 This project aims to build a pipeline that feeds generated training data (brain cell scans) into an image segmentation model that can output cell shape, size and positioning. There are two main goals for this project. <br><br>
 (1) Train a modified Stable Diffusion model (based on DDPM) to create realistic cell scans. <br>
 (2) Optimize a U-net based segmentation model to predict grayscale cell body masks. <br> 

A couple notes about the project:
* All training images come are originally in tif format and come from a publicly available Electron Microscopy Dataset, linked below:
    * https://www.epfl.ch/labs/cvlab/data/data-em/
* Tif images were originally 1024x768, broken into 12 256x256 chunks for training
* Images are kept grayscale and normalized with a mean=0.5 and std=0.5 for enhanced performance

 What has been accomplished thus far, is a modified Stable Diffusion model that produces 
 
![Screenshot 2025-01-28 at 3 45 37 PM](https://github.com/user-attachments/assets/6c120037-6209-436b-a6c8-a8140d811a0d)
