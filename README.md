 # Stable-Diff-Brain-Cell-Segmentation
 This project aims to build a pipeline that feeds generated training data (brain cell scans) into an image segmentation model that can output cell shape, size and positioning. There are two main goals for this project. <br><br>
 (1) Train a modified Stable Diffusion model (based on DDPM) to create realistic cell scans. <br>
 (2) Optimize a U-net based segmentation model to predict grayscale cell body masks. <br> 

A couple notes about the project:
* All training images come from a publicly available Electron Microscopy Dataset, linked below:
    * https://www.epfl.ch/labs/cvlab/data/data-em/

 What has been accomplished thus far, is a modified Stable Diffusion model that produces 
