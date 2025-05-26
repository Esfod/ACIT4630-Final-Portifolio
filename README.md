# ACIT4630-Final-Portifolio

# Burst2Scene: GAN-Based Scene Generation from Burst Frames

This repository contains the implementation of **Burst2Scene**, a deep learning framework that generates a high-resolution, realistic scene image from a burst of low-resolution video frames. The project leverages a ResNet-based generator trained using perceptual and adversarial losses, with a PatchGAN discriminator to enhance local texture realism.

## NOTE
Most of the projects source code and files are too large to upload to GitHub. Please Contact for the source code demonstration


## 📂 Project Structure
- `generator.py` – ResNet-based generator
- `discriminator.py` – Basic global discriminator
- `discriminator_PatchGAN.py` – PatchGAN-style discriminator with spectral normalisation
- `train_colab.py` – Training script (v1: Generator-only)
- `train_colab_PatchGAN.py` – Training script (v2: PatchGAN adversarial setup)
- `burst_dataset_pipeline.py` – Dataset class for loading burst input tensors
- `validation_dataset.py` – Validation set loader
- `validate_generator.py` – Evaluation script (FID, SSIM, PSNR)
- `preview.py` – Visual output preview for trained models
- `video_to_bursts.py` – Script to extract bursts from video input
- `checkpoints/` – Trained model weights (generator & discriminator)
- `previews/` – Sample output images
- `README.md` – This file




## Key Features

- Burst-to-image scene reconstruction from temporally adjacent frames
- Generator trained using perceptual + total variation + adversarial loss
- PatchGAN discriminator with spectral normalisation for local realism
- Preview script for side-by-side output evaluation
- Metric evaluation using SSIM, PSNR, and optional FID

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- torchvision
- numpy, matplotlib, tqdm, PIL
- (Optional) Google Colab Pro/Pro+ for faster training

Install requirements:
```bash
pip install -r requirements.txt
```

Download all the files in this GitHub repository.

## Requirements
Version 1 (Perceptual-only, no discriminator):
```python train_colab.py```

Version 2 (PatchGAN adversarial training):
```python train_colab_PatchGAN.py```
* Make sure to load the pre-trained generator from final_GAN_v1_generator.pth before starting adversarial fine-tuning


## Running the model
```train.py```: run script to train model
```train_colab.py```: run in google colab with A1000 gpu, this trains the model
```preview.py```: uses the trained model by exposing them to unseen bursts images



## Previewing Outputs
Run the preview script to visualise model output on the validation set:
```
python preview.py --checkpoint checkpoints/final_GAN_v2_generator.pth
```
* Output images will be saved under the previews/ directory.


## Citation
Burst2Scene: Adversarial Scene Synthesis from Burst Video Frames using a ResNet–PatchGAN Framework (2025)








