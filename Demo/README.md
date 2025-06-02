# Burst2Scene GAN – Project Submission

This repository contains the implementation and demonstration notebook for the **Burst2Scene AI** project, which generates a high-quality scene image from a burst of video frames using a GAN-based architecture.

## 📦 Deliverables

# Project README

<!-- Your existing content here -->

## Model Storage and Preview

### Accessing Trained Models

Due to file size constraints, trained model files are stored externally on Google Drive.

You can access the full collection of trained models and related files here:

[Google Drive Folder - Trained Models](https://drive.google.com/drive/folders/1HW1A9QRJDhRX8VmGBMufexJd59YFDKda?usp=share_link)

### Preview Directory

Within the Drive folder, the `preview` directory contains example outputs and models used for inference on unseen burst photo datasets.

This directory is a great place to start exploring the performance of the trained models on new data without the need to run the full training pipeline.

---

### Downloading Models

To download models directly in your environment (e.g., Google Colab), you can use:


pip install gdown
gdown --folder https://drive.google.com/drive/folders/1HW1A9QRJDhRX8VmGBMufexJd59YFDKda


The following items are included as part of the final submission:

### ✅ Notebook
- `Burst2Scene_Demo.ipynb` – Complete notebook with:
  - Preprocessing
  - Generator and discriminator loading
  - Inference on burst samples
  - Inline `preview.py` for evaluating trained model (PSNR, SSIM, Confidence)
  - Visual previews saved to `model_preview_dir/`

### ✅ Python Scripts
These scripts are placed in the **project root**:
- `generator_colab.py`
- `discriminator_PatchGAN.py`
- `validation_dataset_colab.py`
- `preview.py`

### ✅ Pretrained Models
These files must be in the **root directory**:
- `generator_last.pth`
- `discriminator_last.pth`

### ✅ Dataset Folders
1. `demo_burst/`  
   - Example burst used in the notebook  
   - At least one folder, e.g. `Validation Video 6_burst_00/` containing `frame_00.jpg` to `frame_09.jpg`  
2. `burst_validation_high_variation/`  
   - Multiple folders like `burst_00/`, `burst_01/` for full evaluation  
   - Each folder should include 10 `.jpg` or `.png` frames  
3. `model_preview_dir/`  
   - Created at runtime to store generated image previews and evaluation metrics  

## 🔧 Setup Instructions

1. **Install dependencies** (virtual environment recommended):


pip install -r requirements.txt
