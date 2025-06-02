# Burst2Scene GAN – Project Submission

This repository contains the implementation and demonstration notebook for the **Burst2Scene AI** project, which generates a high-quality scene image from a burst of video frames using a GAN-based architecture.

## 📦 Deliverables

The following items are included as part of the final submission:

### ✅ Notebook
- `Burst2Scene_InlinePreview.ipynb` – Complete notebook with:
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

1. **Install dependencies** (inside a virtual environment recommended):

```bash
pip install -r requirements.txt
```

2. **Run the notebook** using Jupyter or PyCharm’s Jupyter support:
```bash
jupyter notebook Burst2Scene_InlinePreview_JPGReady.ipynb
```

3. **Output will be saved to**:
```
model_preview_dir/preview_patchgan/
```

## 📂 Directory Structure

```
project_root/
├── Burst2Scene_InlinePreview_JPGReady.ipynb
├── generator_colab.py
├── discriminator_PatchGAN.py
├── validation_dataset_colab.py
├── preview.py
├── generator_last.pth
├── discriminator_last.pth
├── demo_burst/
│   └── Validation Video 6_burst_00/
│       ├── frame_00.jpg
│       └── ...
├── burst_validation_high_variation/
│   └── burst_00/
│       ├── frame_00.jpg
│       └── ...
└── model_preview_dir/
    └── preview_patchgan/
```

## 🔗 Dataset Access

If the full dataset is too large to submit directly, a download link will be provided here:
**[Insert Link to Dataset Here – remains active until exam results are released]**

---

## 📜 Notes
- All files are self-contained and runnable.
- The notebook has been tested on both CPU and GPU environments.
- Evaluation results (PSNR, SSIM) are saved per-sample for assessment.

