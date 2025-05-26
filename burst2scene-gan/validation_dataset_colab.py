
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BurstDataset(Dataset):
    def __init__(self, bursts_dir, burst_size=10, image_size=(256, 256), transform=None):
        self.bursts_dir = bursts_dir
        self.burst_size = burst_size
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.samples = self._gather_burst_folders()

    def _gather_burst_folders(self):
        folders = []
        for root, dirs, _ in os.walk(self.bursts_dir):
            for d in dirs:
                path = os.path.join(root, d)
                if os.path.exists(os.path.join(path, "frame_00.jpg")):
                    folders.append(path)
        return folders

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        burst_path = self.samples[idx]
        burst_frames = []

        for i in range(self.burst_size):
            frame_file = os.path.join(burst_path, f"frame_{i:02d}.jpg")
            img = Image.open(frame_file).convert("RGB").resize(self.image_size)
            burst_frames.append(self.transform(img))

        burst_tensor = torch.stack(burst_frames, dim=0)  # [10, 3, H, W]
        target_tensor = burst_tensor[0]  # Use frame_00 as the target

        return burst_tensor, target_tensor





