"""
data pipeline for the generator
"""
import os
from PIL import Image
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import transforms

class BurstOnlyDataset(Dataset):
    def __init__(self, bursts_dir, burst_size=10, transform=None, image_size=(256, 256)):
        self.bursts_dir = bursts_dir
        self.burst_size = burst_size
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        self.samples = self._gather_burst_folders()

        print(f"Found {len(self.samples)} burst folders.")

    def _gather_burst_folders(self):
        folders = []
        for root, dirs, _ in os.walk(self.bursts_dir):
            for d in dirs:
                path = os.path.join(root, d)
                frame_path = os.path.join(path, "frame_00.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(path, "frame_00.png")
                if os.path.exists(frame_path):
                    folders.append(path)
        return folders

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        burst_path = self.samples[idx]
        frame_files = sorted([
            f for f in os.listdir(burst_path)
            if f.endswith((".jpg", ".png"))
        ])[:self.burst_size]

        frames = []
        for frame_file in frame_files:
            img = Image.open(os.path.join(burst_path, frame_file)).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        burst_tensor = torch.cat(frames, dim=0)  # Shape: (30, 256, 256)
        return burst_tensor


# Reverse normalization for images in [-1, 1]
unnormalize = transforms.Normalize(
    mean=[-1, -1, -1],
    std=[2, 2, 2]
)

def save_burst_images(burst_tensor, output_dir, prefix="frame"):
    """
    Saves 10 RGB frames from a burst tensor of shape [30, H, W].
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = burst_tensor.view(10, 3, 256, 256)  # Adjust (10, 3, H, W) if needed
    for idx, frame in enumerate(frames):
        img = unnormalize(frame.cpu().detach())
        save_image(img, os.path.join(output_dir, f"{prefix}_{idx:02d}.png"))


# --- Test block ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image
    import os

    # Choose dataset class
    dataset = BurstOnlyDataset(bursts_dir="/content/drive/MyDrive/dataset/bursts_dir", image_size=(256, 256))

    burst = dataset[0]
    print("Burst shape:", burst.shape)
    print("Value range:", burst.min().item(), "to", burst.max().item())

    # Reverse normalization
    unnormalize = transforms.Normalize(
        mean=[-1, -1, -1],
        std=[2, 2, 2]
    )

    frames = burst.view(10, 3, 256, 256)
    burst_images = [unnormalize(frame).clamp(0, 1) for frame in frames]

    os.makedirs("output/burst_debug", exist_ok=True)
    for idx, frame in enumerate(burst_images):
        save_image(frame, f"output/burst_debug/frame_{idx:02d}.png")
    print("🖼️ Burst frames saved to output/burst_debug/")

    # Grid view
    grid = make_grid(burst_images, nrow=10, padding=2)
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title("Sample Burst Frames")
    plt.axis("off")
    plt.show()

"""
# Instantiate and test
dataset = BurstOnlyDataset("bursts_dir", burst_size=10)
burst = dataset[0]  # Only one output now

print("Burst shape:", burst.shape)  # Expected: [10, 3, 256, 256]
print("Burst value range:", burst.min().item(), "to", burst.max().item())
"""






