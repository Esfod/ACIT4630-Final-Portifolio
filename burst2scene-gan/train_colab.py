"""
Training loop for the GAN architecture
"""

from torch.cuda.amp import autocast, GradScaler

import os
from datetime import datetime
import csv
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import piq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from burst_dataset_pipeline_colab import BurstOnlyDataset
from generator import Generator
from discriminator import ConditionalDiscriminator


# making the epoch number callable in the terminal    ->   !python /content/train_colab_test.py --num_epochs=100
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
args = parser.parse_args()

num_epochs = args.num_epochs


# Dataset and loader
dataset = BurstOnlyDataset(bursts_dir='/content/drive/MyDrive/dataset/bursts_dir', image_size=(256, 256))
print(f"✅ Found {len(dataset)} bursts")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = ConditionalDiscriminator().to(device)

# Losses
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# Optimisers
optimiser_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimiser_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Label helpers
real_label = 1.0
fake_label = 0.0


# Setup TensorBoard
current_time = datetime.now().strftime("%H-%M")
run_name = f"GAN_{num_epochs}_epoch_time_{current_time}"
log_dir = f"/content/drive/MyDrive/runs/{run_name}"
writer = SummaryWriter(log_dir=log_dir)
writer.add_scalar("Loss/Generator", 0.0, 0)
writer.add_scalar("Loss/Discriminator", 0.0, 0)
writer.add_scalar("Loss/L1", 0.0, 0)
writer.add_scalar("Discriminator/Confidence", 0.0, 0)


# Directory to save models and sample outputs
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)


# Saves evaluation metrics on .csv
csv_log_path = os.path.join(log_dir, "metrics.csv")
with open(csv_log_path, mode='w', newline='') as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(["Epoch", "PSNR", "SSIM"])



# A100 gpu
scaler = GradScaler()



# Training loop
if __name__ == "__main__":

    try:
        for epoch in range(num_epochs):
            print(f"\n🟢 Starting Epoch {epoch+1}/{num_epochs}")

            generator.train()
            discriminator.train()

            running_loss_G, running_loss_D, running_L1, running_conf = 0, 0, 0, 0

            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")

            for i, burst in pbar:
                burst = burst.to(device)
                real_image = burst[:, -3:, :, :]
                input_burst = burst

                # Train Discriminator
                discriminator.zero_grad()

                with autocast():
                    real_out = discriminator(input_burst, real_image)
                    real_labels = torch.ones_like(real_out).to(device)
                    loss_D_real = criterion_GAN(real_out, real_labels)

                    fake_image = generator(input_burst)
                    fake_out = discriminator(input_burst, fake_image.detach())
                    fake_labels = torch.zeros_like(fake_out).to(device)

                    loss_D_fake = criterion_GAN(fake_out, fake_labels)
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                scaler.scale(loss_D).backward()
                scaler.step(optimiser_D)
                scaler.update()

                # Train Generator
                generator.zero_grad()

                with autocast():
                    pred_fake = discriminator(input_burst, fake_image)
                    target_labels = torch.ones_like(pred_fake).to(device)
                    loss_G_GAN = criterion_GAN(pred_fake, target_labels)
                    loss_G_L1 = criterion_L1(fake_image, real_image)
                    loss_G = loss_G_GAN + 100 * loss_G_L1

                scaler.scale(loss_G).backward()
                scaler.step(optimiser_G)
                scaler.update()

                # Confidence score
                with torch.no_grad():
                    confidence_score = torch.sigmoid(pred_fake).mean().item()

                running_loss_G += loss_G.item()
                running_loss_D += loss_D.item()
                running_L1 += loss_G_L1.item()
                running_conf += confidence_score

                # Update progress bar with live stats
                pbar.set_postfix({
                    "G_loss": loss_G.item(),
                    "D_loss": loss_D.item(),
                    "L1": loss_G_L1.item(),
                    "Conf": confidence_score
                })

            # Logging per epoch
            avg_G = running_loss_G / len(dataloader)
            avg_D = running_loss_D / len(dataloader)
            avg_L1 = running_L1 / len(dataloader)
            avg_conf = running_conf / len(dataloader)

            writer.add_scalar("Loss/Generator", avg_G, epoch)
            writer.add_scalar("Loss/Discriminator", avg_D, epoch)
            writer.add_scalar("Loss/L1", avg_L1, epoch)
            writer.add_scalar("Discriminator/Confidence", avg_conf, epoch)

            if (epoch + 1) % 500 == 0 or (epoch + 1) == num_epochs:
                from validate_generator_colab import validate_generator
                from validation_dataset_colab import BurstDataset
                from torch.utils.data import DataLoader

                val_dataset = BurstDataset(bursts_dir="/content/drive/MyDrive/dataset/burst_validation_im", burst_size=10)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

                psnr, ssim = validate_generator(
                    generator,
                    val_loader,
                    device,
                    output_dir=f"/content/drive/MyDrive/validation_outputs/epoch_{epoch + 1}"
                )

                writer.add_scalar("Validation/PSNR", psnr, epoch)
                writer.add_scalar("Validation/SSIM", ssim, epoch)

                with open(csv_log_path, mode='a', newline='') as file:
                    writer_csv = csv.writer(file)
                    writer_csv.writerow([epoch + 1, psnr, ssim])




    except KeyboardInterrupt:
        print("Training interrupted. Saving model.")

    finally:
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "generator_last.pth"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "discriminator_last.pth"))
        print("💾 Last model checkpoint saved.")


