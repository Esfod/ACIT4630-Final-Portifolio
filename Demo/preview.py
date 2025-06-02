
import torch
import os
from torchvision.utils import save_image, make_grid
from validation_dataset_colab import BurstDataset
from generator import Generator
from discriminator import ConditionalDiscriminator
import piq
from PIL import Image, ImageDraw, ImageFont

# --- Config ---
checkpoint_path = "runs/PatchGAN_80_epoch_time_03-53/checkpoints/generator_last.pth"
burst_dir = "burst_validation_high_variation"
preview_name = "preview_PatchGAN_50+30_epochs_high_var"
base_preview_dir = os.path.join("model_preview_dir", preview_name)
os.makedirs(base_preview_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load models ---
generator = Generator().to(device)
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

discriminator = ConditionalDiscriminator().to(device)
discriminator.eval()

# --- Load dataset ---
dataset = BurstDataset(bursts_dir=burst_dir, burst_size=10)



unnorm = lambda x: (x + 1) / 2

def annotate_image(image_path, text):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    width, height = image.size
    draw.rectangle([(0, height - text_height - 10), (width, height)], fill=(0, 0, 0))
    draw.text((10, height - text_height - 5), text, fill="white", font=font)
    image.save(image_path)


for i in range(len(dataset)):
    burst, target = dataset[i]
    burst = burst.to(device)
    target = target.to(device)
    burst_input = burst.view(1, -1, burst.size(2), burst.size(3))

    with torch.no_grad():
        output = generator(burst_input)
        confidence = torch.sigmoid(discriminator(burst_input, output)).mean().item()

    output = unnorm(output.squeeze(0))
    target = unnorm(target)
    burst_frames = [unnorm(frame) for frame in burst]

    # --- Save images ---
    sample_dir = os.path.join(base_preview_dir, f"sample_{i}")
    os.makedirs(sample_dir, exist_ok=True)

    output_path = os.path.join(sample_dir, "generated_output.png")
    save_image(output, output_path)
    save_image(target, os.path.join(sample_dir, "target_frame.png"))
    save_image(make_grid(burst_frames, nrow=5), os.path.join(sample_dir, "burst_grid.png"))

    # --- Compute metrics ---
    output_u = output.unsqueeze(0)
    target_u = target.unsqueeze(0)
    psnr_score = piq.psnr(output_u, target_u, data_range=1.0).item()
    ssim_score = piq.ssim(output_u, target_u, data_range=1.0).item()

    # Annotate generated image
    label = f"PSNR = {psnr_score:.2f}, SSIM = {ssim_score:.4f}, Conf = {confidence:.4f}"
    # Save annotated image separately
    annotated_path = os.path.join(sample_dir, "generated_output_annotated.png")
    save_image(output, annotated_path)
    annotate_image(annotated_path, label)

    print(f"🖼 Sample {i}: {label}")
    with open(os.path.join(sample_dir, "metrics.txt"), "w") as f:
        f.write(f"{label}\n")


# Save all printed sample logs to info.txt
info_log_path = os.path.join(base_preview_dir, "info.txt")

with open(info_log_path, "w") as log_file:
    for i in range(len(dataset)):
        sample_metrics_path = os.path.join(base_preview_dir, f"sample_{i}", "metrics.txt")
        if os.path.exists(sample_metrics_path):
            with open(sample_metrics_path, "r") as sample_file:
                line = sample_file.readline().strip()
                log_file.write(f"Sample {i}: {line}\n")

print(f"✅ Previews saved to: {base_preview_dir}")


