import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math

# Dataset class
class NoisyCleanDefectDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, save_dir='./image_cache', save_images=False, grayscale=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.save_dir = save_dir
        self.save_images = save_images
        self.grayscale = grayscale
        self.image_triplets = []  # To store paths or cached image data

        # Check if pkl files already exist in save_dir
        if os.path.exists(self.save_dir) and len(os.listdir(self.save_dir)) > 0:
            print(f"Loading pre-saved images from {self.save_dir}")
            self.load_images_from_cache()
        else:
            print(f"Processing images from {self.root_dir} and saving to {self.save_dir}")
            self.process_and_save_images()

    def load_images_from_cache(self):
        """Load image triplets from saved .pkl files."""
        for file_name in sorted(os.listdir(self.save_dir)):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(self.save_dir, file_name)
                with open(file_path, 'rb') as f:
                    image_data = pickle.load(f)
                self.image_triplets.append((image_data['noisy'], image_data['clean'], image_data['defect']))

    def process_and_save_images(self):
        """Process and save images as .pkl files if not already cached."""
        # Create save_dir if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Iterate through each object folder in the dataset and process images
        for obj_name in os.listdir(self.root_dir):
            obj_dir = os.path.join(self.root_dir, obj_name)

            if os.path.isdir(obj_dir):
                clean_dir = os.path.join(obj_dir, self.split, 'GT_clean_image')
                noisy_dir = os.path.join(obj_dir, self.split, 'Degraded_image')
                defect_dir = os.path.join(obj_dir, self.split, 'Defect_mask')

                if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir) or not os.path.exists(defect_dir):
                    print(f"Warning: Missing directories for {obj_name} in {self.split}. Skipping.")
                    continue

                for defect_type in os.listdir(clean_dir):
                    clean_type_dir = os.path.join(clean_dir, defect_type)
                    noisy_type_dir = os.path.join(noisy_dir, defect_type)
                    defect_type_dir = os.path.join(defect_dir, defect_type)

                    if os.path.exists(noisy_type_dir) and os.path.exists(defect_type_dir):
                        for clean_img_name in os.listdir(clean_type_dir):
                            clean_img_path = os.path.join(clean_type_dir, clean_img_name)
                            noisy_img_path = os.path.join(noisy_type_dir, clean_img_name)
                            defect_img_name = f"{os.path.splitext(clean_img_name)[0]}_mask{os.path.splitext(clean_img_name)[1]}"
                            defect_img_path = os.path.join(defect_type_dir, defect_img_name)

                            if os.path.exists(noisy_img_path) and os.path.exists(defect_img_path):
                                self.image_triplets.append((noisy_img_path, clean_img_path, defect_img_path))

                                # Save to .pkl if save_images is True
                                if self.save_images:
                                    self.save_image_triplet_to_file(len(self.image_triplets) - 1)

    def save_image_triplet_to_file(self, idx):
        """Save a single image triplet as a .pkl file."""
        noisy_img_path, clean_img_path, defect_img_path = self.image_triplets[idx]
        noisy_img = np.array(Image.open(noisy_img_path).convert('L' if self.grayscale else 'RGB'))
        clean_img = np.array(Image.open(clean_img_path).convert('L' if self.grayscale else 'RGB'))
        defect_img = np.array(Image.open(defect_img_path).convert('L' if self.grayscale else 'RGB'))

        image_data = {
            'noisy': noisy_img,
            'clean': clean_img,
            'defect': defect_img
        }

        save_path = os.path.join(self.save_dir, f'image_triplet_{idx}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(image_data, f)

    def __len__(self):
        return len(self.image_triplets)

    def __getitem__(self, idx):
        noisy_img, clean_img, defect_img = self.image_triplets[idx]

        # Convert numpy arrays to PIL Images
        noisy_img = Image.fromarray(noisy_img)
        clean_img = Image.fromarray(clean_img)
        defect_img = Image.fromarray(defect_img)

        if self.grayscale:
            noisy_img = noisy_img.convert('L')
            clean_img = clean_img.convert('L')
            defect_img = defect_img.convert('L')
        else:
            noisy_img = noisy_img.convert('RGB')
            clean_img = clean_img.convert('RGB')
            defect_img = defect_img.convert('RGB')

        # Apply transformations if specified
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
            defect_img = self.transform(defect_img)

        return noisy_img, clean_img, defect_img
        

# RCAN model
class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out *= self.channel_attention(out)
        return out + residual

class RCAN(nn.Module):
    def __init__(self, num_blocks=10, channels=64, reduction=16):
        super(RCAN, self).__init__()
        self.initial_conv = nn.Conv2d(3, channels, kernel_size=7, padding=3)
        self.res_blocks = nn.Sequential(
            *[ResidualChannelAttentionBlock(channels, reduction) for _ in range(num_blocks)]
        )
        self.final_conv = nn.Conv2d(channels, 3, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return x

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0) - 10 * torch.log10(mse)
    return psnr.item()

# Set up datasets and data loaders
transform = transforms.Compose([
    transforms.Resize((900, 900)),
    transforms.ToTensor()
])

train_dataset = NoisyCleanDefectDataset(root_dir='Denoising_Dataset_train_val', split='Train', save_dir='./train_image_cache', transform=transform)
val_dataset = NoisyCleanDefectDataset(root_dir='Denoising_Dataset_train_val', split='Val', save_dir='./val_image_cache', transform=transform)

batch_size = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RCAN().to(device)
# criterion = nn.MSELoss()
def masked_loss(pred, target, defect_mask):
    loss1 = torch.mean((pred - target) ** 2)
    loss2 = torch.mean((pred - target) ** 2 * defect_mask)
    return (0.3*loss1 + 0.7*loss2)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training and validation loop
num_epochs = 300
best_psnr = 0.0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for noisy_imgs, clean_imgs, mask_imgs in train_loader:
        noisy_imgs, clean_imgs, mask_imgs = noisy_imgs.to(device), clean_imgs.to(device), mask_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        # loss = criterion(outputs, clean_imgs)
        loss = masked_loss(outputs, clean_imgs, mask_imgs)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss / len(train_loader)}")

    # Validation with PSNR calculation
    model.eval()
    total_psnr = 0
    train_psnr = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs, mask_imgs in val_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            psnr = calculate_psnr(outputs, clean_imgs)
            total_psnr += psnr

    avg_psnr = total_psnr / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation PSNR: {avg_psnr:.2f} dB")

    # Early stopping and model checkpointing
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f'rcan_best_{best_psnr}psnr.pth')
        print("Model checkpoint saved.")

    with torch.no_grad():
        for noisy_imgs, clean_imgs, mask_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            psnr_t = calculate_psnr(outputs, clean_imgs)
            train_psnr += psnr_t
    train_avg_psnr = train_psnr / len(train_loader)
    print(f"Training PSNR: {train_avg_psnr:.2f} dB")

print("Training and validation complete.")
