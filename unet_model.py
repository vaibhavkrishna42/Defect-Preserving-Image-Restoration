import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import math

# Dataset class
class NoisyCleanDefectDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, save_dir='./image_cache', save_images=False, grayscale=False):
        self.root_dir = root_dir
        self.split = split  # 'Train' or 'Val'
        self.transform = transform
        self.save_dir = save_dir
        self.save_images = save_images
        self.grayscale = grayscale

        self.image_triplets = []  # To store (noisy_image_path, clean_image_path, defect_mask_path)

        # Iterate through each object folder in the dataset
        for obj_name in os.listdir(self.root_dir):
            obj_dir = os.path.join(self.root_dir, obj_name)

            # Ensure the object directory is valid and is a directory
            if os.path.isdir(obj_dir):
                clean_dir = os.path.join(obj_dir, self.split, 'GT_clean_image')
                noisy_dir = os.path.join(obj_dir, self.split, 'Degraded_image')
                defect_dir = os.path.join(obj_dir, self.split, 'Defect_mask')

                # Ensure the clean, noisy, and defect directories exist
                if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir) or not os.path.exists(defect_dir):
                    print(f"Warning: Directories do not exist for {obj_name} in {self.split}. Skipping.")
                    continue

                # Iterate through each defect type folder
                for defect_type in os.listdir(clean_dir):
                    clean_type_dir = os.path.join(clean_dir, defect_type)
                    noisy_type_dir = os.path.join(noisy_dir, defect_type)
                    defect_type_dir = os.path.join(defect_dir, defect_type)

                    # Ensure that the defect folder exists
                    if os.path.exists(noisy_type_dir) and os.path.exists(defect_type_dir):
                        # Iterate through the clean images
                        for clean_img_name in os.listdir(clean_type_dir):
                            clean_img_path = os.path.join(clean_type_dir, clean_img_name)
                            noisy_img_path = os.path.join(noisy_type_dir, clean_img_name)
                            base_name, ext = os.path.splitext(clean_img_name)  # Split into base name and extension
                            defect_img_name = f"{base_name}_mask{ext}"  # Add _mask before the extension
                            defect_img_path = os.path.join(defect_type_dir, defect_img_name)

                            # Check if corresponding noisy and defect mask exist
                            if os.path.exists(noisy_img_path) and os.path.exists(defect_img_path):
                                self.image_triplets.append((noisy_img_path, clean_img_path, defect_img_path))

        # Save image matrices if required
        if save_images:
            self.save_images_to_file()

    def __len__(self):
        return len(self.image_triplets)

    def __getitem__(self, idx):
        noisy_img_path, clean_img_path, defect_img_path = self.image_triplets[idx]

        # Load images
        noisy_img = Image.open(noisy_img_path)
        clean_img = Image.open(clean_img_path)
        defect_img = Image.open(defect_img_path)

        # Convert to grayscale if specified
        if self.grayscale:
            noisy_img = noisy_img.convert('L')
            clean_img = clean_img.convert('L')
            defect_img = defect_img.convert('L')
        else:
            noisy_img = noisy_img.convert('RGB')
            clean_img = clean_img.convert('RGB')
            defect_img = defect_img.convert('RGB')

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
            defect_img = self.transform(defect_img)

        return noisy_img, clean_img, defect_img

    def save_images_to_file(self):
        """Save image matrices as numpy arrays to a file."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for idx, (noisy_img_path, clean_img_path, defect_img_path) in enumerate(self.image_triplets):
            noisy_img = np.array(Image.open(noisy_img_path).convert('L' if self.grayscale else 'RGB'))
            clean_img = np.array(Image.open(clean_img_path).convert('L' if self.grayscale else 'RGB'))
            defect_img = np.array(Image.open(defect_img_path).convert('L' if self.grayscale else 'RGB'))

            # Store the image data in a dictionary
            image_data = {
                'noisy': noisy_img,
                'clean': clean_img,
                'defect': defect_img
            }

            # Save the dictionary as a pickle file for each triplet
            save_path = os.path.join(self.save_dir, f'image_triplet_{idx}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(image_data, f)

    @staticmethod
    def load_image_triplet_from_file(file_path):
        """Load image triplets (noisy, clean, defect) from a pickle file."""
        with open(file_path, 'rb') as f:
            image_data = pickle.load(f)
        return image_data['noisy'], image_data['clean'], image_data['defect']
    
# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0) - 10 * torch.log10(mse)
    return psnr.item()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((900, 900)),
    transforms.ToTensor()
])

# Instantiate dataset
train_dataset = NoisyCleanDefectDataset(
    root_dir='Denoising_Dataset_train_val',
    split='Train',
    transform=transform,
    save_dir='./train_image_cache',
    save_images=True,  # Set to True to save images to file
    grayscale=False     # Set to False for RGB images as U-Net handles 3 channels
)

val_dataset = NoisyCleanDefectDataset(
    root_dir='Denoising_Dataset_train_val',
    split='Val',
    transform=transform,
    save_dir='./val_image_cache',
    save_images=True,
    grayscale=False
)

# Print the number of images before dividing into batches
print(f"Number of images in Training Set: {len(train_dataset)}")
print(f"Number of images in Validation Set: {len(val_dataset)}")

# Now, based on the number of images, you can decide on batch size
batch_size = 8  # You can adjust this based on your dataset size

# Use DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1)  # 3 channels for RGB output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Custom Loss Function
def masked_loss(pred, target, defect_mask):
    loss1 = torch.mean((pred - target) ** 2)
    loss2 = torch.mean((pred - target) ** 2 * defect_mask)
    return (0.3*loss1 + 0.7*loss2)

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training and validation loop
num_epochs = 50
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
