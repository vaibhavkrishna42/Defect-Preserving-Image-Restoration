import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
from PIL import Image
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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

# Set up datasets and data loaders
transform = transforms.Compose([
    transforms.Resize((900, 900)),
    transforms.ToTensor()
])

test_dataset = NoisyCleanDefectDataset(
    root_dir='Denoising_Dataset_test', # Replace with testing images folder
    split='Test',
    transform=transform,
    save_dir='./test_image_cache',
    save_images=True,
    grayscale=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0) - 10 * torch.log10(mse)
    return psnr.item()

# SSIM Calculation
ssim = StructuralSimilarityIndexMeasure().to(device)

def validate(model, dataloader, l_counts, l_names):
    model.eval()

    object_psnr = {name: 0.0 for name in l_names}
    object_ssim = {name: 0.0 for name in l_names}
    object_counts = {name: 0 for name in l_names}

    # Track object and image indices
    obj_idx = 0
    img_count_in_obj = 0

    with torch.no_grad():
        for images, clean_images, _ in dataloader:
            images = images.to(device)
            clean_images = clean_images.to(device)

            # Run model on the image
            outputs = model(images)

            # Compute PSNR
            image_psnr = calculate_psnr(outputs, clean_images)
            object_name = l_names[obj_idx]
            object_psnr[object_name] += image_psnr

            # Calculate SSIM
            image_ssim = ssim(outputs, clean_images)
            object_ssim[object_name] += image_ssim

            # Update counts
            object_counts[object_name] += 1
            img_count_in_obj += 1

            # Move to the next object if all images are processed
            if img_count_in_obj >= l_counts[obj_idx]:
                obj_idx += 1
                img_count_in_obj = 0
                if obj_idx >= len(l_counts):
                    break  # Stop if all objects are processed

    # Calculate and print average PSNR and SSIM for each object
    avg_psnr_ssim_per_object = {name: {'PSNR': object_psnr[name] / object_counts[name],
                                       'SSIM': object_ssim[name] / object_counts[name]}
                                for name in l_names}

    for obj_name, metrics in avg_psnr_ssim_per_object.items():
        print(f"{obj_name}: Average PSNR = {metrics['PSNR']:.2f}, Average SSIM = {metrics['SSIM']:.4f}")

    return avg_psnr_ssim_per_object

# Loading the model state_dict
model = RCAN().to(device)
model.load_state_dict(torch.load('rcan_best_state_dict.pth', weights_only=True, map_location=device))
model.eval()

l_count = [45, 68, 85, 69, 42, 53, 72, 53, 103, 95, 64, 22, 28, 42, 91]
l_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

store_dict = validate(model, test_loader, l_count, l_names)

def plot_outputparams(l_names, dict,model_name, key):
    vals = [dict[name][key] for name in l_names]
    # plt.bar(l_names, PSNR)

    fig, ax = plt.subplots(figsize =(16, 9))

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)

    ax.invert_yaxis()


    for index, value in enumerate(vals):
        if torch.is_tensor(value):
            value = value.cpu().numpy()

        value = np.round(value, 2) if isinstance(value, np.ndarray) else round(value, 2)

        if key == "PSNR":
            plt.text(value + 0.05, index, str(value), ha='left', va='center')
        else:
            plt.text(value+0.03, index, str(value), ha='right', va='center')

    ax.set_title(f"Object-wise {key} values of {model_name}")
    ax.barh(l_names, vals)
    ax.set_xlabel("PSNR")
    ax.set_ylabel("Object")

    plt.show()

plot_outputparams(l_names, store_dict, "RCAN", "SSIM")