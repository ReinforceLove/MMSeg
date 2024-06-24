import os
import random
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from scipy.ndimage import zoom




def get_dataloader():
    class ISICDataset(Dataset):
        def __init__(self, file_list, root_dir, transform=None, noise_transform=None, target_shape=(512, 512)):
            self.root_dir = root_dir
            self.transform = transform
            self.noise_transform = noise_transform
            self.images = file_list
            self.target_shape = target_shape


        def __len__(self):
            return len(self.images)

        def resample(self, image, target_shape):
            factor = [t / s for t, s in zip(target_shape, image.shape)]
            return zoom(image, factor, order=3)

        def contrastive_transform(self, exclude_transform=None):
            rotation_angle = random.choice([90, 180, 270])
            transforms_list = [
                transforms.RandomVerticalFlip(p=1.0),
                transforms.RandomRotation(rotation_angle),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=1.0, interpolation=3),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            ]

            # 如果提供了 exclude_transform，则从列表中移除
            if exclude_transform:
                transforms_list = [t for t in transforms_list if type(t) != type(exclude_transform)]

            # 随机选择一个变换
            random_transform = random.choice(transforms_list)
            return random_transform

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            # img_name = os.path.join(self.root_dir, self.images[idx])
            img_name = os.path.join(self.root_dir, self.images[idx])
            image_array = np.load(img_name)
            image_array = self.resample(image_array, self.target_shape)
            original_image = Image.fromarray(image_array.astype(np.uint8),mode="L")  # Specify mode='L' for grayscale

            # original_image = Image.open(img_name)
            # Convert the image to numpy array and perform B-spline resampling
            # np_image = np.array(original_image)
            # np_image = skimage_transform.rescale(np_image, 1, mode='reflect', multichannel=True, anti_aliasing=True)
            # np_image = skimage_transform.resize(np_image, (224, 224), mode='reflect', anti_aliasing=True)
            # original_image = Image.fromarray((np_image * 255).astype(np.uint8))
            # original_image = remove_black_borders(original_image)


            # if original_image.mode != 'RGB':
            #     original_image = original_image.convert('RGB')

            # Check if the image has any missing or corrupt information
            if np.any(np.isnan(np.array(original_image))):
                return None



            # Create a copy of the image to apply transformations
            rotated_image = original_image.copy()
            flipped_image = original_image.copy()
            local_rotated_image = original_image.copy()

            contrast_image= original_image.copy()
            mask_image= original_image.copy()




            # Task 1: Perform a random rotation
            rotation_choice = random.randint(1, 3)
            rotated_image = rotated_image.rotate(rotation_choice * 90)

            # Task 2: Perform a random vertical flip
            flip_choice = random.choice([0, 1])
            if flip_choice == 1:
                flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)

            # Task 3: Perform a local rotation
            # Step 1: Choose a square patch from the center of the image
            width, height = local_rotated_image.size
            center_x = width // 2
            center_y = height // 2
            patch_size = max(width, height) // 2
            start_x = center_x - patch_size // 2
            start_y = center_y - patch_size // 2
            patch = local_rotated_image.crop((start_x, start_y, start_x + patch_size, start_y + patch_size))
            patch_rotation_choice = random.randint(1, 3)
            rotated_patch = patch.rotate(patch_rotation_choice * 80)
            local_rotated_image.paste(rotated_patch, (start_x, start_y))


            # Task 4: Image reconstruction
            # Convert to float32
            noisy_image = original_image.copy()
            np_image = np.array(noisy_image)

            # Check if the image is grayscale or RGB and convert to float
            if len(np_image.shape) == 2:
                # Grayscale image
                np_image = np_image.astype(np.float32) / 255.0
            else:
                # RGB image
                np_image = np_image.astype(np.float32) / 255.0

            # Generate noise
            noise = np.random.normal(0, 0.3, np_image.shape)

            # Add noise to the image0
            noisy_image = np_image + noise

            # Clip values to between 0 and 1
            noisy_image = np.clip(noisy_image, 0.0, 1.0)

            # Convert back to uint8
            noisy_image = (noisy_image * 255).astype(np.uint8)

            # Convert back to PIL image
            noisy_image = Image.fromarray(noisy_image)



            # Task 5: Contrastive Learning
            contrast_image1_transform = self.contrastive_transform()
            contrast_image1 = contrast_image1_transform(contrast_image)
            contrast_image2_transform = self.contrastive_transform(exclude_transform=contrast_image1_transform)
            contrast_image2 = contrast_image2_transform(contrast_image1)

            # Task 6: Add a mask and predict the content of the masked area

            # Add a mask to the image
            masked_image = add_mask(mask_image, start_x=200, start_y=200, width=150, height=150)

            # Apply transformations
            if self.transform:
                original_image = self.transform(original_image)
                rotated_image = self.transform(rotated_image)
                flipped_image = self.transform(flipped_image)
                local_rotated_image = self.transform(local_rotated_image)
                noisy_image = self.noise_transform(noisy_image)
                contrast_image1 = self.transform(contrast_image1)
                contrast_image2 = self.transform(contrast_image2)
                masked_image=self.transform(masked_image)

            return (original_image, rotated_image, flipped_image, local_rotated_image, noisy_image, contrast_image1,
                    contrast_image2,masked_image, (rotation_choice, flip_choice, patch_rotation_choice))


    root_dir = r""
    # all_images = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
    all_images = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
    random.shuffle(all_images)

    split_point = int(0.8 * len(all_images))  # Change this to the appropriate ratio
    train_files = all_images[:split_point]
    val_files = all_images[split_point:]

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ISICDataset(train_files, root_dir=root_dir, transform=transformations,
                                noise_transform=transformations)
    val_dataset = ISICDataset(val_files, root_dir=root_dir, transform=transformations, noise_transform=transformations)

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_dataloader, val_dataloader


def enhanced_random_transforms(image, s=1):
    # Existing transformations

    rotation_angle = random.choice([90, 180, 270])
    transforms_list = [
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(rotation_angle),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
        transforms.RandomPerspective(distortion_scale=0.2, p=1, interpolation=3),  # Random perspective transformation
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Gaussian Blur with varying sigma
    ]

    # Randomly select only 1 transform
    random_transform = random.choice(transforms_list)
    return random_transform(image)


def add_mask(image, start_x, start_y, width, height):
    masked_image = image.copy()
    draw = ImageDraw.Draw(masked_image)
    draw.rectangle([start_x, start_y, start_x + width, start_y + height], fill=0) # You can set the fill color as per your requirement
    return masked_image

def remove_black_borders(image, tolerance=0):
    # Convert the image to a numpy array
    img = np.array(image)

    # Define a mask of the non-black pixels. If the image has more than one channel,
    # we consider a pixel to be non-black if any of its channels has a value > tolerance
    mask = np.any(img > tolerance, axis=-1)

    # Get the min/max x and y coordinates of non-black pixels.
    coords = np.argwhere(mask)

    x_start, y_start = coords.min(axis=0)
    x_end, y_end = coords.max(axis=0) + 1

    # Crop the image to these min/max x and y coordinates.
    cropped_img = img[x_start:x_end, y_start:y_end]

    return Image.fromarray(cropped_img.astype('uint8'))  # Convert back to uint8

