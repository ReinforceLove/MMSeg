import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from nibabel.orientations import axcodes2ornt, io_orientation, inv_ornt_aff


def load_nifti_image(path):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    return img, data, affine


def save_nifti_image(data, path, affine):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


def reorient_to_ras(data, affine):
    ornt = axcodes2ornt(('R', 'A', 'S'))
    current_ornt = io_orientation(affine)
    transform = nib.orientations.ornt_transform(current_ornt, ornt)
    ras_data = nib.orientations.apply_orientation(data, transform)
    ras_affine = affine.dot(inv_ornt_aff(transform, data.shape))
    return ras_data, ras_affine


def crop_non_black(image):
    coords = np.array(np.nonzero(image))
    z_min, y_min, x_min = coords.min(axis=1)
    z_max, y_max, x_max = coords.max(axis=1) + 1  # Add 1 since slice is exclusive at the end
    cropped_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    return cropped_image, (z_min, z_max, y_min, y_max, x_min, x_max)


def normalize_intensity(image, window=(-300, 300)):
    min_val, max_val = window
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    return image


def normalize_to_01(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


def scale_to_neg1_pos1(image):
    image = 2 * image - 1
    return image


def resample_image_2d(image, target_shape=(256, 256)):
    image = image.copy()
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    image = F.interpolate(image, size=target_shape, mode='bilinear', align_corners=False)
    image = image.squeeze().numpy()
    return image

def resample_label_2d(label, target_shape=(256, 256)):
    label = label.copy()
    label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)
    label = F.interpolate(label, size=target_shape, mode='nearest')
    label = label.squeeze().numpy()
    return label


def slice_and_save_nifti_z_axis(image_path, label_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img, image, img_affine = load_nifti_image(image_path)
    _, label, lbl_affine = load_nifti_image(label_path)
    image, img_affine = reorient_to_ras(image, img_affine)
    label, lbl_affine = reorient_to_ras(label, lbl_affine)
    image, bounds = crop_non_black(image)
    label = label[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]]
    image = normalize_intensity(image)
    image = normalize_to_01(image)
    image = scale_to_neg1_pos1(image)
    img_id = os.path.basename(image_path).split('.')[0]
    num_slices = image.shape[2]  # z-axis slices
    for slice_idx in range(num_slices):
        slice_img = image[:, :, slice_idx]  # z-axis slice
        slice_lbl = label[:, :, slice_idx]  # z-axis slice
        slice_img = resample_image_2d(slice_img, target_shape=(256, 256))
        np.save(os.path.join(save_dir, f'{img_id}_slice_{slice_idx}_image.npy'), slice_img)
        np.save(os.path.join(save_dir, f'{img_id}_slice_{slice_idx}_label.npy'), slice_lbl)


# Example usage:
image_path = r''
label_path = r''
save_dir = r''
slice_and_save_nifti_z_axis(image_path, label_path, save_dir)
