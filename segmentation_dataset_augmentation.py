"""
segmentation_dataset_augmentation.py
===================================

A production-ready image augmentation pipeline designed specifically for
semantic / instance segmentation datasets.

This script applies deterministic, segmentation-safe augmentations to
image–mask pairs using Albumentations, and saves the results in a clean,
structured directory layout.

Key Features
------------
- Image and mask are always transformed together (segmentation-safe)
- Supports photometric and geometric augmentations
- Automatically infers dataset size
- Clean progress reporting
- Ready for YOLO / COCO-style segmentation workflows

Expected Folder Structure
-------------------------
X/
├── image_1.jpg
├── image_2.jpg
└── image_N.jpg

y/
├── mask_1.png
├── mask_2.png
└── mask_N.png

Output Structure
----------------
augmented_x/
├── RandomBrightnessContrast/
├── Defocus/
├── GlassBlur/
├── ISONoise/
└── ShiftScaleRotate/

augmented_y/
├── RandomBrightnessContrast/
├── Defocus/
├── GlassBlur/
├── ISONoise/
└── ShiftScaleRotate/
"""

# =================================================
# IMPORTS
# =================================================
import os
import cv2
import warnings
import albumentations as A

# =================================================
# GLOBAL CONFIGURATION
# =================================================

# Silence Albumentations warnings (e.g. deprecated arguments)
warnings.filterwarnings("ignore", category=UserWarning)

# Input directories
IMAGE_DIR = "X"
MASK_DIR = "y"

# Output root directories
OUT_IMAGE_ROOT = "augmented_x"
OUT_MASK_ROOT = "augmented_y"

# Supported image extension
IMAGE_EXTENSION = ".jpg"


# =================================================
# AUGMENTATION DEFINITIONS
# =================================================
"""
All transformations are explicitly segmentation-safe.
Geometric transforms are applied to both image and mask
using nearest-neighbor interpolation for masks.
"""

TRANSFORMATIONS = [

    # -----------------------------
    # Photometric Augmentations
    # -----------------------------
    A.RandomBrightnessContrast(
        brightness_limit=0.35,
        contrast_limit=0.35,
        brightness_by_max=True,
        p=1.0
    ),

    A.Defocus(
        radius=(1, 2),
        alias_blur=(0.1, 0.15),
        p=1.0
    ),

    A.GlassBlur(
        sigma=0.1,
        max_delta=1,
        iterations=1,
        mode="fast",
        p=1.0
    ),

    A.ISONoise(
        color_shift=(0.02, 0.03),
        intensity=(0.2, 0.3),
        p=1.0
    ),

    # -----------------------------
    # Geometric Augmentation
    # -----------------------------
    A.ShiftScaleRotate(
        shift_limit=0.02,          # translation
        scale_limit=0.2,           # scaling
        rotate_limit=5,            # rotation (degrees)
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=1.0
    )
]


# =================================================
# DATASET STATISTICS
# =================================================
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSION)
]

num_images = len(image_files)
num_augs = len(TRANSFORMATIONS)
total_outputs = num_images * num_augs

print(f"Detected images          : {num_images}")
print(f"Augmentations per image  : {num_augs}")
print(f"Total outputs to be saved: {total_outputs}")
print("-" * 50)


# =================================================
# AUGMENTATION PIPELINE
# =================================================
counter = 1

for image_name in image_files:

    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARNING] Failed to read image: {image_name}")
        continue

    # Expected naming convention:
    # image_1.jpg  ->  mask_1.png
    base_id = image_name.split("_")[1].split(".")[0]
    mask_name = f"mask_{base_id}.png"
    mask_path = os.path.join(MASK_DIR, mask_name)

    if not os.path.exists(mask_path):
        print(f"[WARNING] Mask not found: {mask_path}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for transform in TRANSFORMATIONS:

        transform_name = transform.__class__.__name__

        # Create output directories
        out_img_dir = os.path.join(OUT_IMAGE_ROOT, transform_name)
        out_mask_dir = os.path.join(OUT_MASK_ROOT, transform_name)

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # Apply augmentation
        augmented = transform(image=image, mask=mask)

        # Output file paths
        out_image_path = os.path.join(
            out_img_dir,
            f"{image_name[:-4]}-{transform_name}.jpg"
        )

        out_mask_path = os.path.join(
            out_mask_dir,
            f"mask_{base_id}-{transform_name}.png"
        )

        # Save augmented results
        cv2.imwrite(out_image_path, augmented["image"])
        cv2.imwrite(out_mask_path, augmented["mask"])

        print(f"[{counter}/{total_outputs}] Saved -> {transform_name}")
        counter += 1


print("\n✅ Dataset augmentation completed successfully.")
