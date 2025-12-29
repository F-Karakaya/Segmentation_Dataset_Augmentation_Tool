# Segmentation Dataset Augmentation Tool
### Segmentation-Safe Image & Mask Augmentation Pipeline (Albumentations-Based)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Albumentations](https://img.shields.io/badge/Library-Albumentations-orange.svg)
![Segmentation](https://img.shields.io/badge/Task-Semantic%20Segmentation-green.svg)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-brightgreen.svg)

---

## 📌 Overview

**Segmentation Dataset Augmentation Tool** is a clean, production-oriented utility designed to perform  
**segmentation-safe data augmentation** on paired **image–mask datasets**.

Unlike naive augmentation approaches that modify images only, this tool guarantees that:

> **Every transformation applied to an image is applied identically to its corresponding mask.**

This is critical for:
- Semantic segmentation
- Instance segmentation
- Industrial inspection pipelines
- Medical and scientific imaging
- Any pixel-aligned learning task

The tool is intentionally designed to be:
- Explicit and readable
- Safe for real-world datasets
- Easy to extend and audit
- Suitable for both research and production use

---

## 🎯 Why This Tool Exists

Data augmentation is essential for improving segmentation robustness, but:

> Applying geometric transformations to images **without synchronizing masks corrupts labels**.

This tool solves that problem by:
- Using **Albumentations** with image–mask pairing
- Applying photometric and geometric transforms safely
- Preserving mask integrity under rotation, translation, and scaling
- Producing a transparent and inspectable output structure

---

## 📦 Dependencies & Installation

This project relies on a minimal and widely adopted stack.

### Required Python Packages

Install dependencies via pip:

```bash
pip install albumentations opencv-python numpy
```

> **Albumentations** is the core library responsible for transformation correctness and performance.

---

## 📂 Project Structure

Below is the complete directory layout of the  
**Segmentation_Dataset_Augmentation_Tool** module:

```text
Segmentation_Dataset_Augmentation_Tool/
├── X/                          (Original input images)
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── image_3.jpg
│
├── y/                          (Original segmentation masks)
│   ├── mask_1.png
│   ├── mask_2.png
│   └── mask_3.png
│
├── augmented_X/                (Augmented images, grouped by transform)
│   ├── RandomBrightnessContrast/
│   ├── Defocus/
│   ├── GlassBlur/
│   ├── ISONoise/
│   └── ShiftScaleRotate/
│
├── augmented_y/                (Augmented masks aligned with augmented images)
│   ├── RandomBrightnessContrast/
│   ├── Defocus/
│   ├── GlassBlur/
│   ├── ISONoise/
│   └── ShiftScaleRotate/
│
├── segmentation_dataset_augmentation.py
│                               (Main augmentation pipeline)
│
└── README.md                   (This documentation)
```

Each augmentation type is stored in its **own subfolder**, ensuring:
- Clear traceability
- Easy ablation studies
- Fast debugging and validation

---

## 🧠 Expected Naming Convention

The tool assumes a simple and explicit naming rule:

```text
image_1.jpg  →  mask_1.png
image_2.jpg  →  mask_2.png
...
```

This makes dataset integrity easy to validate and avoids ambiguity.

---

## ⚙️ Augmentation Pipeline Explained

The core logic resides in:

```bash
segmentation_dataset_augmentation.py
```

### 🔹 Applied Transformations

The pipeline applies the following **segmentation-safe** augmentations:

- **RandomBrightnessContrast**  
  Simulates illumination variability without altering geometry.

- **Defocus**  
  Models optical defocus and mild blur effects.

- **GlassBlur**  
  Introduces strong local blur and distortion patterns.

- **ISONoise**  
  Simulates sensor noise commonly seen in real cameras.

- **ShiftScaleRotate**  
  Performs **translation, scaling, and rotation** while keeping
  image–mask alignment intact.

All geometric operations:
- Apply the same affine matrix to image and mask
- Use nearest-neighbor logic for masks
- Fill borders with background-safe values

---

## ▶️ How to Run

From inside the `Segmentation_Dataset_Augmentation_Tool` directory:

```bash
python segmentation_dataset_augmentation.py
```

The script will:
1. Scan the `X/` directory for images
2. Match each image with its corresponding mask in `y/`
3. Apply all defined augmentations
4. Save results into `augmented_X/` and `augmented_y/`
5. Print a clean progress log to the console

---

## 🖼️ Visual Examples (Original vs Augmented)

Visual inspection is **strongly recommended** for segmentation datasets.

Below are representative examples (not exhaustive).

---

### 🔹 ShiftScaleRotate Example (image_1)

**Original Image**
  
![Original Image](X/image_2.jpg)

**Augmented Image**
  
![Augmented Image](augmented_X/ShiftScaleRotate/image_2-ShiftScaleRotate.jpg)

**Augmented Mask**
  
![Augmented Mask](augmented_y/ShiftScaleRotate/mask_2-ShiftScaleRotate.png)

This example demonstrates:
- Correct rotation
- Proper spatial translation
- Perfect mask alignment

---

### 🔹 RandomBrightnessContrast Example (image_3)

**Original Image**
  
![Original Image](X/image_3.jpg)

**Augmented Image**
  
![Augmented Image](augmented_X/RandomBrightnessContrast/image_3-RandomBrightnessContrast.jpg)

**Corresponding Mask (unchanged geometry)**
  
![Augmented Mask](augmented_y/RandomBrightnessContrast/mask_3-RandomBrightnessContrast.png)

This example shows:
- Photometric variation
- Zero geometric distortion
- Mask integrity preservation

---

## 🧪 Typical Use Cases

This tool is well-suited for:

- Semantic segmentation dataset expansion
- Industrial inspection pipelines
- Medical image segmentation
- Small dataset amplification
- Robustness testing
- Pre-training data preparation
- Academic research experiments

---

## 🧩 Design Philosophy

- Dataset integrity over blind augmentation
- Explicit over implicit logic
- Visual verification encouraged
- Minimal dependencies
- Clean folder semantics
- Easy extensibility

---

## 👤 Author

**Furkan Karakaya**  
AI & Computer Vision Engineer  
📧 se.furkankarakaya@gmail.com  

---

⭐ If this tool improves your segmentation pipeline, feel free to star the repository or adapt it to your own workflows.
