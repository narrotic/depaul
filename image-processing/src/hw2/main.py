# src/hw2/main.py
# CSC 380/481 - Homework 2: Image Transformations
# This program demonstrates various image processing techniques including
# interpolation, gray level transforms, histogram equalization, and pixel replication.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- Utilities --------------------

def load_grayscale(path):
    """Load image and convert to grayscale (used in all parts)"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def show_images(images, titles, cols=4, cmap='gray'):
    """Utility function to display multiple images in a grid layout"""
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# -------------------- Part 1: Image Scaling by Interpolation --------------------
# Requirements:
# a) Convert image to grayscale ✓ (done in load_grayscale)
# b) Pick reduction between 5% (1/20) and 25% (5/20) ✓ (scale=0.2 = 20%)
# c) Nearest neighbor interpolation - down-sample and up-sample ✓
# d) Bilinear interpolation - down-sample and up-sample ✓
# e) Bicubic interpolation - down-sample and up-sample ✓
# f) Display 7 images: original + 3 reduced + 3 restored ✓

def interpolation_demo(img, scale=0.2):
    """
    Part 1: Demonstrates image scaling using different interpolation methods.

    Args:
        img: Grayscale input image
        scale: Reduction factor (0.2 = 20% = 1/5, within required 5%-25% range)

    Displays:
        - Original image
        - 3 down-sampled images (nearest, bilinear, bicubic)
        - 3 up-sampled/restored images (nearest, bilinear, bicubic)
    """
    h, w = img.shape
    new_size = (int(w * scale), int(h * scale))

    results = []
    titles = []

    # Original image
    results.append(img)
    titles.append("Original")

    # Define interpolation methods
    methods = {
        "Nearest": cv2.INTER_NEAREST,    # c) Nearest neighbor
        "Bilinear": cv2.INTER_LINEAR,    # d) Bilinear
        "Bicubic": cv2.INTER_CUBIC,      # e) Bicubic
    }

    # For each method: down-sample then up-sample
    for name, method in methods.items():
        down = cv2.resize(img, new_size, interpolation=method)  # Down-sample
        up = cv2.resize(down, (w, h), interpolation=method)     # Up-sample to original size
        results.extend([down, up])
        titles.extend([f"{name} Down", f"{name} Up"])

    # Display all 7 images (f)
    show_images(results, titles, cols=3)


# -------------------- Part 2: Basic Gray Level Transformations --------------------
# Requirements:
# a) Load image and make it grayscale ✓ (done in load_grayscale)
# b) Convert to floating point values ✓ (astype(np.float32))
# c) Convert range from 0-255 to 0.0-1.0 ✓ (divide by 255.0)
# d) Create inverted image: New = 1.0 - Old ✓
# e) Use power function for contrast adjustment ✓
#    - 2 values between 0.25-0.99 ✓ (0.4, 0.8 - decrease contrast)
#    - 2 values between 1.1-4.0 ✓ (1.5, 3.0 - increase contrast)
# Display: original + inverted + 2 decreased contrast + 2 increased contrast ✓

def gray_level_transforms(img):
    """
    Part 2: Demonstrates basic gray level transformations.

    Args:
        img: Grayscale input image (0-255 range)

    Displays:
        - Original image
        - Inverted image (1.0 - intensity)
        - 2 images with decreased contrast (gamma < 1.0)
        - 2 images with increased contrast (gamma > 1.0)
    """
    # b) Convert to floating point
    # c) Normalize to 0.0-1.0 range
    img_f = img.astype(np.float32) / 255.0

    # d) Create inverted image: New = 1.0 - Old
    inverted = 1.0 - img_f

    # e) Power function for contrast adjustment
    # 2 values between 0.25-0.99 (decrease contrast)
    # 2 values between 1.1-4.0 (increase contrast)
    gammas = [0.4, 0.8, 1.5, 3.0]
    gamma_imgs = [np.power(img_f, g) for g in gammas]

    # Prepare results for display (convert back to 0-255 range)
    results = [img, (inverted * 255).astype(np.uint8)]
    titles = ["Original", "Inverted"]

    for g, gi in zip(gammas, gamma_imgs):
        results.append((gi * 255).astype(np.uint8))
        titles.append(f"Gamma {g}")

    # Display all 6 images
    show_images(results, titles, cols=3)


# -------------------- Part 3: Histogram Equalization --------------------
# Requirements:
# a) Read image and make it grayscale ✓ (done in load_grayscale)
# b) Calculate and display histogram of original image ✓
# c) Enhance contrast using histogram equalization ✓
#    Display uniform histogram and enhanced image ✓
# d) Explain why histograms are different (in report)
# e) Perform image subtraction: d(x,y) = 128 + (o(x,y) - r(x,y))/2 ✓
#    Display difference image ✓
# Display: original + histogram, equalized + histogram, difference image ✓

def histogram_equalization(img):
    """
    Part 3: Demonstrates histogram equalization for contrast enhancement.

    Args:
        img: Grayscale input image

    Displays:
        - Original image and its histogram
        - Equalized image and its histogram
        - Difference image between original and equalized

    The difference image uses the formula: d(x,y) = 128 + (o(x,y) - r(x,y))/2
    where 128 is added to handle negative values and avoid underflow.
    """
    # b) Calculate histogram of original image
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # c) Perform histogram equalization
    eq = cv2.equalizeHist(img)
    hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256])

    # e) Image subtraction: d(x,y) = 128 + (o(x,y) - r(x,y))/2
    # Using int16 to handle negative values, then adding 128 to center at gray
    diff = 128 + (img.astype(np.int16) - eq.astype(np.int16))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Display all results in a 2x3 grid
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Original histogram
    plt.subplot(2, 3, 2)
    plt.plot(hist)
    plt.title("Original Histogram")

    # Equalized image
    plt.subplot(2, 3, 4)
    plt.imshow(eq, cmap='gray')
    plt.title("Equalized")
    plt.axis('off')

    # Equalized histogram (more uniform distribution)
    plt.subplot(2, 3, 5)
    plt.plot(hist_eq)
    plt.title("Equalized Histogram")

    # Difference image
    plt.subplot(2, 3, 6)
    plt.imshow(diff, cmap='gray')
    plt.title("Difference Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# -------------------- Part 4: Image Scaling by Pixel Replication (CSC 481 only) --------------------
# Requirements:
# a) Convert input image to grayscale ✓ (done in load_grayscale)
# b) Shrink image by factor of 8 in each dimension ✓ (factor=-7 → 1/8 size)
# c) Zoom image back to original size ✓ (factor=8)
# d) Calculate difference using Part 3.e method ✓
# Display: grayscale + shrunk + restored + difference ✓
#
# Implementation notes:
# - Positive factor: expands image by replicating each pixel factor×factor times
# - Negative factor: shrinks image by decimating (keeping every (|factor|+1)th pixel)
# - Zero: returns copy of original
# - Uses only numpy array operations (no cv2.resize or similar)

def pixel_scale(img, factor):
    """
    Custom implementation of image scaling using pixel replication and decimation.

    Args:
        img: Input image
        factor: Scaling factor
            - Positive: zoom by replicating pixels (e.g., 3 → 3x3 replication)
            - Negative: shrink by decimation (e.g., -2 → keep every 3rd pixel)
            - Zero: return copy of original

    Returns:
        Scaled image

    Examples:
        factor=3:  Each pixel becomes a 3×3 block (image 3× larger)
        factor=-2: Keep every 3rd pixel (image 1/3 size)
    """
    if factor == 0:
        return img.copy()

    if factor > 0:
        # Zoom: replicate each pixel factor×factor times
        # np.repeat on axis=0 (rows), then axis=1 (columns)
        return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)
    else:
        # Shrink: decimate by keeping every (|factor|+1)th pixel
        # factor=-2 → step=3 → keep indices 0,3,6,9...
        step = abs(factor) + 1
        return img[::step, ::step]


def pixel_replication_demo(img):
    """
    Part 4: Demonstrates image scaling using custom pixel replication/decimation.

    Args:
        img: Grayscale input image

    Displays:
        - Original grayscale image
        - Shrunk image (1/8 size using factor=-7)
        - Restored image (zoomed back to original size using factor=8)
        - Difference image between original and restored
    """
    # b) Shrink by factor of 8 (use -7: step=8, so 1/8 size)
    shrunk = pixel_scale(img, -7)

    # c) Zoom back to original size
    restored = pixel_scale(shrunk, 8)

    # Crop both to common size to avoid shape mismatch
    # (restored might be slightly different size due to rounding)
    h = min(img.shape[0], restored.shape[0])
    w = min(img.shape[1], restored.shape[1])

    # d) Calculate difference using same method as Part 3.e
    # d(x,y) = 128 + (o(x,y) - r(x,y))/2
    diff = 128 + (img[:h, :w].astype(np.int16) - restored[:h, :w].astype(np.int16))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Display all 4 images
    show_images(
        [img, shrunk, restored, diff],
        ["Original", "Shrunk", "Restored", "Difference"],
        cols=2,
    )


# -------------------- Main --------------------

def main():
    """
    Main function that processes all images in the data/hw2 directory.

    Expected images (as per assignment requirements):
    1. A picture of an interesting dish from your hometown
    2. A picture of an interesting dish from Chicago (taken by you)
    3. The image provided in the submission folder

    For each image, runs all four parts:
    - Part 1: Image Scaling by Interpolation
    - Part 2: Basic Gray Level Transformations
    - Part 3: Histogram Equalization
    - Part 4: Image Scaling by Pixel Replication (CSC 481 only)
    """
    base = Path(__file__).resolve().parents[2] / "data" / "hw2"
    images = list(base.glob("*.jpg")) + list(base.glob("*.jpeg"))

    if not images:
        print(f"No images found in {base}")
        print("Please add your 3 required images to the data/hw2 directory")
        return

    print(f"Found {len(images)} image(s) to process")

    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")

        img = load_grayscale(img_path)
        print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")

        print("\nPart 1: Image Scaling by Interpolation...")
        interpolation_demo(img, scale=0.2)

        print("Part 2: Basic Gray Level Transformations...")
        gray_level_transforms(img)

        print("Part 3: Histogram Equalization...")
        histogram_equalization(img)

        print("Part 4: Image Scaling by Pixel Replication (CSC 481)...")
        pixel_replication_demo(img)


if __name__ == "__main__":
    main()
