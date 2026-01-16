import cv2
import numpy as np
import matplotlib.pyplot as plt

print("All imports working!")


# Paths to images
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
print(DATA_DIR)
img_path = os.path.join(DATA_DIR, "1-bean-chicago.jpg")
print(img_path)

image_files = ["1-bean-chicago.jpg", "1-fountain.jpeg", "1-noor-mahal.jpg"]

# Part 1 – Getting familiar with image manipulation

# a) read the images
for fname in image_files:
    print(f"\nProcessing (Part 2): {fname}")

    path = os.path.join(DATA_DIR, fname)
    img_bgr = cv2.imread(path)

    if img_bgr is None:
        print(f"❌ Could not load {fname}")
        continue

    # img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img_bgr)
    # plt.show()

    # b) grayscale copy of the image
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_bgr)
    # plt.show()
    # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    print(img_gray.shape)

    # c) size (width, height and total number of pixels) of the image
    height, width = img_bgr.shape[:2]
    total_pixels = img_bgr.size
    total_pixels_resolution = height * width

    height_gray, width_gray = img_gray.shape
    total_pixels_gray = img_gray.size
    total_pixels_resolution_gray = height_gray * width_gray

    print("Image Dimensions Original:")
    print(f"  Width Orignimal: {width} pixels")
    print(f"  Height Original: {height} pixels")
    print(f"Total number of pixels Original: {total_pixels_resolution}")

    print("Image Dimensions Grayscale:")
    print(f"  Width Grayscale: {width_gray} pixels")
    print(f"  Height Grayscale: {height_gray} pixels")
    print(f"Total number of pixels Grayscale: {total_pixels_resolution_gray}")

    # d)  minimum and maximum pixel values
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)
    print(f"Minimum pixel value Grayscale: {min_val}")
    print(f"Maximum pixel value Grayscale: {max_val}")

    # e) mean pixel value
    mean_val = np.mean(img_gray)
    print(f"Mean pixel value Grayscale: {mean_val}")

    # f) simple binarization Change the pixel values of the image in the following way: all pixels’ values less than the average calculated at (e) will be equal to 0 and all the others will be equal to 1. When displaying the image, make sure your displayed image is black and white, not black and near black.
    _, img_bin = cv2.threshold(img_gray, mean_val, 1, cv2.THRESH_BINARY)
    # plt.imshow(img_bin, cmap="gray")
    # plt.show()

    # g) Colored binarization. Use the same idea from step e to compute the per-channel mean pixel value on the original color image. Then, apply the simple binarization process described on step f on each channel, independently, and then combine the results into a single image for display. The resulting image should be a color image.

    # ***** NUMPY WAY (explicitly showing the per channel mean and binarization) *******
    # color_binary = np.zeros_like(img_bgr)
    # for c in range(3):  # R, G, B
    #     channel_mean = np.mean(img_bgr[:, :, c])
    #     color_binary[:, :, c] = (img_bgr[:, :, c] >= channel_mean).astype(np.uint8) * 255
    # plt.imshow(cv2.cvtColor(color_binary, cv2.COLOR_BGR2RGB))
    # plt.title("Colored Binarized")
    # plt.axis('off')
    # plt.show()

    b, g, r = cv2.split(img_bgr)

    # Apply threshold per channel
    _, b_bin = cv2.threshold(b, np.mean(b), 255, cv2.THRESH_BINARY)
    _, g_bin = cv2.threshold(g, np.mean(g), 255, cv2.THRESH_BINARY)
    _, r_bin = cv2.threshold(r, np.mean(r), 255, cv2.THRESH_BINARY)

    # Merge back into a color image
    color_binary = cv2.merge([b_bin, g_bin, r_bin])

    # plt.imshow(cv2.cvtColor(color_binary, cv2.COLOR_BGR2RGB))  # convert for correct display in matplotlib
    # plt.show()

    # plt.figure(figsize=(12,6))
    # plt.subplot(1,3,1)
    # plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    # plt.title("Original (Part 1)")

    # plt.subplot(1,3,2)
    # plt.imshow(img_gray, cmap="gray")
    # plt.title("Grayscale (Part 1)")

    # plt.subplot(1,3,3)
    # plt.imshow(img_bin*255, cmap="gray")  # multiply by 255 for display
    # plt.title("Simple Binarized (Part 1)")

    # plt.show()

    # # Then display colored binarized separately
    # plt.imshow(cv2.cvtColor(color_binary, cv2.COLOR_BGR2RGB))
    # plt.title("Colored Binarized (Part 1)")
    # plt.show()
    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"Part 1: Original ({fname})")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(img_gray, cmap="gray")
    plt.title(f"Part 1: Grayscale ({fname})")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(img_bin*255, cmap="gray")  # multiply by 255 for display
    plt.title(f"Part 1: Simple Binarized ({fname})")
    plt.axis('off')

    plt.show()

    # Then display colored binarized separately
    plt.imshow(cv2.cvtColor(color_binary, cv2.COLOR_BGR2RGB))
    plt.title(f"Part 1: Colored Binarized ({fname})")
    plt.axis('off')
    plt.show()



# --- Part 2: Patch-based image manipulation ---
# a) read the images
for fname in image_files:
    print(f"\nProcessing (Part 2): {fname}")
    
    path = os.path.join(DATA_DIR, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"❌ Could not load {fname}")
        continue
    
    # b) Convert to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    height, width = img_bgr.shape[:2]
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # c) --- Compute patch grid ---
    Pw = int(round(width / 100))   # patches along width (columns)
    Ph = int(round(height / 100))  # patches along height (rows)
    print(f"Patch grid size: Pw={Pw}, Ph={Ph}")
    
    # d) Compute patch sizes per row/col to handle remainder
    patch_widths = [width // Pw] * Pw
    for i in range(width % Pw):
        patch_widths[i] += 1
    
    patch_heights = [height // Ph] * Ph
    for i in range(height % Ph):
        patch_heights[i] += 1
    
    # # e) --- Patch-wise grayscale binarization ---
    # gray_patch_bin = np.zeros_like(img_gray)
    
    # y_start = 0
    # for ph in patch_heights:
    #     x_start = 0
    #     for pw in patch_widths:
    #         patch = img_gray[y_start:y_start+ph, x_start:x_start+pw]
    #         patch_mean = np.mean(patch)
    #         bin_patch = (patch >= patch_mean).astype(np.uint8) * 255
    #         gray_patch_bin[y_start:y_start+ph, x_start:x_start+pw] = bin_patch
    #         x_start += pw
    #     y_start += ph
    
    # e) --- Patch-wise grayscale binarization with mean report ---
    gray_patch_bin = np.zeros_like(img_gray)
    patch_means = []  # store means for report

    y_start = 0
    for row_idx, ph in enumerate(patch_heights):
        x_start = 0
        for col_idx, pw in enumerate(patch_widths):
            patch = img_gray[y_start:y_start+ph, x_start:x_start+pw]
            patch_mean = np.mean(patch)
            patch_means.append(patch_mean)
            print(f"Patch ({row_idx},{col_idx}) mean: {patch_mean:.2f}")
            bin_patch = (patch >= patch_mean).astype(np.uint8) * 255
            gray_patch_bin[y_start:y_start+ph, x_start:x_start+pw] = bin_patch
            x_start += pw
        y_start += ph

    # f) --- Patch-wise colored binarization ---
    color_patch_bin = np.zeros_like(img_bgr)
    
    # Split channels
    b, g, r = cv2.split(img_bgr)
    channels = [b, g, r]
    color_bin_channels = []
    
    for ch in channels:
        ch_bin = np.zeros_like(ch)
        y_start = 0
        for ph in patch_heights:
            x_start = 0
            for pw in patch_widths:
                patch = ch[y_start:y_start+ph, x_start:x_start+pw]
                patch_mean = np.mean(patch)
                bin_patch = (patch >= patch_mean).astype(np.uint8) * 255
                ch_bin[y_start:y_start+ph, x_start:x_start+pw] = bin_patch
                x_start += pw
            y_start += ph
        color_bin_channels.append(ch_bin)
    
    # Merge back into color image
    color_patch_bin = cv2.merge(color_bin_channels)
    
    # --- Display results ---
    plt.figure(figsize=(12,10))

    plt.subplot(2,2,1)
    plt.imshow(img_rgb)
    plt.title(f"Part 2: Original ({fname})")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(img_gray, cmap='gray')
    plt.title(f"Part 2: Grayscale ({fname})")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(gray_patch_bin, cmap='gray')
    plt.title(f"Part 2: Patch-wise Binarized (Grayscale) ({fname})")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(cv2.cvtColor(color_patch_bin, cv2.COLOR_BGR2RGB))
    plt.title(f"Part 2: Patch-wise Binarized (Color) ({fname})")
    plt.axis('off')

    plt.show()


# Part 3 – Reducing the number of Gray Levels in an Image
# Write code that reduces the number of gray levels in an image from 256 to 2, in integer powers of 2. Do not use library functions to do this, implement in your code the pixel math necessary for the gray level reduction. For each input image, your display should include 8 images (as we saw in class), the original and the 7 reduced intensity images. Strive to have your images not to display darker and darker as you reduce the number of intensities. In addition, write code such that the desired number of gray levels does not have to be a power of 2. Show an example run of your code using a non-integer power of 2 (e.g. 2^3.7) number of gray levels. For every input image (you have 3 inputs, see Data), you must show: the original image, the grayscale image, the images for each number of grayscale level (128, 64, …, 4, 2), and one image that uses a custom non-integer power of 2.

for fname in image_files:
    print(f"\nProcessing Part 3: {fname}")
    
    path = os.path.join(DATA_DIR, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"❌ Could not load {fname}")
        continue
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # List of gray levels (powers of 2)
    gray_levels_list = [128, 64, 32, 16, 8, 4, 2, 1]  # can add more if needed
    
    # Custom non-integer power of 2
    custom_levels = int(2**3.7)  # example: 2^3.7 ≈ 13 levels
    
    plt.figure(figsize=(20,8))
    
    # 1. Original grayscale image
    plt.subplot(2,5,1)
    plt.imshow(img_gray, cmap='gray')
    plt.title(f"Original Gray ({fname})")
    plt.axis('off')
    
    # 2. Quantized images for each gray level
    for idx, L in enumerate(gray_levels_list):
        # Reduce gray levels
        img_quant = np.floor(img_gray / 256 * L) * (256 / L)
        # Convert to uint8 for matplotlib display
        img_quant = img_quant.astype(np.uint8)
        
        # idx + 2 because we already used 1 for original
        plt.subplot(2,5,idx+2)
        plt.imshow(img_quant, cmap='gray')
        plt.title(f"{L} Levels")
        plt.axis('off')
    
    # 3. Non-integer power of 2 example
    img_custom = np.floor(img_gray / 256 * custom_levels) * (256 / custom_levels)
    img_custom = img_custom.astype(np.uint8)
    plt.subplot(2,5,9)
    plt.imshow(img_custom, cmap='gray')
    plt.title(f"{custom_levels} Levels (2^3.7)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Part 4 – Bit Plane Splicing

for fname in image_files:
    print(f"\nProcessing Part 4: {fname}")
    
    path = os.path.join(DATA_DIR, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"❌ Could not load {fname}")
        continue

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape[:2]

    # --- 1. Extract all bit planes first ---
    bit_planes = []
    for i in range(8):
        plane = (img_gray >> i) & 1  # shift right i bits and mask LSB
        plane = plane * 255          # scale 0/1 -> 0/255
        plane = plane.astype(np.uint8)
        bit_planes.append(plane)

    # --- 2. Display original + bit planes ---
    plt.figure(figsize=(20,6))
    plt.subplot(2,5,1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Gray (Part 4)")
    plt.axis('off')

    for i, plane in enumerate(bit_planes):
        plt.subplot(2,5,i+2)
        plt.imshow(plane, cmap='gray')
        plt.title(f"Bit plane {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # --- 3. Gradually assemble the image from MSB down ---
    reconstructed_images = []
    reconstructed = np.zeros_like(img_gray, dtype=np.uint8)
    
    for i in reversed(range(8)):  # 7 -> 0
        reconstructed += (bit_planes[i] // 255) * (2**i)  # convert 255->1, multiply by bit value
        reconstructed_images.append(reconstructed.copy())

    # --- 4. Display progressive reconstruction ---
    plt.figure(figsize=(20,6))
    for idx, rec_img in enumerate(reconstructed_images):
        plt.subplot(2,4,idx+1)
        plt.imshow(rec_img, cmap='gray')
        plt.title(f"Planes 7->{7-idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
