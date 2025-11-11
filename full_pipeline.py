import os
import subprocess
import sys
import gc

import subprocess

# Full shell command with conda activation
command = "source /root/anaconda3/etc/profile.d/conda.sh && conda activate segformer3d && conda info --envs"

result = subprocess.run(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

print(result.stdout)
print(result.stderr)

def run_command(command_list, description):
    """Helper function to run a subprocess command with error handling."""
    print(f"--- {description} ---")
    try:
        subprocess.run(command_list, check=True)
        print(f"Success: {description}\n")
    except subprocess.CalledProcessError:
        print(f"Error: {description} failed.\n")
        sys.exit(1)

# Change to the target working directory
target_dir = "/work/SebiButa/"
try:
    os.chdir(target_dir)
    print(f"Changed directory to: {os.getcwd()}\n")
except FileNotFoundError:
    print(f"Error: Directory {target_dir} does not exist.")
    sys.exit(1)

# Clone the LightGlue repository (if not already cloned)
repo_url = "https://github.com/cvg/LightGlue.git"
repo_name = "LightGlue"

if not os.path.exists(repo_name):
    run_command(["git", "clone", repo_url], "Cloning LightGlue repository")
else:
    print(f"Repository '{repo_name}' already exists. Skipping clone.\n")

# Change into the repo directory and install in editable mode
os.chdir(repo_name)
print(f"Changed directory to repo: {os.getcwd()}\n")
run_command(["python", "-m", "pip", "install", "-e", "."], "Installing LightGlue in editable mode")

# Install additional dependencies
run_command(["python", "-m", "pip", "install", "scikit-image>=0.22"], "Installing compatible scikit-image version")
run_command(["python", "-m", "pip", "install", "matplotlib"], "Installing matplotlib")
run_command(["python", "-m", "pip", "install", "kornia"], "Installing kornia")

# DIRECTORIES 
INPUT_FOLDER = "Dataset_full/images"
OUTPUT_FOLDER = "Dataset_full/images_result_full"

import os
from PIL import Image
import PIL
import random
import cv2
import shutil
import numpy as np
import torch
import math
import json
from typing import List
from skimage.metrics import structural_similarity as ssim
from typing import Optional

import matplotlib.pyplot as plt
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint, SIFT
from lightglue.utils import load_image, rbd

PIL.Image.MAX_IMAGE_PIXELS = 933120000
plt.rcParams['figure.figsize'] = [15, 15]

# Change to the target working directory
target_dir = "/work/SebiButa/"
try:
    os.chdir(target_dir)
    print(f"Changed directory to: {os.getcwd()}\n")
except FileNotFoundError:
    print(f"Error: Directory {target_dir} does not exist.")
    sys.exit(1)

name_map = {
    "0000.png": "0000.png",
    "0005.png": "0001.png",
    "0001.png": "0002.png",
    "0010.png": "0003.png",
    "0006.png": "0004.png",
    "0002.png": "0005.png",
    "0015.png": "0006.png",
    "0011.png": "0007.png",
    "0007.png": "0008.png",
    "0003.png": "0009.png",
    "0020.png": "0010.png",
    "0016.png": "0011.png",
    "0012.png": "0012.png",
    "0008.png": "0013.png",
    "0004.png": "0014.png",
    "0021.png": "0015.png",
    "0017.png": "0016.png",
    "0013.png": "0017.png",
    "0009.png": "0018.png",
    "0022.png": "0019.png",
    "0018.png": "0020.png",
    "0014.png": "0021.png",
    "0023.png": "0022.png",
    "0019.png": "0023.png",
    "0024.png": "0024.png"
}

def rename_and_copy_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            base_name = os.path.basename(filename)
            if base_name in name_map:
                new_name = name_map[base_name]
                src = os.path.join(input_folder, filename)
                dst = os.path.join(output_folder, new_name)
                shutil.copyfile(src, dst)
                # print(f"Copied {filename} as {new_name}")
            else:
                print(f"Skipping {filename}: not in mapping")

def pad_images(
    input_folder: str,
    output_folder: str,
    padding_fraction: float = 0.2
) -> None:
    """
    Pads all images in the input folder with black borders and saves them to the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path where padded images will be saved.
    - padding_fraction (float): The fraction of 1/5 of the image's height and width to be used as padding.
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.svs')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            img: Optional[np.ndarray] = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image: {filename}")
                continue

            h, w = img.shape[:2]

            # Calculate padding amounts (20% of 1/5 of original size)
            pad_height_total = int(h * padding_fraction * 0.2)
            pad_width_total = int(w * padding_fraction * 0.2)

            # Split total padding equally on both sides
            pad_top = pad_bottom = pad_height_total // 2
            pad_left = pad_right = pad_width_total // 2

            # Apply black border padding
            padded_img = cv2.copyMakeBorder(
                img,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # Black color
            )

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, padded_img)
            print(f"Padded and saved: {filename} → {output_path}")

def split_image_into_patches(image_path, output_dir, grid_size=(5, 5)):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Overlap ranges (as before)
    h_overlap_min, h_overlap_max = 0.17, 0.23  # 20% ± 3%
    v_overlap_min, v_overlap_max = 0.175, 0.225  # 20% ± 2.5%

    # Random overlaps per row/column
    h_overlaps = [random.uniform(h_overlap_min, h_overlap_max) for _ in range(grid_size[0] - 1)]
    v_overlaps = [random.uniform(v_overlap_min, v_overlap_max) for _ in range(grid_size[1] - 1)]

    # Calculate patch dimensions (approximate)
    avg_h_overlap = sum(h_overlaps) / len(h_overlaps)
    avg_v_overlap = sum(v_overlaps) / len(v_overlaps)
    patch_width = img_width / (grid_size[0] - (grid_size[0] - 1) * avg_h_overlap)
    patch_height = img_height / (grid_size[1] - (grid_size[1] - 1) * avg_v_overlap)

    # Random translational offsets (±5% of patch size)
    max_offset_x = int(patch_width * 0.05)
    max_offset_y = int(patch_height * 0.05)

    patch_num = 0
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            # Base position (with overlaps)
            h_step = patch_width * (1 - (h_overlaps[col - 1] if col > 0 else 0))
            v_step = patch_height * (1 - (v_overlaps[row - 1] if row > 0 else 0))

            left = int(col * h_step)
            upper = int(row * v_step)
            right = int(left + patch_width)
            lower = int(upper + patch_height)

            # Apply random offsets (misalignment)
            offset_x = random.randint(-max_offset_x, max_offset_x)
            offset_y = random.randint(-max_offset_y, max_offset_y)

            left += offset_x
            upper += offset_y
            right += offset_x
            lower += offset_y

            # Clamp to image bounds
            left = max(0, left)
            upper = max(0, upper)
            right = min(img_width, right)
            lower = min(img_height, lower)

            # Crop and save
            patch = img.crop((left, upper, right, lower))
            patch_name = f"{patch_num:04d}.png"
            patch.save(os.path.join(output_dir, patch_name))
            patch_num += 1

# Read image and convert them to gray!!
def read_image(path, resize=1.0):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found at path: {path}")

    # Resize to 80% of original size
    height, width = img.shape[:2]
    new_size = (int(width * 0.8), int(height * 0.8))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_gray, img, img_rgb

def KAZE(img):
    kazeDetector = cv2.KAZE_create()  # Create KAZE detector (supports both AKAZE and KAZE)

    kp, des = kazeDetector.detectAndCompute(img, None)
    return kp, des

def plot_kaze(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    best_inliers = 0

    for i in range(iters):
        points = random_point(matches)
        H = homography(points)

        #  avoid dividing by zero
        if np.linalg.matrix_rank(H) < 3:
            continue

        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    # print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def stitch_img(left, right, H):
    # print("Stitching image ...")

    # Normalize images to float in [0,1]
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Get dimensions of both images
    height_l, width_l, _ = left.shape
    height_r, width_r, _ = right.shape

    # Compute corners for the left image and transform them using H
    corners_left = np.array([[0, 0, 1],
                             [width_l, 0, 1],
                             [width_l, height_l, 1],
                             [0, height_l, 1]]).T  # shape (3,4)
    warped_corners_left = H @ corners_left
    warped_corners_left /= warped_corners_left[2, :]  # Normalize homogeneous coordinates

    # Compute corners for the right image (identity transform)
    corners_right = np.array([[0, 0, 1],
                              [width_r, 0, 1],
                              [width_r, height_r, 1],
                              [0, height_r, 1]]).T  # shape (3,4)

    # Combine corners to find overall bounds
    all_x = np.concatenate((warped_corners_left[0, :], corners_right[0, :]))
    all_y = np.concatenate((warped_corners_left[1, :], corners_right[1, :]))

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)

    # Create a translation matrix to shift all images so that no coordinate is negative
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0
    translation_mat = np.array([[1, 0, tx],
                                [0, 1, ty],
                                [0, 0, 1]])

    # New canvas size: use ceiling to ensure full coverage
    width_new = int(np.ceil(max_x - min_x))
    height_new = int(np.ceil(max_y - min_y))
    size = (width_new, height_new)

    # Warp left image with the composite transform: translation_mat @ H
    warped_left = cv2.warpPerspective(left, translation_mat @ H, size)
    # Warp right image with just the translation matrix (identity warp + translation)
    warped_right = cv2.warpPerspective(right, translation_mat, size)

    # Vectorized blending:
    # Create masks where any channel is non-zero (assumed as non-black)
    mask_left = np.any(warped_left != 0, axis=2)
    mask_right = np.any(warped_right != 0, axis=2)

    # Initialize the stitched image as black
    stitch_image = np.zeros_like(warped_left)

    # Pixels only from left image
    only_left = mask_left & ~mask_right
    stitch_image[only_left] = warped_left[only_left]

    # Pixels only from right image
    only_right = mask_right & ~mask_left
    stitch_image[only_right] = warped_right[only_right]

    # Pixels where both images contribute (average the pixel values)
    both = mask_left & mask_right
    stitch_image[both] = (warped_left[both] + warped_right[both]) / 2

    return stitch_image
def mask_image_top_left(img_gray, k):
    """
    Keeps the top k% rows and left-most k% columns intact, rest set to black.

    Parameters:
        img_gray (np.ndarray): Grayscale image.
        k (float): Percentage (0 < k <= 100) of image to retain in top and left.

    Returns:
        np.ndarray: Modified image.
    """
    if not (0 < k <= 100):
        raise ValueError("k must be between 0 and 100 (exclusive of 0).")

    h, w = img_gray.shape
    mask = np.zeros_like(img_gray)

    # Calculate cutoff indices
    cutoff_row = int(h * k / 100)
    cutoff_col = int(w * k / 100)

    # Retain top k% rows
    mask[:cutoff_row, :] = img_gray[:cutoff_row, :]

    # Retain left-most k% columns
    mask[:, :cutoff_col] = img_gray[:, :cutoff_col]

    return mask

def detect_black_border_depths(image_pil, black_threshold=10):
    """
    Detects how much black padding exists on each edge of the image separately.

    Args:
        image_pil: PIL.Image object (RGB or grayscale).
        black_threshold: Maximum pixel value considered "black" (0-255).

    Returns:
        (top, bottom, left, right): number of black pixels to crop from each side.
    """
    image_np = np.array(image_pil)

    if image_np.ndim == 3:
        # Convert to grayscale if RGB
        image_np = np.mean(image_np, axis=2)

    h, w = image_np.shape

    # Initialize cropping values
    top = 0
    bottom = 0
    left = 0
    right = 0

    # Detect top
    for y in range(h):
        if np.all(image_np[y, :] <= black_threshold):
            top += 1
        else:
            break

    # Detect bottom
    for y in range(h-1, -1, -1):
        if np.all(image_np[y, :] <= black_threshold):
            bottom += 1
        else:
            break

    # Detect left
    for x in range(w):
        if np.all(image_np[:, x] <= black_threshold):
            left += 1
        else:
            break

    # Detect right
    for x in range(w-1, -1, -1):
        if np.all(image_np[:, x] <= black_threshold):
            right += 1
        else:
            break

    return top, bottom, left, right

def crop_black_borders(image_pil, black_threshold=10):
    """
    Crop black borders independently from each edge.

    Args:
        image_pil: PIL.Image object (RGB or grayscale).
        black_threshold: Maximum pixel value considered "black" (0-255).

    Returns:
        Cropped PIL.Image object.
    """
    if isinstance(image_pil, np.ndarray):
        # Convert ndarray to PIL Image
        image_pil = Image.fromarray(image_pil)

    top, bottom, left, right = detect_black_border_depths(image_pil, black_threshold)
    w, h = image_pil.size

    # Calculate new box
    left_crop = left
    upper_crop = top
    right_crop = w - right
    lower_crop = h - bottom

    if left_crop >= right_crop or upper_crop >= lower_crop:
        # If everything is cropped away, return original (or could raise an error)
        print("Warning: Cropping would remove entire image. Returning original.")
        return image_pil

    return image_pil.crop((left_crop, upper_crop, right_crop, lower_crop))

def KAZE_matching(img_path1, img_path2, output_dir="./kaze_output", kaze_thresholding=0.7, ransac_thresholding=100, percentage_of_image_used=1.0):
    left_gray, left_origin, left_rgb = read_image(img_path1)
    right_gray, right_origin, right_rgb = read_image(img_path2)

    right_gray = mask_image_top_left(right_gray, percentage_of_image_used * 100)

    kp_left, des_left = KAZE(left_gray)
    kp_right, des_right = KAZE(right_gray)

    matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.7)

    inliers, H = ransac(matches, 100, 2000)

    kaze_stitched = stitch_img(left_rgb, right_rgb, H)

    kaze_stitched_uint8 = (kaze_stitched * 255).astype(np.uint8)
    image_pil = Image.fromarray(kaze_stitched_uint8)

    output_filename = f"0000-{os.path.splitext(os.path.basename(img_path2))[0]}_KAZE_{kaze_thresholding}_{percentage_of_image_used}.jpg"
    output_filepath = os.path.join(output_dir, output_filename)

    cropped_image = crop_black_borders(image_pil, black_threshold=10)
    cropped_image.save(output_filepath, "JPEG", quality=95)

    return output_filename

def torch_to_cv2_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a C×H×W RGB torch tensor (on CPU or CUDA) to a H×W×C BGR uint8 numpy array.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    image = tensor.clone().detach()
    # (C,H,W) -> (H,W,C)
    image = image.permute(1, 2, 0).numpy()
    # assume floats in [0,1] or ints in [0,255]
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    # RGB->BGR
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def keep_top_left_percent(image, k: float):
    """
    Keep only the top and left k-percent of pixels (an upside-down 'L'), black out the rest.
    Works on either
      • numpy.ndarray (H×W×C BGR)  or
      • torch.Tensor   (C×H×W RGB floats [0,1] or [0,255], on CPU or CUDA)
    """
    # --- validate k ---
    if not (0 < k <= 1):
        raise ValueError("Parameter k must be a float between 0 and 1.")

    # --- branch on type ---
    if isinstance(image, torch.Tensor):
        # print("Is tensor")
        # tensor path: C×H×W
        C, H, W = image.shape
        top_h = math.ceil(k * H)
        left_w = math.ceil(k * W)

        # build mask on same device & dtype
        mask2d = torch.zeros((H, W), dtype=image.dtype, device=image.device)
        mask2d[:top_h, :] = 1
        mask2d[:, :left_w] = 1

        # expand to C×H×W
        mask3d = mask2d.unsqueeze(0).expand(C, H, W)

        # apply mask
        out_t = image * mask3d

        return out_t

    elif isinstance(image, np.ndarray):
        print("Is not tensor!")
        # numpy path: H×W×C BGR
        H, W = image.shape[:2]
        top_h = math.ceil(k * H)
        left_w = math.ceil(k * W)

        mask = np.zeros((H, W), dtype=bool)
        mask[:top_h, :] = True
        mask[:, :left_w] = True

        black = np.zeros_like(image)
        out_np = np.where(mask[..., None], image, black)

        return out_np

    else:
        raise TypeError(
            f"Unsupported image type {type(image)} – must be numpy.ndarray or torch.Tensor."
        )


def resize_torch(image0, new_size):
  # image0 is a C×H×W float tensor on CUDA
  # 1) send to CPU and turn into a H×W×C NumPy array
  np_img = image0.cpu().permute(1, 2, 0).numpy()

  # 2) do your OpenCV resize
  h, w = np_img.shape[:2]
  new_w = (w * int(new_size * 10)) // 10
  new_h = (h * int(new_size * 10)) // 10
  resized_np = cv2.resize(np_img, (new_w, new_h))

  # 3) (optional) turn back into a tensor, same dtype/device as before
  return torch.from_numpy(resized_np).permute(2, 0, 1).to(image0.dtype).cuda()


def crop_tensor_image(image: torch.Tensor, h1: int, w1: int, h2: int, w2: int) -> torch.Tensor:
    """
    Crop an image tensor based on bounding box coordinates.

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W), values in [0,1] or [0,255].
        h1 (int): Top coordinate (inclusive).
        w1 (int): Left coordinate (inclusive).
        h2 (int): Bottom coordinate (exclusive).
        w2 (int): Right coordinate (exclusive).

    Returns:
        torch.Tensor: Cropped image tensor of shape (C, h2-h1, w2-w1).
    """
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D tensor (C, H, W)")

    C, H, W = image.shape

    # Clamp coordinates to image boundaries
    h1 = max(0, min(H, h1))
    h2 = max(0, min(H, h2))
    w1 = max(0, min(W, w1))
    w2 = max(0, min(W, w2))

    if h1 >= h2 or w1 >= w2:
        raise ValueError("Invalid bounding box: (h1, w1) must be above and to the left of (h2, w2)")

    return image[:, h1:h2, w1:w2]

def poz(matrix_size, i):
  init_i = i

  if i < 1:
    return None
  if i > (matrix_size * matrix_size) - 1:
    return None

  max_line = matrix_size - 1

  diag_nr = 1
  x = diag_nr
  y = 0
  prev_poz = 0, 0
  while True:
    i -= 1
    if i == 0:
      if init_i < ((matrix_size * (matrix_size + 1)) // 2):
        prev_x, prev_y = prev_poz
        if prev_x == 0:
          return x, y, prev_x + prev_y + 1, prev_x + prev_y + 1
        else:
          return x, y, prev_x + prev_y + 1, prev_x + prev_y
      return x, y, matrix_size, matrix_size

    if i == 1:
      prev_poz = x, y

    if x == 0 or y == max_line:
      diag_nr += 1

      if diag_nr > max_line:
        x = max_line
        y = diag_nr - max_line
      else:
        x = diag_nr
        y = 0
    else:
      x -= 1
      y += 1

def poz_list(path, paths):
  max_0, max_1 = 0, 0
  for path1 in paths:
    if path == path1:
      break
    path1 = convert_image_number_base(path1, 10, 5)
    x, y = int(path1.split(".jpg")[0][2:][0]), int(path1.split(".jpg")[0][2:][1])
    max_0 = max(max_0, x)
    max_1 = max(max_1, y)
  return int(path.split(".jpg")[0][2:][0]), int(path.split(".jpg")[0][2:][1]), max_0 + 1, max_1 + 1

import torchvision.utils as vutils

def save_image(tensor: torch.Tensor, filename: str = "image.png"):
    """
    Save a tensor as an image file.

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W).
        filename (str): Path to save the image.
    """
    # If the tensor is [0, 255] floats, normalize it to [0, 1]
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    vutils.save_image(tensor, filename)

def detect_black_border_depths(image_pil, black_threshold=10):
    """
    Detects how much black padding exists on each edge of the image separately.

    Args:
        image_pil: PIL.Image object (RGB or grayscale).
        black_threshold: Maximum pixel value considered "black" (0-255).

    Returns:
        (top, bottom, left, right): number of black pixels to crop from each side.
    """
    image_np = np.array(image_pil)

    if image_np.ndim == 3:
        # Convert to grayscale if RGB
        image_np = np.mean(image_np, axis=2)

    h, w = image_np.shape

    # Initialize cropping values
    top = 0
    bottom = 0
    left = 0
    right = 0

    # Detect top
    for y in range(h):
        if np.all(image_np[y, :] <= black_threshold):
            top += 1
        else:
            break

    # Detect bottom
    for y in range(h-1, -1, -1):
        if np.all(image_np[y, :] <= black_threshold):
            bottom += 1
        else:
            break

    # Detect left
    for x in range(w):
        if np.all(image_np[:, x] <= black_threshold):
            left += 1
        else:
            break

    # Detect right
    for x in range(w-1, -1, -1):
        if np.all(image_np[:, x] <= black_threshold):
            right += 1
        else:
            break

    return top, bottom, left, right

"""
def crop_black_borders(image_pil, black_threshold=10):
    top, bottom, left, right = detect_black_border_depths(image_pil, black_threshold)
    w, h = image_pil.size

    # Calculate new box
    left_crop = left
    upper_crop = top
    right_crop = w - right
    lower_crop = h - bottom

    if left_crop >= right_crop or upper_crop >= lower_crop:
        # If everything is cropped away, return original (or could raise an error)
        print("Warning: Cropping would remove entire image. Returning original.")
        return image_pil

    return image_pil.crop((left_crop, upper_crop, right_crop, lower_crop))
"""

def torch_image_shape(image):
  np_img = image.cpu().permute(1, 2, 0).numpy()
  h, w = np_img.shape[:2]
  return h, w

def LightGlue_matrix_scan_diagonal_matching(
    img_path1,
    img_path2,
    basepath_img2,
    path_list,
    i_value,
    matrix_size,
    output_dir,
    ransac_thresholding=100,
    percentage_of_image_used=1.0,
    pair_extractor="SIFT",
    verbose=False
):

  # print(convert_image_number_base(basepath_img2, 10, 5))
  poz_h, poz_w, nr_h, nr_w = poz_list(basepath_img2, path_list)

  if verbose:
    print(f"We want to crop the matrix crop {(poz_h, poz_w)} out of a {matrix_size - 1, matrix_size - 1} matrix!")
    print(f"{nr_h}, {nr_w}")

  matches_maxx = []

  # for iii in range(0, nr_h * 10 - 5, 5):
  #   for jjj in range(0, nr_w * 10 - 5, 5):
  #     i = iii / 10
  #     j = jjj / 10
  for i in range(nr_h):
    for j in range(nr_w):
      torch.cuda.empty_cache()
      left_gray, left_origin, left_rgb = read_image(img_path1, 0.8)
      right_gray, right_origin, right_rgb = read_image(img_path2, 0.8)

      # right_height, right_width = img_path2.shape[:2]
      # new_height = int(left_height * percentage_of_image_used)
      # new_width = int(left_width * percentage_of_image_used)

      image0 = load_image(img_path1).cuda()
      image1 = load_image(img_path2).cuda()

      image0 = resize_torch(image0, 0.8)
      image1 = resize_torch(image1, 0.8)

      # image0 = keep_top_left_percent(image0, percentage_of_image_used)
      image1 = keep_top_left_percent(image1, percentage_of_image_used)

      h, w = torch_image_shape(image0)

      if verbose:
        print(f"Image size of {h, w}")

      h_size, w_size = h // nr_h, w // nr_w

      if verbose:
        print(f"Tile size of {(h_size, w_size)}")

      flag = True

      if verbose:
        print(f"h_size = {h_size}")
        print(f"w_size = {w_size}")

        print(f"i = {i}")
        print(f"j = {j}")

      top_left_h = int(max(0, (j * h_size) - (h_size * 0.65)))
      top_left_w = int(max(0, (i * w_size) - (w_size * 0.65)))

      bottom_right_h = int(min(h, ((j + 1) * h_size) + (0.65 * h_size)))
      bottom_right_w = int(min(w, ((i + 1) * w_size) + (0.65 * w_size)))

      if verbose:
        print(f"Box proposed is {(top_left_h, top_left_w, bottom_right_h, bottom_right_w)}")

      image0 = crop_tensor_image(image0, top_left_h, top_left_w, bottom_right_h, bottom_right_w)

      # cropped = crop_tensor_image(image0, top_left_h, top_left_w, bottom_right_h, bottom_right_w)
      # save_image(image0, f"00000_{i_value}_{i}_{j}.png")
      # save_image(image1, f"00001_{i_value}.png")

      if pair_extractor == "SIFT":
        extractor = SIFT(max_num_keypoints=None).eval().cuda()
        matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).cuda()
      elif pair_extractor =="SuperPoint":
        extractor = SuperPoint(max_num_keypoints=None).cuda()
        matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).cuda()
      else:
        raise ValueError(f"Invalid pair extractor: {pair_extractor}")

      try:
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']
        points0 = feats0['keypoints'][matches[..., 0]]
        points1 = feats1['keypoints'][matches[..., 1]]

        matches = np.concatenate((points0.cpu(), points1.cpu()), axis=1)
      except Exception as e:
        print(f"While doing Deep Learning matching got exception: {e}")
        matches = []


      if verbose:
        print(f"\n\n\nAvem {len(matches)} matches!\n\n\n")

      for ii in range(len(matches)):
        matches[ii][0] += top_left_w
        matches[ii][1] += top_left_h

      if len(matches) > len(matches_maxx):
        if verbose:
          print("Updated maxx matches!")
        matches_maxx = matches

  inliers, H = ransac(matches_maxx, 100, 2000)

  kaze_stitched = stitch_img(left_rgb, right_rgb, H)

  kaze_stitched_uint8 = (kaze_stitched * 255).astype(np.uint8)
  image_pil = Image.fromarray(kaze_stitched_uint8)

  output_filename = f"0000-{os.path.splitext(os.path.basename(img_path2))[0]}_{pair_extractor}_{percentage_of_image_used}.jpg"
  output_filepath = os.path.join(output_dir, output_filename)

  if verbose:
    print(f"Saving at {output_filepath}")

  cropped_image = crop_black_borders(image_pil, black_threshold=10)

  cropped_image.save(output_filepath, "JPEG", quality=95)

  return output_filename

def convert_image_number_base(filepath: str, b1: int, b2: int) -> str:
    if not (1 <= b1 <= 10 and 1 <= b2 <= 10):
        raise ValueError("Bases b1 and b2 must be between 1 and 10 (inclusive)")

    # Split into directory, filename, extension
    dirname, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)

    if ext.lower() not in ['.jpg', '.png']:
        raise ValueError("File must be a .jpg or .png")

    # Ensure the number part is 4 digits
    if not (len(name) == 4 and name.isdigit()):
        raise ValueError("Filename must contain a 4-digit number")

    # Convert number from base b1 to int
    try:
        number = int(name, base=b1)
    except ValueError:
        raise ValueError(f"The number '{name}' is not valid in base {b1}")

    # Convert from int to base b2
    def to_base(n: int, base: int) -> str:
        if n == 0:
            return '0'
        digits = []
        while n:
            digits.append(str(n % base))
            n //= base
        return ''.join(reversed(digits))

    new_number = to_base(number, b2).zfill(4)
    new_filename = new_number + ext
    return os.path.join(dirname, new_filename) if dirname else new_filename

def diagonal_submatrix(k, x, y, image_paths):
    matrix_size = k  # Assumes a 10x10 matrix
    """
    Return the k×k submatrix of a width×width grid stored in `image_paths` (row-major order),
    starting at top-left coordinate (x, y), traversed in diagonal order.

    Parameters:
        image_paths (List[str]): Flat list of image image_paths of length width×width.
        k (int): Size of the submatrix (k rows, k columns).
        x (int): Row index of the submatrix's top-left corner (0-based).
        y (int): Column index of the submatrix's top-left corner (0-based).

    Returns:
        List[str]: The k×k submatrix's elements in diagonal order.
    """
    result = []
    # There are 2k-1 diagonals, indexed by sum d = i+j from 0 to 2(k-1).
    for d in range(2 * k - 1):
        # For each diagonal, i goes from min(d, k-1) down to max(0, d-(k-1)).
        start = min(d, k - 1)
        end = max(0, d - (k - 1))
        for i in range(start, end - 1, -1):
            j = d - i
            global_i = x + i
            global_j = y + j
            idx = global_i * matrix_size + global_j
            result.append(image_paths[idx])
    return result

def filter_paths(paths):
  def is_ok(x):
    return int(x[2]) < 5 and int(x[3]) < 5
  return [x for x in paths if is_ok(x)]

"""
def crop_black_borders(image, threshold=10):
    # Create a mask of where the image is not black
    # (using the maximum channel value per pixel)
    if len(image.shape) == 3:
        mask = image.max(axis=2) > threshold
    else:
        mask = image > threshold

    # Get coordinates of non-black pixels
    coords = np.argwhere(mask)
    if coords.size == 0:
        # No non-black pixels; return original image
        return image

    # Find the bounding box of the non-black region
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    cropped = image[y0:y1, x0:x1]
    return cropped
"""

def compute_stitched_ssim(image_path, k, seam_width=10, black_thresh=10):
    """
    Computes the average SSIM for seam areas between adjacent sub-images
    in a K x K stitched image stored in 'image_path'.

    Parameters:
        image_path (str): The file path to the stitched image.
        k (int): The number of rows/columns in the stitched grid (total sub-images = k*k).
        seam_width (int): The width (in pixels) of the region along each seam to compare.
        black_thresh (int): Pixel intensity threshold to consider as black when cropping.

    Returns:
        average_ssim (float): The average SSIM computed over all seams.
        seam_ssim_scores (dict): A dictionary with keys 'vertical' and 'horizontal' containing lists of SSIM scores.
    """
    # Load the image using OpenCV (reads as BGR)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image from '{image_path}'")

    # Convert to RGB for consistency
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Crop out surrounding black borders
    cropped_image = crop_black_borders(image_rgb, black_threshold=black_thresh)

    if isinstance(cropped_image, Image.Image):
        cropped_image = np.array(cropped_image)

    # Get dimensions of the cropped image
    h, w = cropped_image.shape[:2]

    # Determine size of each sub-image (using integer division)
    sub_h = h // k
    sub_w = w // k

    # Create a list to hold the sub-images
    sub_images = []
    for i in range(k):
        row = []
        for j in range(k):
            # Extract the (i, j)th sub-image
            x_start = j * sub_w
            x_end = (j + 1) * sub_w
            y_start = i * sub_h
            y_end = (i + 1) * sub_h
            sub_img = cropped_image[y_start:y_end, x_start:x_end]
            row.append(sub_img)
        sub_images.append(row)

    # Prepare lists to hold SSIM scores from vertical and horizontal seams
    vertical_scores = []
    horizontal_scores = []

    # Compute SSIM on vertical seams (between left & right adjacent sub-images)
    for i in range(k):
        for j in range(k - 1):
            left_img = sub_images[i][j]
            right_img = sub_images[i][j + 1]
            # Select the right seam of left_img and left seam of right_img
            left_seam = left_img[:, -seam_width:]
            right_seam = right_img[:, :seam_width]
            # Convert to grayscale for SSIM calculation
            left_gray = cv2.cvtColor(left_seam, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_seam, cv2.COLOR_RGB2GRAY)
            score, _ = ssim(left_gray, right_gray, full=True)
            vertical_scores.append(score)

    # Compute SSIM on horizontal seams (between top & bottom adjacent sub-images)
    for i in range(k - 1):
        for j in range(k):
            top_img = sub_images[i][j]
            bottom_img = sub_images[i + 1][j]
            # Select the bottom seam of top_img and top seam of bottom_img
            top_seam = top_img[-seam_width:, :]
            bottom_seam = bottom_img[:seam_width, :]
            top_gray = cv2.cvtColor(top_seam, cv2.COLOR_RGB2GRAY)
            bottom_gray = cv2.cvtColor(bottom_seam, cv2.COLOR_RGB2GRAY)
            score, _ = ssim(top_gray, bottom_gray, full=True)
            horizontal_scores.append(score)

    # Combine all scores and compute an average SSIM
    all_scores = vertical_scores + horizontal_scores
    # average_ssim = np.mean(all_scores) if all_scores else None
    penalized_average_ssim = 1 - np.mean((np.array(all_scores) - 0.6)**2)

    seam_ssim_scores = {
        "vertical": vertical_scores,
        "horizontal": horizontal_scores
    }

    return penalized_average_ssim, seam_ssim_scores

"""
def crop_black_borders(image, threshold=10):
    if image.ndim == 3:
        mask = image.max(axis=2) > threshold
    else:
        mask = image > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]
"""

def compute_improved_scan_ssim(
    image_path: str,
    patch_h: int,
    patch_w: int,
    stride: int = 1,
    seam_width: int = 10,
    black_thresh: int = 10,
    α: float = 0.5,
    β: float = 0.4,
    γ: float = 0.1
):
    """
    Sliding‐window scan that computes a combined seam‐quality score:
      Q = α·(intensity SSIM)
        + β·(edge SSIM)
        - γ·(normalized gradient‐difference)
    Returns:
      vert_map, hori_map: arrays of Q‐scores
      mean_Q, rms_Q: overall statistics
    """
    # 1) load & crop
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open {image_path!r}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img = crop_black_borders(img, black_thresh)
    H, W = img.shape[:2]

    # 2) compute grid of sliding windows
    n_rows = (H - patch_h)//stride + 1
    n_cols = (W - patch_w)//stride + 1

    vert_map = np.full((n_rows, n_cols-1), np.nan, dtype=np.float32)
    hori_map = np.full((n_rows-1, n_cols), np.nan, dtype=np.float32)

    # helper: gradient magnitude
    def grad_mag(gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    for i in range(n_rows):
        y = i*stride
        for j in range(n_cols):
            x = j*stride
            patch = img[y:y+patch_h, x:x+patch_w]

            # prepare raw gray and edge maps
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            gmag = grad_mag(gray)

            # vertical seam
            if j+1 < n_cols:
                # neighbor patch
                nx = x + stride
                neigh = img[y:y+patch_h, nx:nx+patch_w]
                gray2 = cv2.cvtColor(neigh, cv2.COLOR_RGB2GRAY)
                edges2 = cv2.Canny(gray2, 100, 200)
                gmag2 = grad_mag(gray2)

                L = gray[:, -seam_width:]
                R = gray2[:, :seam_width]
                E1 = edges[:, -seam_width:]
                E2 = edges2[:, :seam_width]
                G1 = gmag[:, -seam_width:]
                G2 = gmag2[:, :seam_width]

                s_int = ssim(L, R)
                s_edge = ssim(E1, E2)
                # normalized gradient diff
                diff = np.abs(G1 - G2)
                g_diff = np.mean(diff) / (np.max([G1.max(), G2.max(), 1e-6]))
                vert_map[i,j] = α*s_int + β*s_edge - γ*g_diff

            # horizontal seam
            if i+1 < n_rows:
                ny = y + stride
                neigh = img[ny:ny+patch_h, x:x+patch_w]
                gray2 = cv2.cvtColor(neigh, cv2.COLOR_RGB2GRAY)
                edges2 = cv2.Canny(gray2, 100, 200)
                gmag2 = grad_mag(gray2)

                T = gray[-seam_width: , :]
                B = gray2[:seam_width, :]
                E1 = edges[-seam_width: , :]
                E2 = edges2[:seam_width, :]
                G1 = gmag[-seam_width: , :]
                G2 = gmag2[:seam_width, :]

                s_int = ssim(T, B)
                s_edge = ssim(E1, E2)
                diff = np.abs(G1 - G2)
                g_diff = np.mean(diff) / (np.max([G1.max(), G2.max(), 1e-6]))
                hori_map[i,j] = α*s_int + β*s_edge - γ*g_diff

    # flatten valid scores
    all_Q = np.concatenate([
        vert_map[~np.isnan(vert_map)],
        hori_map[~np.isnan(hori_map)]
    ])
    mean_Q = np.mean(all_Q) if all_Q.size else np.nan
    rms_Q  = np.sqrt(np.mean(all_Q**2)) if all_Q.size else np.nan

    return vert_map, hori_map, mean_Q, rms_Q

def summarize_scores(all_m, combo_lambda=0.5, geom_shift=1.0, eps=1e-3):
    """
    all_m : 1D array of your seam scores (can be negative)
    Returns a dict with:
      - mean
      - minimum
      - combo = λ·min + (1−λ)·mean
      - geometric (with shift)
      - harmonic (with clamping)
    """
    N = all_m.size
    mean_s = np.mean(all_m)
    min_s  = np.min(all_m)

    # combined min+mean
    combo = combo_lambda*min_s + (1-combo_lambda)*mean_s

    # geometric mean with shift
    shifted = all_m + geom_shift
    # ensure non-negative
    if np.any(shifted <= 0):
        print("Need all (score+shift)>0 for geometric mean")
    geom = shifted.prod()**(1.0/N) - geom_shift

    # harmonic mean with clamped positives
    clamped = np.maximum(all_m, eps)
    harm = N / np.sum(1.0 / clamped)

    return {
        "mean": mean_s,
        "min": min_s,
        "combo": combo,
        "geo": geom,
        "harm": harm
    }

"""
def crop_black_borders(image, threshold=10):
    if image.ndim == 3:
        mask = image.max(axis=2) > threshold
    else:
        mask = image > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]
"""

def phase_seam_score(patch1, patch2, lam=1.0):
    """
    Compute phase correlation between two same-sized gray patches.
    Returns score = response - lam * (shift_mag / max_dim).
    """
    # convert to float and apply a window to reduce edge artefacts
    f1 = patch1.astype(np.float32)
    f2 = patch2.astype(np.float32)
    win = cv2.createHanningWindow(patch1.shape[::-1], cv2.CV_32F)
    f1 *= win
    f2 *= win

    # compute phase correlation
    (dx, dy), response = cv2.phaseCorrelate(f1, f2)
    shift_mag = np.hypot(dx, dy)
    max_dim = max(patch1.shape)
    score = response - lam * (shift_mag / max_dim)
    return score, response, (dx, dy)

def compute_phasecorr_scan(
    image_path: str,
    patch_h: int,
    patch_w: int,
    stride: int = 1,
    seam_width: int = 10,
    black_thresh: int = 10,
    lam: float = 1.0
):
    """
    Slides a patch_h×patch_w window across the image (by 'stride'),
    and for each vertical/horizontal seam, computes a phase-correlation
    based score.

    Returns:
      vert_map: (n_rows, n_cols-1) array of combined scores
      horiz_map: (n_rows-1, n_cols) array of combined scores
      mean_score: average of all valid M scores
      rms_score: RMS of all valid M scores
    """
    # 1) load & crop
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open {image_path!r}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img = crop_black_borders(img, black_threshold=black_thresh)

    if isinstance(img, Image.Image):
        img = np.array(img)
    
    H, W = img.shape

    # 2) sliding grid
    n_rows = (H - patch_h)//stride + 1
    n_cols = (W - patch_w)//stride + 1

    vert_map  = np.full((n_rows, n_cols-1),  np.nan, dtype=np.float32)
    horiz_map = np.full((n_rows-1, n_cols),  np.nan, dtype=np.float32)

    # 3) scan and compute
    for i in range(n_rows):
        y = i*stride
        for j in range(n_cols):
            x = j*stride
            patch = img[y:y+patch_h, x:x+patch_w]

            # vertical seam: patch vs. right neighbor
            if j+1 < n_cols:
                nx = x + stride
                neigh = img[y:y+patch_h, nx:nx+patch_w]
                L = patch[:, -seam_width:]
                R = neigh[:, :seam_width]
                m, resp, shift = phase_seam_score(L, R, lam=lam)
                vert_map[i, j] = m

            # horizontal seam: patch vs. bottom neighbor
            if i+1 < n_rows:
                ny = y + stride
                neigh = img[ny:ny+patch_h, x:x+patch_w]
                T = patch[-seam_width:, :]
                B = neigh[:seam_width, :]
                m, resp, shift = phase_seam_score(T, B, lam=lam)
                horiz_map[i, j] = m

    # aggregate
    all_m = np.concatenate([
        vert_map[~np.isnan(vert_map)],
        horiz_map[~np.isnan(horiz_map)]
    ])

    # Example usage within your scanning function:
    all_m = np.concatenate([vert_map[~np.isnan(vert_map)],
                            horiz_map[~np.isnan(horiz_map)]])
    scores = summarize_scores(all_m,
                              combo_lambda=0.5,
                              geom_shift=1.0,
                              eps=1e-3)

    return vert_map, horiz_map, scores

def warp_convex_quad_to_rect(image_path, points):
    """
    Warps a convex quadrilateral region in the image to a rectangle.

    Args:
        image_path (str): Path to the input image.
        points (list of tuple): 4 (x, y) tuples defining a convex quadrilateral.

    Returns:
        warped (np.ndarray): The resulting rectified rectangle image.
    """
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required.")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Convert input points to np.array
    pts_src = np.array(points, dtype=np.float32)

    # Compute bounding box size heuristically
    def side_lengths(pts):
        return [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]

    def estimate_dimensions(pts):
        lengths = side_lengths(pts)
        width = int((lengths[0] + lengths[2]) / 2)
        height = int((lengths[1] + lengths[3]) / 2)
        return width, height

    width, height = estimate_dimensions(pts_src)

    # Destination points for the warped rectangle
    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Compute homography and apply it
    H, _ = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, H, (width, height))

    return warped

def extract_corners_function(image_path: str) -> list:
    """
    Extracts 4 corner-like points of the purple tissue using shape approximation.
    Returns a list of 4 (x, y) tuples in top-left, top-right, bottom-right, bottom-left order.
    """

    def order_points(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]     # top-left
        ordered[2] = pts[np.argmax(s)]     # bottom-right
        ordered[1] = pts[np.argmin(diff)]  # top-right
        ordered[3] = pts[np.argmax(diff)]  # bottom-left
        return ordered

    def filter_unique_points(points, threshold=5):
        unique = []
        for p in points:
            if all(np.linalg.norm(np.array(p) - np.array(q)) > threshold for q in unique):
                unique.append(p)
        return unique

    # Load and convert to HSV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask purple regions
    lower_purple = np.array([110, 20, 20])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)

    # Try increasing approximation until 4 corners are found
    for epsilon_ratio in np.linspace(0.01, 0.1, 20):
        approx = cv2.approxPolyDP(largest, epsilon_ratio * peri, True)
        if len(approx) == 4:
            corners = [tuple(pt[0]) for pt in approx]
            return order_points(np.array(corners)).tolist()

    # Fallback to minAreaRect if approx fails
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    corners = [tuple(pt) for pt in box]
    corners = filter_unique_points(corners)

    if len(corners) != 4:
        raise ValueError("Could not extract 4 unique corners from fallback.")

    return order_points(np.array(corners)).tolist()

def full_pipeline(
    input_folder: str,
    output_folder: str,
    verbose: bool = False
) -> None:
    # Padding the images
    print("Padding the images...")
    os.makedirs(f"{output_folder}/padded_images", exist_ok=True)
    pad_images(
        input_folder = input_folder,
        output_folder = f"{output_folder}/padded_images",
    )
    print("COMPLETE\n")

    # Breaking down the images in patches
    print("Breaking down the images in patches...")
    os.makedirs(f"{output_folder}/patches", exist_ok=True)
    for filename in os.listdir(f"{output_folder}/padded_images"):
        split_image_into_patches(f"{output_folder}/padded_images/{filename}", f"{output_folder}/patches/{''.join(filename.split('.')[:-1])}_patches")
        rename_and_copy_images(f"{output_folder}/patches/{''.join(filename.split('.')[:-1])}_patches", f"{output_folder}/patches/{''.join(filename.split('.')[:-1])}_reordered_patches")

    print("COMPLETE\n")

    # Running KAZE
    print("Running KAZE...")

    os.makedirs(f"{output_folder}/KAZE", exist_ok=True)
    folder_paths = [path for path in os.listdir(f"{output_folder}/patches") if 'reordered' in path]

    for folder_name in tqdm(folder_paths, desc="Looping Through all patch-folders", position = 0):
        # The folder path without the "_reordered_patches" part
        os.makedirs(f"{output_folder}/KAZE/{folder_name[:-18]}", exist_ok=True)

        input_path = f"{output_folder}/patches/{folder_name}"
        output_path = f"{output_folder}/KAZE/{folder_name[:-18]}"
        paths = sorted(os.listdir(input_path))

        for i in tqdm(range(0, len(paths)), desc="Stitching all patches", position=1, leave=False):
            if i == 0:
                continue
            elif i == 1:
                try:
                    stitched_img_path = KAZE_matching(
                        img_path1=os.path.join(input_path, paths[i - 1]),
                        img_path2=os.path.join(input_path, paths[i]),
                        output_dir=output_path,
                        kaze_thresholding=0.7,
                        ransac_thresholding=100,
                        percentage_of_image_used=1.0
                    )
                except Exception as e:
                    print(f"KAZE failed after {i} stitchings.\nReason: {str(e)}")
                    break
                continue

            try:
                stitched_img_path = KAZE_matching(
                    img_path1=os.path.join(output_path, f"0000-{paths[i - 1][:-4]}_KAZE_0.7_1.0.jpg"),
                    img_path2=os.path.join(input_path, paths[i]),
                    output_dir=output_path,
                    kaze_thresholding=0.7,
                    ransac_thresholding=100,
                    percentage_of_image_used=1.0
                )
            except Exception as e:
                print(f"KAZE failed after {i} stitchings.\nReason: {str(e)}")
                break


    # Running SuperPoint
    print("Running SuperPoint...")

    os.makedirs(f"{output_folder}/SuperPoint", exist_ok=True)
    folder_paths = [path for path in os.listdir(f"{output_folder}/patches") if 'reordered' not in path]

    for folder_name in tqdm(folder_paths, desc="Looping Through all patch-folders", position = 0):
        # The folder path without the "_reordered_patches" part
        os.makedirs(f"{output_folder}/SuperPoint/{folder_name[:-8]}", exist_ok=True)

        input_path = f"{output_folder}/patches/{folder_name}"
        output_path = f"{output_folder}/SuperPoint/{folder_name[:-8]}"
        paths = sorted(os.listdir(input_path))
        paths = diagonal_submatrix(5, 0, 0, paths)

        # print(paths)

        for i in tqdm(range(0, len(paths)), desc="Stitching all patches", position=1, leave=False):
            if i == 0:
                continue
            elif i == 1:
                try:
                    stitched_img_path = LightGlue_matrix_scan_diagonal_matching(
                        img_path1 = os.path.join(input_path, paths[i - 1]),
                        img_path2 = os.path.join(input_path, paths[i]),
                        basepath_img2 = paths[i],
                        path_list = paths,
                        i_value = i,
                        matrix_size = 5,
                        output_dir=output_path,
                        ransac_thresholding=100,
                        percentage_of_image_used=1.0,
                        pair_extractor="SuperPoint",
                        verbose=verbose
                    )
                except Exception as e:
                    print(f"SuperPoint failed after {i} stitchings.\nReason: {str(e)}")
                    break
                continue

            try:
                stitched_img_path = LightGlue_matrix_scan_diagonal_matching(
                    img_path1 = os.path.join(output_path, f"0000-{paths[i - 1][:-4]}_SuperPoint_1.0.jpg"),
                    img_path2 = os.path.join(input_path, paths[i]),
                    basepath_img2 = paths[i],
                    path_list = paths,
                    i_value = i,
                    matrix_size = 5,
                    output_dir=output_path,
                    ransac_thresholding=100,
                    percentage_of_image_used=1.0,
                    pair_extractor="SuperPoint",
                    verbose=verbose
                )
            except Exception as e:
                print(f"SuperPoint failed after {i} stitchings.\nReason: {str(e)}")
                break

    print("COMPLETE\n")

    torch.cuda.empty_cache()

    # Running SIFT
    print("Running SIFT...")

    os.makedirs(f"{output_folder}/SIFT", exist_ok=True)
    folder_paths = [path for path in os.listdir(f"{output_folder}/patches") if 'reordered' not in path]

    for folder_name in tqdm(folder_paths, desc="Looping Through all patch-folders", position = 0):
        # The folder path without the "_reordered_patches" part
        os.makedirs(f"{output_folder}/SIFT/{folder_name[:-8]}", exist_ok=True)

        input_path = f"{output_folder}/patches/{folder_name}"
        output_path = f"{output_folder}/SIFT/{folder_name[:-8]}"
        paths = sorted(os.listdir(input_path))
        paths = diagonal_submatrix(5, 0, 0, paths)

        # print(paths)

        for i in tqdm(range(0, len(paths)), desc="Stitching all patches", position=1, leave=False):
            if i == 0:
                continue
            elif i == 1:
                try:
                    stitched_img_path = LightGlue_matrix_scan_diagonal_matching(
                        img_path1 = os.path.join(input_path, paths[i - 1]),
                        img_path2 = os.path.join(input_path, paths[i]),
                        basepath_img2 = paths[i],
                        path_list = paths,
                        i_value = i,
                        matrix_size = 5,
                        output_dir=output_path,
                        ransac_thresholding=100,
                        percentage_of_image_used=1.0,
                        pair_extractor="SIFT",
                        verbose=verbose
                    )
                except Exception as e:
                    print(f"SIFT failed after {i} stitchings.\nReason: {str(e)}")
                    break
                continue

            try:
                stitched_img_path = LightGlue_matrix_scan_diagonal_matching(
                    img_path1 = os.path.join(output_path, f"0000-{paths[i - 1][:-4]}_SIFT_1.0.jpg"),
                    img_path2 = os.path.join(input_path, paths[i]),
                    basepath_img2 = paths[i],
                    path_list = paths,
                    i_value = i,
                    matrix_size = 5,
                    output_dir=output_path,
                    ransac_thresholding=100,
                    percentage_of_image_used=1.0,
                    pair_extractor="SIFT",
                    verbose=verbose
                )
            except Exception as e:
                print(f"SIFT failed after {i} stitchings.\nReason: {str(e)}")
                break

    torch.cuda.empty_cache()

    # Homography the final images
    for folder_name in tqdm(folder_paths, desc="Looping Through all patch-folders", position = 0):
        input_path_to_kaze_image = f"{output_folder}/KAZE/{folder_name[:-8]}/0000-0024_KAZE_0.7_1.0.jpg"
        input_path_to_superpoint_image = f"{output_folder}/Superpoint/{folder_name[:-8]}/0000-0024_SuperPoint_1.0.jpg"
        input_path_to_sift_image = f"{output_folder}/SIFT/{folder_name[:-8]}/0000-0024_SIFT_1.0.jpg"

        output_path_to_kaze_image = f"{output_folder}/KAZE/{folder_name[:-8]}/homography_0000-0024_KAZE_0.7_1.0.jpg"
        output_path_to_superpoint_image = f"{output_folder}/Superpoint/{folder_name[:-8]}/homography_0000-0024_SuperPoint_1.0.jpg"
        output_path_to_sift_image = f"{output_folder}/SIFT/{folder_name[:-8]}/homography_0000-0024_SIFT_1.0.jpg"


        # points = [(1155, 442), (939, 4753), (6813, 4561), (7141, 245)]

        if os.path.exists(input_path_to_kaze_image):
            points = extract_corners_function(input_path_to_kaze_image)
            warped_image = warp_convex_quad_to_rect(input_path_to_kaze_image, points)
            cv2.imwrite(output_path_to_kaze_image, warped_image)

        if os.path.exists(input_path_to_superpoint_image):
            points = extract_corners_function(input_path_to_superpoint_image)
            warped_image = warp_convex_quad_to_rect(input_path_to_superpoint_image, points)
            cv2.imwrite(output_path_to_superpoint_image, warped_image)

        if os.path.exists(input_path_to_sift_image):
            points = extract_corners_function(input_path_to_sift_image)
            warped_image = warp_convex_quad_to_rect(input_path_to_sift_image, points)
            cv2.imwrite(output_path_to_sift_image, warped_image)

    # Calculating and storing metrics

    output_scores = []

    folder_paths = [path for path in os.listdir(f"{output_folder}/patches") if 'reordered' not in path]

    for folder_name in tqdm(folder_paths, desc="Looping Through all patch-folders", position = 0):
        dict_metrics = {
            "Image": folder_name[:-8]
        }

        path_to_kaze_image = f"{output_folder}/KAZE/{folder_name[:-8]}/homography_0000-0024_KAZE_0.7_1.0.jpg"
        path_to_superpoint_image = f"{output_folder}/Superpoint/{folder_name[:-8]}/homography_0000-0024_SuperPoint_1.0.jpg"
        path_to_sift_image = f"{output_folder}/SIFT/{folder_name[:-8]}/homography_0000-0024_SIFT_1.0.jpg"

        # KAZE IMAGE

        if os.path.exists(path_to_kaze_image):
            avg_ssim, _ = compute_stitched_ssim(path_to_kaze_image, 5, seam_width=50, black_thresh=10)
            _, _, μ, _ = compute_improved_scan_ssim(
                path_to_kaze_image,
                patch_h=200, patch_w=200,
                stride=50, seam_width=10,
                black_thresh=10,
                α=0.2, β=0.4, γ=0.4
            )
            _, _, scores = compute_phasecorr_scan(
                path_to_kaze_image,
                patch_h   = 200,
                patch_w   = 200,
                stride    = 50,
                seam_width= 10,
                black_thresh=10,
                lam=1.0
            )
            phase_corr_score = scores["mean"]

            dict_metrics["KAZE"] = {
                "SSIM_score": float(avg_ssim),
                "MQ_score":float(μ),
                "Phase_Correlation_score": float(phase_corr_score)
            }
        else:
            dict_metrics["KAZE"] = {
                "SSIM_score": -1,
                "MQ_score": -1,
                "Phase_Correlation_score": -1
            }

        # SuperPoint IMAGE

        if os.path.exists(path_to_superpoint_image):
            avg_ssim, _ = compute_stitched_ssim(path_to_superpoint_image, 5, seam_width=50, black_thresh=10)
            _, _, μ, _ = compute_improved_scan_ssim(
                path_to_superpoint_image,
                patch_h=200, patch_w=200,
                stride=50, seam_width=10,
                black_thresh=10,
                α=0.2, β=0.4, γ=0.4
            )
            _, _, scores = compute_phasecorr_scan(
                path_to_superpoint_image,
                patch_h   = 200,
                patch_w   = 200,
                stride    = 50,
                seam_width= 10,
                black_thresh=10,
                lam=1.0
            )
            phase_corr_score = scores["mean"]

            dict_metrics["SuperPoint"] = {
                "SSIM_score": float(avg_ssim),
                "MQ_score":float(μ),
                "Phase_Correlation_score": float(phase_corr_score)
            }
        else:
            dict_metrics["SuperPoint"] = {
                "SSIM_score": -1,
                "MQ_score": -1,
                "Phase_Correlation_score": -1
            }

        # SIFT IMAGE

        if os.path.exists(path_to_sift_image):
            avg_ssim, _ = compute_stitched_ssim(path_to_sift_image, 5, seam_width=50, black_thresh=10)
            _, _, μ, _ = compute_improved_scan_ssim(
                path_to_sift_image,
                patch_h=200, patch_w=200,
                stride=50, seam_width=10,
                black_thresh=10,
                α=0.2, β=0.4, γ=0.4
            )
            _, _, scores = compute_phasecorr_scan(
                path_to_sift_image,
                patch_h   = 200,
                patch_w   = 200,
                stride    = 50,
                seam_width= 10,
                black_thresh=10,
                lam=1.0
            )
            phase_corr_score = scores["mean"]

            dict_metrics["SIFT"] = {
                "SSIM_score": float(avg_ssim),
                "MQ_score":float(μ),
                "Phase_Correlation_score": float(phase_corr_score)
            }
            print(dict_metrics["SIFT"])
        else:
            dict_metrics["SIFT"] = {
                "SSIM_score": -1,
                "MQ_score": -1,
                "Phase_Correlation_score": -1
            }

        print(dict_metrics)

        output_scores.append(dict_metrics)

    with open(f"{output_folder}/output.json", "w", encoding="utf-8") as g:
        json.dump(output_scores, g, ensure_ascii=False, indent=2)
    print("COMPLETE\n")

full_pipeline(
    input_folder = INPUT_FOLDER,
    output_folder = OUTPUT_FOLDER,
    verbose = False
)