import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _img_path(data_dir: str, path: str) -> str:
    return os.path.join(data_dir, "IMG", os.path.basename(str(path).strip()))


def load_driving_log(csv_path: str, data_dir: str) -> pd.DataFrame:
    """Load driving log and normalize paths for center/left/right cameras."""
    cols = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    df = pd.read_csv(csv_path, names=cols)
    for col in ["center", "left", "right"]:
        df[col] = df[col].apply(lambda p: _img_path(data_dir, p))
    df["steering"] = df["steering"].astype(np.float32)
    return df[["center", "left", "right", "steering"]]

def clip_steering(df: pd.DataFrame, clip: float = 0.8) -> pd.DataFrame:
    df = df.copy()
    df["steering"] = df["steering"].clip(-clip, clip)
    return df

def balance_steering(df: pd.DataFrame, bins: int = 25, samples_per_bin: int = 800) -> pd.DataFrame:
    hist, bin_edges = np.histogram(df["steering"], bins=bins)
    keep_indices = []
    for i in range(bins):
        idx = np.where((df["steering"] >= bin_edges[i]) & (df["steering"] < bin_edges[i + 1]))[0]
        if len(idx) > samples_per_bin:
            idx = np.random.choice(idx, samples_per_bin, replace=False)
        keep_indices.extend(idx)
    return df.iloc[keep_indices].reset_index(drop=True)

def plot_steering_histogram(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(df["steering"], bins=31, edgecolor="k")
    plt.xlabel("Steering")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    # Crop sky/hood
    cropped = image_rgb[60:-25, :, :]
    yuv = cv2.cvtColor(cropped, cv2.COLOR_RGB2YUV)
    blur = cv2.GaussianBlur(yuv, (3, 3), 0)
    resized = cv2.resize(blur, (200, 66))
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def random_flip(image: np.ndarray, steering: float):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering

def random_translate(image: np.ndarray, steering: float, range_x=35, range_y=8, steering_scale=0.002):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering += trans_x * steering_scale
    trans_mat = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_mat, (width, height))
    return image, steering

def random_brightness(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_rotate(image: np.ndarray, steering: float, max_deg: float = 6.0):
    angle = max_deg * (np.random.rand() - 0.5) * 2
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, (w, h))
    steering += angle / 25.0
    return image, steering

def augment(image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    image, steering = random_flip(image, steering)
    image, steering = random_translate(image, steering)
    image, steering = random_rotate(image, steering)
    image = random_brightness(image)
    return image, steering

def batch_generator(
    df: pd.DataFrame,
    batch_size: int,
    is_training: bool = True,
    augment_prob: float = 0.3,
    steering_correction: float = 0.12,
):
    #Yield batches; randomly sample center/left/right with steering correction.
    num = len(df)
    while True:
        indices = np.random.permutation(num)
        for start in range(0, num, batch_size):
            batch_idx = indices[start:start + batch_size]
            images, steers = [], []
            for i in batch_idx:
                row = df.iloc[i]
                # Build candidate cameras present in this row
                candidates = []
                if pd.notna(row["center"]):
                    candidates.append((row["center"], float(row["steering"])))
                if pd.notna(row["left"]):
                    candidates.append((row["left"], float(row["steering"] + steering_correction)))
                if pd.notna(row["right"]):
                    candidates.append((row["right"], float(row["steering"] - steering_correction)))
                if not candidates:
                    continue

                # Favor center frames but still sample side cams for recovery behavior
                weights = []
                for idx in range(len(candidates)):
                    weights.append(0.7 if idx == 0 else 0.15)
                probs = np.array(weights) / np.sum(weights)
                choice_idx = np.random.choice(len(candidates), p=probs)
                img_path, angle = candidates[choice_idx]

                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if is_training and random.random() < augment_prob:
                    img, angle = augment(img, angle)
                img = preprocess(img)
                images.append(img)
                steers.append(angle)
            if images:
                yield np.asarray(images, dtype=np.float32), np.asarray(steers, dtype=np.float32)
