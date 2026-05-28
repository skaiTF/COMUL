# Imports

import os
import sys
import pickle
from glob import glob
from collections import Counter
 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage import io

# Seed initialization
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

path = "/home/skaideer/COMUL/Lab 5 - Introduction to Machine Learning/data/simpsons_split"

label_dict = {
    "bart_simpson":  0,
    "homer_simpson": 1,
    "marge_simpson": 2,
    "lisa_simpson":  3,
}
inv_label_dict = {v: k for k, v in label_dict.items()}
class_names = ["Bart", "Homer", "Marge", "Lisa"]

def load_image(f, size=(128, 128)):
    img = io.imread(f)
    if img.ndim == 3 and img.shape[2] == 4:  # drop alpha channel
        img = img[:, :, :3]
    if img.ndim == 2:                         # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    return torch.tensor(resize(img, size, anti_aliasing=True), dtype=torch.float32)

def load_label(f):
    return os.path.split(os.path.split(f)[0])[1]

def load_split(split):
    split_path = os.path.join(path, split)
    files = sorted(glob(os.path.join(split_path, "**/*"), recursive=True))
    files = [f for f in files if not os.path.isdir(f)]
    print(f"Found {len(files)} files in '{split}'")

    images, labels = [], []
    for i, f in enumerate(files):
        label_name = load_label(f)
        if label_name not in label_dict:
            continue
        images.append(load_image(f))
        labels.append(label_dict[label_name])
        sys.stdout.write(f"  Loading {split}: {i+1}/{len(files)}\r")
        sys.stdout.flush()
    print()
    return torch.stack(images), torch.tensor(labels, dtype=torch.int64)


X_train, y_train = load_split("train")
X_test,  y_test  = load_split("test")
print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

def count_occurences(split):
    split_path = os.path.join(path, split)
    files = glob(os.path.join(split_path, "**/*"), recursive=True)
    files = [f for f in files if not os.path.isdir(f)]
    counts = Counter(load_label(f) for f in files if load_label(f) in label_dict)
    return counts, len(files)

train_counts, train_total = count_occurences("train")
test_counts, test_total = count_occurences("test")

header = f"{'Character':<20} {'Train':>8} {'Train %':>9} {'Test':>8} {'Test %':>9} {'Total':>8}"
print(header)
print("-" * len(header))
for char in label_dict:
    tr  = train_counts.get(char, 0)
    te  = test_counts.get(char, 0)
    print(f"{char:<20} {tr:>8} {tr/train_total*100:>8.1f}% {te:>8} {te/test_total*100:>8.1f}% {tr+te:>8}")
print("-" * len(header))
print(f"{'TOTAL':<20} {train_total:>8} {'100.0%':>9} {test_total:>8} {'100.0%':>9} {train_total+test_total:>8}")

