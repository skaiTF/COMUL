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


def extract_features(image):
    img = image.numpy()
    features = []
 
    # Colour histograms: 16 bins x 3 channels = 48 values
    for c in range(3):
        hist, _ = np.histogram(img[:, :, c], bins=16, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8
        features.append(hist)
 
    # Downscaled grayscale 16x16 = 256 values
    gray = 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]
    features.append(resize(gray, (16, 16), anti_aliasing=True).astype(np.float32).flatten())
 
    return torch.tensor(np.concatenate(features))
 
 
print("\n── Extracting features ───────────────────────────────────────────────────")
train_feats = []
for i, img in enumerate(X_train):
    train_feats.append(extract_features(img))
    sys.stdout.write(f"  Train: {i+1}/{len(X_train)}\r")
    sys.stdout.flush()
X_train_feat = torch.stack(train_feats)
print()

test_feats = []
for i, img in enumerate(X_test):
    test_feats.append(extract_features(img))
    sys.stdout.write(f"  Test: {i+1}/{len(X_test)}\r")
    sys.stdout.flush()
X_test_feat = torch.stack(test_feats)
print(f"\nFeature vector size: {X_train_feat.shape[1]}")
 
# Save features to .pkl
with open("data_features.pkl", "wb") as f:
    pickle.dump(((X_train_feat, y_train), (X_test_feat, y_test), label_dict), f)
print("Features saved to 'data_features.pkl'")
 

feat_mean = X_train_feat.mean(0)
feat_std  = X_train_feat.std(0) + 1e-8
X_train_norm = (X_train_feat - feat_mean) / feat_std
X_test_norm  = (X_test_feat  - feat_mean) / feat_std
 
X_tr = torch.cat([torch.ones(X_train_norm.shape[0], 1), X_train_norm], dim=1)
X_te = torch.cat([torch.ones(X_test_norm.shape[0],  1), X_test_norm],  dim=1)
 
y_train_oh = F.one_hot(y_train, num_classes=4).float()
y_test_oh  = F.one_hot(y_test,  num_classes=4).float()
 
X_tr, X_te             = X_tr.to(device), X_te.to(device)
y_train_oh, y_test_oh  = y_train_oh.to(device), y_test_oh.to(device)
 
class LogisticRegression(torch.nn.Module):
    def __init__(self, no_params, no_classes):
        super(LogisticRegression, self).__init__()
        self.theta = torch.randn(no_classes, no_params, requires_grad=True)
        self.no_classes = no_classes
        self.no_params = no_params
 
    def forward(self, input):
        z = input @ self.theta.t()
        if self.no_classes > 1:
            return self.softmax(z.squeeze())
        else:
            return self.sigmoid(z.squeeze())
 
    def _reset_weights(self):
        self.theta = torch.randn(self.no_classes, self.no_params, requires_grad=True)
 
    def sigmoid(self, z):
        return 1/(1 + torch.exp(-z))
 
    def softmax(self, z):
        return torch.exp(z) / torch.exp(z).sum(1).unsqueeze(1)
 
 
def loss_fn_multiclass(y_true, y_pred):
    return -(y_true * torch.log(y_pred + 1e-9)).sum(1).mean()
 
def accuracy(y_true, y_pred):
    return (torch.argmax(y_true, 1) == torch.argmax(y_pred, 1)).float().mean()
 
 
print("\n── Training ──────────────────────────────────────────────────────────────")
 
steps = 2000
lr    = 0.1
 
log_reg = LogisticRegression(X_tr.shape[1], y_train_oh.shape[1])
log_reg._reset_weights()
log_reg.theta = log_reg.theta.to(device).detach().requires_grad_(True)
 
train_losses, test_losses, train_accs, test_accs = [], [], [], []
 
for i in range(steps):
    train_pred = log_reg(X_tr)
    train_loss = loss_fn_multiclass(y_train_oh, train_pred)
    train_loss.backward()
 
    with torch.no_grad():
        log_reg.theta -= lr * log_reg.theta.grad
        log_reg.theta.grad.zero_()
 
    train_acc = accuracy(y_train_oh, train_pred.detach())
 
    with torch.no_grad():
        test_pred = log_reg(X_te)
        test_loss = loss_fn_multiclass(y_test_oh, test_pred)
        test_acc  = accuracy(y_test_oh, test_pred)
 
    train_losses.append(train_loss.detach().item())
    test_losses.append(test_loss.detach().item())
    train_accs.append(train_acc.item())
    test_accs.append(test_acc.item())
 
    sys.stdout.write(
        f"iter {i+1}/{steps} - "
        f"train_loss: {train_loss.item():.5f}, train_acc: {train_acc:.3f} - "
        f"test_loss: {test_loss.item():.5f}, test_acc: {test_acc:.3f}\r"
    )
    sys.stdout.flush()
 
print(f"\nFinal test accuracy: {test_accs[-1]*100:.1f}%")
 

 
# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train NLL")
ax1.plot(test_losses,  label="Test NLL")
ax1.set_title("Loss (Negative Log-Likelihood)")
ax1.set_xlabel("Iteration"); ax1.set_ylabel("NLL")
ax1.legend(); ax1.grid(True)
 
ax2.plot([a*100 for a in train_accs], label="Train Accuracy")
ax2.plot([a*100 for a in test_accs],  label="Test Accuracy")
ax2.set_title("Accuracy (%)")
ax2.set_xlabel("Iteration"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(); ax2.grid(True)
 
plt.suptitle("Multiclass Logistic Regression — Simpsons Dataset", fontsize=13)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
 
# Confusion matrix
with torch.no_grad():
    pred_labels = log_reg(X_te).argmax(1).cpu().numpy()
    true_labels = y_test.cpu().numpy()
 
conf = np.zeros((4, 4), dtype=int)
for t, p in zip(true_labels, pred_labels):
    conf[t, p] += 1
 
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(4)); ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Test Set")
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                color="white" if conf[i, j] > conf.max()/2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
 
print("\nPer-class accuracy:")
for name, acc in zip(class_names, conf.diagonal() / conf.sum(axis=1)):
    print(f"  {name:<10}: {acc*100:.1f}%")
