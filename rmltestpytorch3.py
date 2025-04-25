# rml2018_pytorch.py
# ------------------------------------------------------------
# PyTorch equivalent of the original Keras/TensorFlow pipeline
# ------------------------------------------------------------
import os, random, pickle, sys
import numpy as np
import h5py
from pathlib import Path
from sklearn.metrics import accuracy_score
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns                       # pip install seaborn if needed
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# 1.  Environment & GPU setup
# ----------------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Using device : {device}")

# -------------------------------
# 2.  Load the HDF5 RML2018 file
# -------------------------------
filename = "/home/017448899/RML2018/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

def load_hdf5(path: str):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())          # e.g. ['X', 'Y', 'Z']
        X = f[keys[0]][:]
        Y = f[keys[1]][:]
        Z = f[keys[2]][:].squeeze()    # <- flatten here
    return X, Y, Z

X, Y_onehot, Z = load_hdf5(filename)
print("Loaded:", X.shape, Y_onehot.shape, Z.shape)

# ---------------------------
# 3.  Class names / filtering
# ---------------------------
classes = np.array([
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
    'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
    '128APSK','AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
    'AM-DSB-WC', 'OOK', '16QAM'
])
easy_classes = np.array([22, 11, 8, 19, 9, 23, 10, 16, 3, 4, 6])
n_classes = len(classes)

# Keep only samples with SNR ≥ ‑14 dB
mask = Z >= -14
X, Y_onehot, Z = X[mask], Y_onehot[mask], Z[mask]
print("After SNR filter :", X.shape)

# Convert one‑hot → integer labels once, so PyTorch can use CrossEntropyLoss
y_labels = Y_onehot.argmax(axis=1).astype(np.int64)

# ------------------------------------------------------------
# 4.  Stratified train / test split   (same helper as before)
# ------------------------------------------------------------
def stratified_bool_split(y, train_proportion=0.5):
    y = np.asarray(y)
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask  = np.zeros(len(y), dtype=bool)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        n_train = int(train_proportion * len(idx))
        train_mask[idx[:n_train]] = True
        test_mask[idx[n_train:]] = True
    return train_mask, test_mask

train_idx, test_idx = stratified_bool_split(Y_onehot, 0.5)

X_train, X_test   = X[train_idx],  X[test_idx]
y_train, y_test   = y_labels[train_idx], y_labels[test_idx]

print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------------------------------------
# 5.  PyTorch Dataset & DataLoader layer
# ---------------------------------------
class RMLDataset(Dataset):
    def __init__(self, X_arr, y_arr):
        # X_arr : (N, 1024, 2)   => torch (N, 2, 1, 1024)
        self.X = X_arr.astype(np.float32)
        self.y = y_arr.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).T.unsqueeze(1)   # (2,1,1024)
        y = torch.tensor(self.y[idx])
        return x, y

batch_size = 32
train_ds   = RMLDataset(X_train, y_train)
test_ds    = RMLDataset(X_test,  y_test)
train_dl   = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True,  pin_memory=True)
test_dl    = DataLoader(test_ds,  batch_size=batch_size,
                        shuffle=False, pin_memory=True)

xb_chk, _ = next(iter(train_dl))
print("💡 first batch shape :", xb_chk.shape)   # expect (32, 2, 1, 1024)

# ------------------------------
# 6.  CNN architecture (PyTorch)
# ------------------------------
class RMLNet(nn.Module):
    def __init__(self, n_classes=24, dropout=0.5):
        super().__init__()
        def conv_block(in_ch):
            block = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=(1,3), padding=(0,1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            )
            return block

        self.model = nn.Sequential(
            conv_block(2),   # input channels = 2
            conv_block(64),
            conv_block(64),
            conv_block(64),
            conv_block(64),
            nn.Sequential(                       # 6th conv + AvgPool
                nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=(1,16))
            ),
            nn.Flatten(),
            nn.Linear(64 * 1 * (1024 // (2**5) // 16), 128),  # matches Keras
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.model(x)

net = RMLNet(n_classes=n_classes, dropout=0.5).to(device)
def init_weights_keras(m):
    if isinstance(m, nn.Conv2d):
        # Keras conv default = glorot_uniform
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # Keras dense default = he_normal (fan_out, relu)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(m.bias)

for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.99          # Keras: momentum (1‑decay)
        m.eps      = 1e-3          # Keras: epsilon

net.apply(init_weights_keras)
print(net)

# --------------------------
# 7.  Loss, optimiser, etc.
# --------------------------
lr = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

# Simple early‑stopping & checkpoint
best_val_loss = np.inf
patience      = 5
patience_cnt  = 0
ckpt_path     = Path("rml2018_best.pt")

# -----------------------
# 8.  Training loop
# -----------------------
def run_epoch(dataloader, training=True):
    if training:
        net.train()
    else:
        net.eval()

    total_loss, n_obs = 0.0, 0
    all_preds, all_true = [], []

    with torch.set_grad_enabled(training):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            logits = net(xb)             # (B, n_classes)
            loss   = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_obs      += xb.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_true.append(yb.cpu())

    avg_loss = total_loss / n_obs
    acc      = accuracy_score(torch.cat(all_true),
                              torch.cat(all_preds))
    return avg_loss, acc

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = run_epoch(train_dl, training=True)
    val_loss,   val_acc   = run_epoch(test_dl,  training=False)

    print(f"Epoch {epoch:3d} | "
          f"train loss {train_loss:7.4f} acc {train_acc:6.3f} | "
          f"val loss {val_loss:7.4f} acc {val_acc:6.3f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_cnt  = 0
        torch.save(net.state_dict(), ckpt_path)
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("⏹  Early stopping triggered.")
            break

# ---------------------------
# 9.  Final evaluation
# ---------------------------
net.load_state_dict(torch.load(ckpt_path, map_location=device))
test_loss, test_acc = run_epoch(test_dl, training=False)
print(f"\nTest‑set  loss : {test_loss:.4f} | accuracy : {test_acc:.4f}")


sns.set(style="whitegrid")                  # nicer heat‑maps

num_classes     = len(classes)
snrs            = np.unique(Z)              # or any subset you prefer
easy_set        = set(easy_classes.tolist())
batch_eval_size = 256                       # bigger batch is fine for pure inference

net.eval()
with torch.no_grad():
    for snr in snrs:

        # ---------------- Filter the global TEST split ----------------
        snr_mask      = (Z[test_idx] == snr)
        X_snr         = X_test[snr_mask]
        y_snr_int     = y_test[snr_mask]              # int labels 0‑23

        easy_mask     = np.isin(y_snr_int, easy_classes)
        X_snr         = X_snr[easy_mask]
        y_snr_int     = y_snr_int[easy_mask]

        if X_snr.size == 0:
            print(f"⚠️  No easy‑class examples at SNR {snr} dB — skipping.")
            continue

        # ---------------- Batch inference in PyTorch ------------------
        snr_ds  = RMLDataset(X_snr, y_snr_int)
        snr_dl  = DataLoader(snr_ds, batch_size=batch_eval_size,
                             shuffle=False, pin_memory=True)

        preds = []
        for xb, _ in snr_dl:
            xb = xb.to(device)
            logits = net(xb)
            preds.append(logits.argmax(1).cpu().numpy())
        y_pred_int = np.concatenate(preds)

        # ---------------- Confusion matrix (easy classes only) --------
        full_conf = confusion_matrix(
            y_snr_int, y_pred_int, labels=np.arange(n_classes), normalize=None
        )
        # Re‑order & slice the “easy” classes exactly like original TF code
        conf      = full_conf[np.ix_(easy_classes, easy_classes)]
        conf_norm = conf / conf.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_norm,
                    annot=False,
                    cmap="Blues",
                    square=True,
                    cbar=True,
                    xticklabels=classes[easy_classes],
                    yticklabels=classes[easy_classes],
                    ax=ax)
        ax.set_title(f"Confusion Matrix (SNR = {snr} dB)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fname = f"confmat_snr_{snr}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"✅  Saved {fname}")

        # ---------------- Classification report -----------------------
        true_lbls  = classes[y_snr_int]
        pred_lbls  = classes[y_pred_int]

        print(f"\n📊  Classification report for SNR {snr} dB (easy classes):")
        print(classification_report(true_lbls, pred_lbls,
                                    zero_division=0,
                                    labels=classes[easy_classes]))