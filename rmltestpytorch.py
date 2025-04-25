"""
PyTorch re‑implementation of the Keras/TensorFlow RML2018 CNN
that now matches layer‑by‑layer, tensor shapes, and pooling
schedule one‑for‑one with the original model.
"""

# ——————————————————————— imports & global setup —————————————————————— #
import os, random, h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ——————————————————————— data loading ——————————————————————— #
FILE = "/home/017448899/RML2018/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

def load_hdf5(fname):
    with h5py.File(fname, "r") as f:
        k0, k1, k2 = list(f.keys())        # 'X', 'Y', 'Z'
        X = f[k0][:]                       # (N, 1024, 2) float64
        Y = f[k1][:]                       # (N, 24)      one‑hot
        Z = f[k2][:]                       # (N,)         SNR
    return X, Y, Z

X, Y, Z = load_hdf5(FILE)

# match TensorFlow slice: keep SNR ≥ ‑14 dB
sel = np.where(Z >= -14)[0]
X, Y = X[sel].astype(np.float32), Y[sel].astype(np.float32)

# ———————————————————— stratified split —————————————————————— #
def stratified_bool_split(y_onehot, train_fraction=0.5):
    """Return boolean masks for train / test using exact TF logic."""
    y = y_onehot
    m_train = np.zeros(len(y), dtype=bool)
    m_test  = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y, axis=0):
        idx = np.flatnonzero((y == cls).all(axis=1))
        np.random.shuffle(idx)
        n = int(train_fraction * len(idx))
        m_train[idx[:n]] = True
        m_test[idx[n:]]  = True
    return m_train, m_test

m_train, m_test = stratified_bool_split(Y)

X_tr, X_te = X[m_train], X[m_test]
Y_tr, Y_te = Y[m_train], Y[m_test]
in_shp = list(X_tr.shape[1:])

print("X_train.shape: {}".format(X_tr.shape))
print("Y_train.shape: {}".format(Y_tr.shape))
print("X_test.shape: {}".format(X_te.shape))
print("Y_test.shape: {}".format(Y_te.shape))
print("input shape: {}".format(in_shp))

# ——————————————————————— dataset & loader —————————————————————— #
class SignalDataset(Dataset):
    """
    • Input  shape expected by network: (N, 1, 2, 1024)
      (channels = 1, height = 2 (I/Q), width = 1024 samples)
    • Labels provided as integer class indices (CrossEntropyLoss).
    """
    def __init__(self, X, Y):
        # (N, 1024, 2) → (N, 1, 2, 1024)
        self.X = torch.from_numpy(X).permute(0, 2, 1).unsqueeze(1)
        self.y = torch.from_numpy(np.argmax(Y, axis=1)).long()

    def __len__(self):            return len(self.X)
    def __getitem__(self, idx):   return self.X[idx], self.y[idx]

BATCH = 32
train_loader = DataLoader(SignalDataset(X_tr, Y_tr),
                          batch_size=BATCH, shuffle=True, drop_last=True)
test_loader  = DataLoader(SignalDataset(X_te, Y_te),
                          batch_size=BATCH, shuffle=False)

NUM_CLASSES = Y.shape[1]          # 24

# ———————————————————————— model ————————————————————————— #
class SignalCNN(nn.Module):
    def __init__(self, n_classes, in_shape=(1, 2, 1024), drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (1, 3), padding=(0, 1))
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.bn5   = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.bn6   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d((1, 2))
        self.avgpool = nn.AvgPool2d((1, 16))      # width 32 → 2

        self.flat  = nn.Flatten()
        # infer flattened feature size once
        with torch.no_grad():
            dummy = torch.zeros(1, *in_shape)
            feat  = self._forward_features(dummy)
            self.feat_dim = feat.shape[1]

        self.fc1   = nn.Linear(self.feat_dim, 128)
        self.drop  = nn.Dropout(drop)
        self.fc2   = nn.Linear(128, n_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.avgpool(F.relu(self.bn6(self.conv6(x))))
        return self.flat(x)

    def forward(self, x):
        x = self._forward_features(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model = SignalCNN(NUM_CLASSES).to(device)
print(model)

# ———————————————————————— training —————————————————————— #
LR        = 1e-3
EPOCHS    = 100
PATIENCE  = 10

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")
wait      = 0

for epoch in range(1, EPOCHS + 1):
    # ─── train ─────────────────────────────────────────────── #
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    # ─── validate ─────────────────────────────────────────── #
    model.eval()
    val_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out    = model(xb)
            val_loss += criterion(out, yb).item()
            preds.append(out.argmax(1).cpu())
            labels.append(yb.cpu())
    val_loss /= len(test_loader)
    val_acc   = accuracy_score(torch.cat(labels), torch.cat(preds))

    print(f"Epoch {epoch:03d}:  val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    # ─── early stopping ───────────────────────────────────── #
    if val_loss < best_loss:
        best_loss = val_loss
        wait      = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("⏹️ Early stopping")
            break

# ——————————————————————— evaluation ———————————————————— #
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_loss, preds, labels = 0.0, [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        test_loss += loss.item()
        preds.append(out.argmax(1).cpu())
        labels.append(yb.cpu())

test_loss /= len(test_loader)
test_acc = accuracy_score(torch.cat(labels), torch.cat(preds))

print("loss: {:.4f}".format(test_loss))
print("accuracy: {:.4f}".format(test_acc))