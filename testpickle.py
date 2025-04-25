import pickle
import h5py
from data import build_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
# Open the file in binary read mode
from torch.utils.data import Dataset, DataLoader
with open('/home/017448899/RML2018/RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # or try encoding='utf-8' if needed

# Now `data` contains whatever was stored in the pickle file
print(type(data))
print(data.keys())
print(len(data.keys()))
# train, val, test, le = build_dataset(dataset_name="RML2016.10a", path='/home/017448899/RML2018/RML2016.10a_dict.pkl')
# print('classes: ', len(le))

# filename = "/home/017448899/RML2018/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
# def load_hdf5(path: str):
#     with h5py.File(path, "r") as f:
#         keys = list(f.keys())          # e.g. ['X', 'Y', 'Z']
#         print(keys)
#         X = f[keys[0]][:]
#         Y = f[keys[1]][:]
#         Z = f[keys[2]][:].squeeze()    # <- flatten here
#     return X, Y, Z

# X, Y_onehot, Z = load_hdf5(filename)
# mask = Z >= -14
# X, Y_onehot, Z = X[mask], Y_onehot[mask], Z[mask]
# print("After SNR filter :", X.shape)
# print("Loaded:", X.shape, Y_onehot.shape, Z.shape)

X_list = []
Y_list = []
label_map = {}  # optional: maps numeric ID -> (mod, SNR)
visited = set()
mod_to_idx = {}   
for idx, (label, samples) in enumerate(data.items()):
    #print(label, samples.shape)
    visited.add(label[0])
    
print("visited:", visited)
for label_tuple, samples in data.items():
    mod_type = label_tuple[0]  # drop SNR

    # Skip if not in the visited (target) set
    if mod_type not in visited:
        continue

    # Assign class index if not already mapped
    if mod_type not in mod_to_idx:
        mod_to_idx[mod_type] = idx_counter
        label_map[idx_counter] = mod_type
        idx_counter += 1

    label_id = mod_to_idx[mod_type]

    X_list.append(samples)
    Y_list.append(np.full(samples.shape[0], label_id))  # Assign label_id to all 1000 samples

X = np.concatenate(X_list, axis=0)  # shape (N_total, 2, 128)
Y = np.concatenate(Y_list, axis=0)  # shape (N_total,)
n_classes = len(label_map)
print(X.shape, Y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print("Train:", X_train.shape, " Test:", X_test.shape)

class RMLDataset2(Dataset):
    def __init__(self, X_arr, y_arr):
        # X_arr : (N, 1024, 2)   => torch (N, 2, 1, 1024)
        self.X = X_arr.astype(np.float32)
        self.y = y_arr.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(1)   # (2,1,1024)
        y = torch.tensor(self.y[idx])
        return x, y

batch_size = 32
train_ds   = RMLDataset2(X_train, y_train)
test_ds    = RMLDataset2(X_test,  y_test)
train_dl   = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True,  pin_memory=True)
test_dl    = DataLoader(test_ds,  batch_size=batch_size,
                        shuffle=False, pin_memory=True)

xb_chk, _ = next(iter(train_dl))
print("💡 first batch shape :", xb_chk.shape)   # expect (32, 2, 1, 1024)