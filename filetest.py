import os, random, pickle, sys
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict

# ----------------------------
# Load Data
# ----------------------------
filename = "/home/017448899/RML2018/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

def load_hdf5(path: str):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())          # typically ['X', 'Y', 'Z']
        X = f[keys[0]][:]              # IQ samples: (N, 1024, 2)
        Y = f[keys[1]][:]              # one-hot labels: (N, 24)
        Z = f[keys[2]][:].squeeze()    # SNR: (N,)
    return X, Y, Z

X, Y_onehot, Z = load_hdf5(filename)

# Print the first sample
print("🧠 First IQ sample (X[0]):")
print(X[0])  # Shape: (1024, 2)

print("\n🏷 Label (one-hot, Y[0]):")
print(Y_onehot[0])  # Shape: (24,)

print("\n📶 SNR value (Z[0]):")
print(Z[0])  # Single float value

# # ----------------------------
# # Map Class Indices to Labels
# # ----------------------------
# mod_classes = [
#     '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
#     'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
#     '128APSK','AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
#     'AM-DSB-WC', 'OOK', '16QAM'
# ]
# # Assumes class ordering matches the one-hot indices
# class_idx_to_label = {i: label for i, label in enumerate(mod_classes)}

# # ----------------------------
# # Build (modulation, SNR) -> list of IQ samples
# # ----------------------------
# iq_by_mod_snr = defaultdict(list)

# for i in range(len(X)):
#     class_index = np.argmax(Y_onehot[i])
#     mod_label = class_idx_to_label[class_index]
#     snr = int(Z[i])
#     key = (mod_label, snr)
#     # Transpose shape from (1024, 2) -> (2, 1024) so channels come first
#     iq_by_mod_snr[key].append(X[i].T)

# # Optional: convert lists to numpy arrays
# iq_by_mod_snr = {k: np.array(v) for k, v in iq_by_mod_snr.items()}

# # Example usage
# print(f"✅ Total unique (modulation, SNR) combinations: {len(iq_by_mod_snr)}")
# sample_key = ('QPSK', 10)
# print(f"Shape of IQ data for {sample_key}: {iq_by_mod_snr[sample_key].shape}")

# output_path = "iq_by_mod_snr.pkl"

# with open(output_path, "wb") as f:
#     pickle.dump(iq_by_mod_snr, f)

# print(f"✅ Saved dictionary to: {output_path}")