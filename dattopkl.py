import pickle

# Load the pickle file
with open('RML2016.10b.dat', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Print all keys and value shapes
for key, value in data.items():
    print(f"Key: {key}, Value shape: {value.shape}")

# Extract unique modulation types
unique_modulations = set(k[0] for k in data.keys())
print("\nUnique modulations:", unique_modulations)

# --- Print 1 data sample ---

# Pick one key
sample_key = list(data.keys())[0]

# Pick one sample from that key
sample_data = data[sample_key][0]

print("\nSample key:", sample_key)
print("Sample data shape:", sample_data.shape)
print("Sample data (array):\n", sample_data)