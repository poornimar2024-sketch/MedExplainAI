import os

# path to your dataset
dataset_path = "dataset"

print("Folders inside dataset:")
print(os.listdir(dataset_path))

print("\nTrain folder contents:")
print(os.listdir(os.path.join(dataset_path, "train")))