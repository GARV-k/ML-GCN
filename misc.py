import os
import pandas as pd
from datasets import DatasetDict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# Load and concatenate CSVs
df = pd.concat([
    pd.read_csv('ML-GCN/archive-4/final_stable_diffusion_31k.csv'),
    pd.read_csv('ML-GCN/archive-4/final_artifact_presence_stable_diffusion.csv'),
    pd.read_csv('ML-GCN/archive-4/artifact_presence_latent_diffusion.csv'),
    pd.read_csv('ML-GCN/archive-4/artifact_presence_giga_gan.csv'),
    pd.read_csv('ML-GCN/archive-4/artifact_presence_giga_gan_t2i_coco256.csv').dropna()
], axis=0)

# Shuffle and reset index
df = df.sample(frac=1).reset_index(drop=True)

# Ensure labels are numeric
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
df.fillna(0, inplace=True)  # Replace NaNs with 0
print(df.columns)
print(len(df.columns))

def split_dataframe_by_dirs(df, base_path):
    """
    Splits a DataFrame into train, val, and test subsets based on folder contents.
    """
    train_rows, test_rows, valid_rows = [], [], []
    cnt = 0
    for _, row in df.iterrows():
        img = row['img_name']
        if os.path.exists(os.path.join(base_path, 'test', img)):
            test_rows.append(row)
        elif os.path.exists(os.path.join(base_path, 'train', img)):
            train_rows.append(row)
        elif os.path.exists(os.path.join(base_path, 'val', img)):
            valid_rows.append(row)
        else:
            cnt += 1
    print(f"Files not found: {cnt}")
    train_df = pd.DataFrame(train_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)
    valid_df = pd.DataFrame(valid_rows, columns=df.columns)

    return DatasetDict({
        'train': train_df,
        'test': test_df,
        'val': valid_df,
    })

# Split the DataFrame
dataset_dict = split_dataframe_by_dirs(df, base_path='./ML-GCN/archive-4')

# Define the dataset class
class MultiLabelDataset(Dataset):
    def __init__(self, df, split="train", transform=None, base_path=None):
        """
        Dataset for multi-label classification.
        """
        self.data = df[split]
        self.image_names = self.data.iloc[:, 0].values
        self.labels = self.data.iloc[:, 1:-1].values
        self.transform = transform
        self.split = split
        self.base_path = base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.base_path, self.split, img_name)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert labels to tensor
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        labels = torch.where(labels == 0, torch.tensor(-1.0), labels)
        print(labels)
        return image, labels

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
])

# Initialize DataLoader
batch_size = 16
train_dataset = MultiLabelDataset(dataset_dict, split="train", base_path='./ML-GCN/archive-4', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = MultiLabelDataset(dataset_dict, split="val", base_path='./ML-GCN/archive-4', transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test the DataLoader
print("Testing DataLoader...")
for images, labels in train_dataloader:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break
