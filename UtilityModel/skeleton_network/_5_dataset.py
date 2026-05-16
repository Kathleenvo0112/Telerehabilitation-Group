import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random


# loads the dataset from .pt file, normalizes for scaling and centering
class NTUSkeletonDataset(Dataset):
    def __init__(self, pt_file='ntu_skeleton_dataset.pt', transform=None):
        data = torch.load(pt_file)

        # [N, C, T, V, M]
        self.skeletons = data['skeletons']  

        # for zero-indexing
        self.labels = data['labels'] - 1

        # if needed
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        skeleton = self.skeletons[idx]
        label = self.labels[idx]

        # Centering
        center_joint = skeleton[:, :, 1:2, :]
        skeleton = skeleton - center_joint

        # Scale by magnitude
        scale = skeleton.norm(dim=0, keepdim=True).mean()
        if scale > 0:
            skeleton = skeleton / scale

        if self.transform:
            skeleton = self.transform(skeleton)
        return skeleton, label


# Data augmentation
class SkeletonAugmentation:
    def __init__(self, rotate=True, scale=True, noise=True, mirror=True):
        self.rotate = rotate
        self.scale = scale
        self.noise = noise
        self.mirror = mirror

    def __call__(self, skeleton):

        # random small rotation around Z-axis (assuming X-Y plane)
        if self.rotate:
            angle = random.uniform(-0.2, 0.2)  # radians

            cos, sin = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            rotation_matrix = torch.tensor([[cos, -sin],
                                            [sin, cos]])
            
                                                                              # first 2 coordinate channels (x and y)
            skeleton[..., :2] = torch.einsum('ij,ctvj->ctvi', rotation_matrix, skeleton[..., :2])

        # random scale
        if self.scale:
            factor = random.uniform(0.85, 1.15)
            skeleton[..., :2] *= factor

        # random jitter noise
        if self.noise:
            skeleton += torch.randn_like(skeleton) * 0.01
        
        # random flip of X axis
        if self.mirror and random.random() < 0.5:
            # just flip X axis
            skeleton[0] = -skeleton[0]

        return skeleton 
    

if __name__ == "__main__":
    dataset = NTUSkeletonDataset(pt_file='ntu_skeleton_dataset.pt')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Example: iterate over one batch
    for x, y in train_loader:
        print("Batch skeleton shape:", x.shape)
        print("Batch labels shape:", y.shape)  
        break