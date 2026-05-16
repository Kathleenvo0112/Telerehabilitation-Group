import os
import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score



#Parsing skeletal file 
#returns (T, 25, 3) array of joint coords

def parse_skeleton_file(filepath):
    #open skeleton file and read line by line
    try:
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
        idx = 0
        num_frames = int(lines[idx])
        idx += 1
        frames = []

        #go through each frame, read body count, then read 25 joints for only the first body
        #if multiple bodies exist then we ignore the second body
        for _ in range(num_frames):
            num_bodies = int(lines[idx])
            #skips body metadata
            idx += 1
            first_joints = None

            for b in range(num_bodies):
                idx += 1
                num_joints = int(lines[idx])
                idx += 1
                joints = []

                for j in range(num_joints):
                    vals = lines[idx].split()
                    idx += 1
                    #x y z coords
                    x = float(vals[0])
                    y = float(vals[1])
                    z = float(vals[2])
                    joints.append([x, y, z])


                if b == 0:
                    #only keeping first body if multiple people exist
                    first_joints = joints

            if first_joints is not None:
                frames.append(first_joints)

        #checking correct shape type
        arr = np.array(frames, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 25 or arr.shape[0] == 0:
            print(f"skipping {filepath}, wrong shape {arr.shape}")
            return None
        return arr

    except Exception:
        return None


def normalize_sequence(seq):
    #centers skeleton on joint 0, spine
    hip = seq[:, 0:1, :]
    seq = seq - hip

    #normalizes by joint 0 to joint 1 hip to spine
    joint0 = seq[:, 0, :]
    joint1 = seq[:, 1, :]
    bone_lengths = np.linalg.norm(joint1 - joint0, axis=-1)
    bone=bone_lengths.mean()

    #checks if bone length is very small to avoid division by zero
    if bone > 1e-6:
        seq = seq / bone
    return seq

#resamples sequence to fixed number of frames
def fixed_length_sample(seq, target_len=300):
    T = seq.shape[0]
    #if length is already correct then return
    if T == target_len:
        return seq
    #fix the length by resampling with linear interpolation
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    resampled = np.zeros((target_len, 25, 3), dtype=np.float32)

    #for each joint
    for j in range(25):
        for c in range(3):
            #stretch or compress sequence to fit number of frames
            resampled[:, j, c] = np.interp(new_idx, old_idx, seq[:, j, c])

    return resampled.astype(np.float32)


def load_dataset(data_dir, max_subjects=None, max_files=None, seq_len=300):
    #regex to exrtact subject ID from filename.
    # S001C001P001R001A001.skeleton is subject ID = 001
    pattern = re.compile(r"S\d+C\d+P(\d+)R\d+A\d+\.skeleton", re.IGNORECASE)
    
    #all skeleton files
    files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(".skeleton"):
            files.append(f)

    print(f"Found {len(files)} .skeleton files in {data_dir}")

    #filter subset of subjects for quick testing
    if max_subjects is not None:
        subject_ids_seen = set()
        filtered = []
        for f in files:
            m = pattern.match(f)
            if m:
                subject_ids_seen.add(m.group(1))
                if len(subject_ids_seen) <= max_subjects:
                    filtered.append(f)
        files = filtered
        print(f"  Filtered to {max_subjects} subjects, {len(files)} files")

    if max_files is not None:
        files = files[:max_files]

    sequences = []
    labels = []
    skipped = 0

    for i, fname in enumerate(files):
        m = pattern.match(fname)
        if not m:
            skipped += 1
            continue

        #extracting ID from filename with regex
        subject_id = m.group(1)
        fpath = os.path.join(data_dir, fname)

        #parse skeleton file to get (T, 25, 3) array of joint coords
        seq = parse_skeleton_file(fpath)
        if seq is None:
            skipped += 1
            continue

        #calling normalization and resampling functions
        seq = normalize_sequence(seq)
        seq = fixed_length_sample(seq, target_len=seq_len)

        sequences.append(seq)
        labels.append(subject_id)

        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(files)} files")

    print(f"Loaded {len(sequences)} sequences, skipped {skipped}.")
    print(f"Unique subjects: {len(set(labels))}")
    return np.array(sequences), np.array(labels)

class SkeletonLoader(Dataset):

    def __init__(self, sequences, labels):
        #rearrange dims so channels come first for conv2d input
        self.X = torch.from_numpy(sequences.transpose(0, 3, 1, 2))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ReIDModel(nn.Module):

    def __init__(self, num_classes, seq_len=300, dropout=0.4):
        super().__init__()

        #spatial stage, learns relationships between the joints itself
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
        )

        #temporal stage, learns overall motion patterns across time
        self.temporal = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        #classifier that produces probability score for each subject ID
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        return self.classifier(x)



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        #forward pass
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        #backward pass and updating weights
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    #no gradient calculation during eval
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            #gathering the probabilities and predictions
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)

            preds = predicted.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n = len(all_labels)
    top1 = accuracy_score(all_labels, all_preds)
    k = min(5, num_classes)
    top5 = top_k_accuracy_score(all_labels, np.array(all_probs), k=k)
    return total_loss / n, top1, top5, np.array(all_preds), np.array(all_labels)



def main(data_dir, pt_file, epochs, batch_size, lr, seq_len, max_subjects, max_files, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #load data from pt file or raw skeleton folder
    if pt_file is not None:
        print(f"Loading from .pt file: {pt_file}")
        data = torch.load(pt_file, map_location="cpu")
        skeletons = data["skeletons"]
        raw_labels = data["labels"]
        #convert labels to correct format
        if isinstance(raw_labels, torch.Tensor):
            raw_labels = list(raw_labels.numpy().astype(str))
        sequences = skeletons[:, :, :, :, 0].permute(0, 2, 3, 1).numpy()
    else:
        sequences, raw_labels = load_dataset(
            data_dir,
            max_subjects=max_subjects,
            max_files=max_files,
            seq_len=seq_len,
        )

    #convert string subject IDs to integers
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    num_classes = len(le.classes_)
    print(f"Num classes (subjects): {num_classes}")

    #split into train val and test
    idx = list(range(len(sequences)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    print(f"Train: {len(train_idx)} Val: {len(val_idx)} Test: {len(test_idx)}")

    #creating dataset objects and dataloaders for each split
    train_ds = SkeletonLoader(sequences[train_idx], labels[train_idx])
    val_ds   = SkeletonLoader(sequences[val_idx],   labels[val_idx])
    test_ds  = SkeletonLoader(sequences[test_idx],  labels[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    #setting up model here so we can load best model later for test evaluation
    model = ReIDModel(num_classes=num_classes, seq_len=seq_len).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    #keep track of best model
    best_val_top1 = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_top1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top5, _, _ = eval_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} "
              f"Train loss {tr_loss:.4f} acc {tr_top1:.3f} "
              f"Val loss {val_loss:.4f} acc {val_top1:.3f} top5 {val_top5:.3f}")
        
        #save best model based on validation accuracy
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), save_path)

    #load the best model and evaluate on test set
    print("\nTest Evaluation")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_top1, test_top5, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device, num_classes
    )


    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_top1*100:.1f}%")
    print(f"Test Top-5 Accuracy: {test_top5*100:.1f}%")
    print(f"Chance level: {100/num_classes:.1f}%")

    return test_top1, test_top5


if __name__ == "__main__":
    # change these settings as needed
    DATA_DIR = "nturgb+d_skeletons"
    PT_FILE = None
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 1e-3
    SEQ_LEN = 300
    MAX_SUBJECTS = None
    MAX_FILES = None
    SAVE_PATH = "model_original.pt"

    main(DATA_DIR, PT_FILE, EPOCHS, BATCH_SIZE, LR, SEQ_LEN, MAX_SUBJECTS, MAX_FILES, SAVE_PATH)