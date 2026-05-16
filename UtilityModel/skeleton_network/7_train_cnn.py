import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import your dataset and model
from _5_dataset import NTUSkeletonDataset, SkeletonAugmentation
from _6_cnn_model import SkeletonCNN

PT_FILE = "ntu_quantized_20.pt"
MODEL_PATH = "skeleton_network/skeleton_cnn_model.pth"
EVAL_ONLY = True

# move stuff to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# create dataset and augment
augment = SkeletonAugmentation(rotate=True, scale=True, noise=True, mirror=True)
dataset = NTUSkeletonDataset(pt_file=PT_FILE)

# split the dataset into 80/20 and apply augmentations
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset.dataset.transform = augment
test_dataset.dataset.transform = None

# set batch size and create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# model, loss function, optimizer
model = SkeletonCNN(num_classes=60).to(device)
criterion = nn.CrossEntropyLoss()

if EVAL_ONLY:
    print("Loading model")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    exit()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# training loop
num_epochs = 30

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print losses to keep track of progress
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # print accuracy for each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch}/{num_epochs}] Validation Accuracy: {val_acc:.2f}%\n")


# save the finished model
torch.save(model.state_dict(), 'skeleton_cnn_model.pth')
print("Training finished. Model saved.")