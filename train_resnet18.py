# train_resnet18.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from datetime import datetime

# Directories
TRAIN_DIR = "train"
VAL_DIR = "val"
MODEL_PATH = "best_model.pt"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets and Loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Setup
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
print("Starting training...")
print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🏋️  Device: {DEVICE}")
print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val images")
print(f"⚙️  Config: {EPOCHS} epochs, batch size {BATCH_SIZE}")
print(f"Total training batches per epoch: {len(train_loader)}")
print(f"Total validation batches per epoch: {len(val_loader)}")
print("=" * 60)

best_acc = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\n🚀 Starting Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress indicator every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_acc = 100 * correct / total
            elapsed = time.time() - epoch_start
            batches_per_sec = (batch_idx + 1) / elapsed
            remaining_batches = len(train_loader) - (batch_idx + 1)
            eta_seconds = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f"  📊 Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {current_acc:.2f}% | "
                  f"Progress: {100*(batch_idx+1)/len(train_loader):.1f}% | "
                  f"ETA: {eta_minutes:.1f}m")

    train_acc = 100 * correct / total

    # Validation
    print(f"  🔍 Running validation...")
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            # Validation progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"    Val batch {batch_idx+1}/{len(val_loader)} | "
                      f"Progress: {100*(batch_idx+1)/len(val_loader):.1f}%")

    val_acc = 100 * val_correct / val_total
    epoch_time = time.time() - epoch_start
    total_elapsed = time.time() - start_time
    
    print(f"\n✅ Epoch {epoch+1} Complete (took {epoch_time/60:.1f} minutes):")
    print(f"   Training   - Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"   Validation - Accuracy: {val_acc:.2f}%")
    print(f"   Total elapsed time: {total_elapsed/60:.1f} minutes")
    
    # Estimate remaining time
    if epoch < EPOCHS - 1:
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining_epochs = EPOCHS - (epoch + 1)
        eta_total = remaining_epochs * avg_epoch_time
        print(f"   🕒 Estimated time remaining: {eta_total/60:.1f} minutes")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("   💾 Saved best model!")
    
    print("-" * 60)

print("=" * 60)
print(f"🎉 Training complete! Best validation accuracy: {best_acc:.2f}%")
print(f"📁 Best model saved as: {MODEL_PATH}")
print("=" * 60)
