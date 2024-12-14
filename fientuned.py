import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EngineDataset(Dataset):
    def __init__(self, synthetic_engine_dir, real_engine_dir, non_engine_dir, transform=None, non_engine_sample_size=3200):
        self.transform = transform

        # Loadinf of all engine images (synthetic + real)
        synthetic_paths = list(Path(synthetic_engine_dir).glob('*.png')) + \
                          list(Path(synthetic_engine_dir).glob('*.jpg')) + \
                          list(Path(synthetic_engine_dir).glob('*.jpeg')) + \
                          list(Path(synthetic_engine_dir).glob('*.JPEG'))
        real_paths = list(Path(real_engine_dir).glob('*.png')) + \
                     list(Path(real_engine_dir).glob('*.jpg')) + \
                     list(Path(real_engine_dir).glob('*.jpeg')) + \
                     list(Path(real_engine_dir).glob('*.JPEG'))
        
        # Loading sample non-engine images
        all_non_engine_paths = np.array(list(Path(non_engine_dir).glob('*.png')) + \
                                        list(Path(non_engine_dir).glob('*.jpg')))  # Convert to numpy array
        non_engine_paths = np.random.choice(all_non_engine_paths,
                                            size=non_engine_sample_size,
                                            replace=False).tolist()

        print(f"Dataset composition:")
        print(f"Synthetic engines: {len(synthetic_paths)}")
        print(f"Real engines: {len(real_paths)}")
        print(f"Non-engines: {len(non_engine_paths)}")

        self.image_paths = synthetic_paths + real_paths + non_engine_paths
        self.labels = [1]*(len(synthetic_paths)+len(real_paths)) + [0]*len(non_engine_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (128, 128))
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    preds_all, labels_all = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds_all.extend((outputs > 0).long().cpu().numpy())
        labels_all.extend(labels.long().cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    _, _, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='binary')
    return avg_loss, acc, f1

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            preds_all.extend((outputs > 0).long().cpu().numpy())
            labels_all.extend(labels.long().cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    _, _, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average='binary')
    return avg_loss, acc, f1

def plot_training_history(history, save_path='training_history_mobilenet.png'):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Directories
    synthetic_engine_dir = 'generated_engines'
    real_engine_dir = 'enginesreal__2'
    non_engine_dir = 'non_engine_images'

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Full dataset
    full_dataset = EngineDataset(synthetic_engine_dir, real_engine_dir, non_engine_dir, transform=train_transform)

    # Splittin of  dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    
    val_dataset.dataset.transform = val_transform

   
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight_value = num_negative / num_positive if num_positive > 0 else 1.0
    print(f"Num positive: {num_positive}, Num negative: {num_negative}, pos_weight: {pos_weight_value}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained ( MobileNetV2 ) 
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Replacing the  classifier
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        scheduler.step(val_loss)

        # Early stopping when epochs collapse into normalcy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mobilenet_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Plotting of training history
    plot_training_history(history, 'training_history_mobilenet.png')

if __name__ == "__main__":
    main()
