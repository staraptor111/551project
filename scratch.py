import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EngineDataset(Dataset):
    def __init__(self, synthetic_engine_dir, real_engine_dir, non_engine_dir, transform=None,
                 non_engine_sample_size=3200):
        self.transform = transform

        # Loading of  all engine images (synthetic + real)
        synthetic_paths = list(Path(synthetic_engine_dir).glob('*.png')) + \
                          list(Path(synthetic_engine_dir).glob('*.jpg')) + \
                          list(Path(synthetic_engine_dir).glob('*.jpeg')) + \
                          list(Path(synthetic_engine_dir).glob('*.JPEG'))
        real_paths = list(Path(real_engine_dir).glob('*.png')) + \
                     list(Path(real_engine_dir).glob('*.jpg')) + \
                     list(Path(real_engine_dir).glob('*.jpeg')) + \
                     list(Path(real_engine_dir).glob('*.JPEG'))
        # Load and sampling ( random ) non-engine images
        all_non_engine_paths = np.array(list(Path(non_engine_dir).glob('*.png')) + \
                                        list(Path(non_engine_dir).glob('*.jpg')))  # Convert to numpy array
        non_engine_paths = np.random.choice(all_non_engine_paths,
                                            size=non_engine_sample_size,
                                            replace=False).tolist()

        print(f"Dataset composition:")
        print(f"Synthetic engines: {len(synthetic_paths)}")
        print(f"Real engines: {len(real_paths)}")
        print(f"Non-engines (sampled): {len(non_engine_paths)} (from {len(all_non_engine_paths)})")

        self.image_paths = synthetic_paths + real_paths + non_engine_paths
        self.labels = [1] * (len(synthetic_paths) + len(real_paths)) + \
                      [0] * len(non_engine_paths)
        self.is_real_engine = [0] * len(synthetic_paths) + \
                              [1] * len(real_paths) + \
                              [0] * len(non_engine_paths)

        
        self.num_positive = len(synthetic_paths) + len(real_paths)
        self.num_negative = len(non_engine_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            
            image = Image.new('RGB', (128, 128))

        label = self.labels[idx]
        is_real_engine = self.is_real_engine[idx]

        if self.transform:
            if is_real_engine:
                image = self.transform['real'](image)
            else:
                image = self.transform['simple'](image)

        return image, label

    def __len__(self):
        return len(self.image_paths)

class EngineClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First block - input: 128x128x3
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64x32

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32x64

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x128

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8x256
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Removed nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, save_path='models', pos_weight_value=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

   
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    
    os.makedirs(save_path, exist_ok=True)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((outputs > 0).long().cpu().numpy())
            train_labels.extend(labels.long().cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend((outputs > 0).long().cpu().numpy())
                val_labels.extend(labels.long().cpu().numpy())

        # Calculation of  metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        _, _, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='binary')
        _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')

        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')

        # Debug statements
        print(f"Train predictions distribution: {np.bincount(train_preds)}")
        print(f"Train labels distribution: {np.bincount(train_labels)}")
        print(f"Val predictions distribution: {np.bincount(val_preds)}")
        print(f"Val labels distribution: {np.bincount(val_labels)}")

        # Learning rate scheduling & re-scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_path, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return history

def plot_training_history(history):
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
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Data transformation
    simple_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    real_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_dict = {
        'simple': simple_transform,
        'real': real_transform
    }

    
    synthetic_engine_dir = 'generated_engines'  # Replace with the path to your synthetic engine images
    real_engine_dir = 'enginesreal__2'  # Replace with the path to your real engine images
    non_engine_dir = 'non_engine_images'  # Replace with the path to your non-engine images

    
    full_dataset = EngineDataset(
        synthetic_engine_dir=synthetic_engine_dir,
        real_engine_dir=real_engine_dir,
        non_engine_dir=non_engine_dir,
        transform=transforms_dict
    )

    # Split data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight_value = num_negative / num_positive
    print(f"Number of positive samples: {num_positive}")
    print(f"Number of negative samples: {num_negative}")
    print(f"Pos weight: {pos_weight_value}")

 
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

  
    model = EngineClassifier()
    history = train_model(model, train_loader, val_loader, pos_weight_value=pos_weight_value)

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
