import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowers")

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class ImagesFromFoldersDataset(Dataset):
    def __init__(self, samples, root, transform=None):
        self.samples = samples
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def gather_samples(root, class_list):
    samples = []
    for label, cls_name in enumerate(class_list):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples

if __name__ == '__main__':
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    num_classes = len(classes)

    all_samples = gather_samples(root_dir, classes)

    train_indices = []
    val_indices = []
    random.seed(42)

    for c in range(num_classes):
        class_samples = [i for i, (_, lbl) in enumerate(all_samples) if lbl == c]
        random.shuffle(class_samples)
        split = int(0.8 * len(class_samples))
        train_indices.extend(class_samples[:split])
        val_indices.extend(class_samples[split:])

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    train_dataset = ImagesFromFoldersDataset(train_samples, root_dir, transform=train_transform)
    val_dataset = ImagesFromFoldersDataset(val_samples, root_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    sota_model = models.efficientnet_b0(weights=None)

    try:
        in_features = sota_model.classifier[1].in_features
        sota_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    except Exception:
        if hasattr(sota_model.classifier, "in_features"):
            in_features = sota_model.classifier.in_features
            sota_model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Не удалось определить входные признаки последнего слоя.")

    sota_model = sota_model.to(device)

    for param in sota_model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sota_model.parameters()), lr=3e-4, weight_decay=1e-4)

    epochs = 25
    best_val_acc = 0.0
    best_model_path = "flower_effnet_b0.pth"

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    all_val_true = []
    all_val_pred = []

    for epoch in range(epochs):
        sota_model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        print(f"\n--- Эпоха {epoch+1}/{epochs} начало ---")

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = sota_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        sota_model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct = 0
        all_val_true_epoch = []
        all_val_pred_epoch = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = sota_model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                all_val_true_epoch.extend(labels.cpu().numpy())
                all_val_pred_epoch.extend(preds.cpu().numpy())

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(sota_model.state_dict(), best_model_path)

        all_val_true.extend(all_val_true_epoch)
        all_val_pred.extend(all_val_pred_epoch)

    print(f"\nЛучшая точность на валидации: {best_val_acc:.4f}")
    print(f"Лучший путь к модели: {best_model_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss в процессе обучения')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy во время обучения')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_accuracy.png')
    plt.close()

    cm = confusion_matrix(all_val_true, all_val_pred, labels=list(range(num_classes)))
    print("Confusion Matrix (final validation):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix на валидации')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    with open('confusion_matrix.txt', 'w') as f:
        f.write("Confusion Matrix (final validation):\n")
        f.write(np.array2string(cm))
