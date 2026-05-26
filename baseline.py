import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# -------------------------
# Настройки и данные
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры
batch_size = 64
img_size = 64  # Размер картинок
epochs = 25
lr = 1e-4

# Директория с папками цветов
root_dir = './flowers'

# Трансформации
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Датасет
dataset = datasets.ImageFolder(root=root_dir, transform=transform)

# Классы
flower_names = dataset.classes  
num_classes = len(flower_names)

# Делим на train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Простая CNN под 64x64
class FlowerCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #сверточные слои
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        #слои пулинга
        self.pool = nn.MaxPool2d(2, 2)
        #обнуление элементов для регуляризации
        self.dropout = nn.Dropout(0.25)
        #полносвязные слои
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 64->32
        x = self.pool(F.relu(self.conv2(x)))   # 32->16
        x = self.pool(F.relu(self.conv3(x)))   # 16->8
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = FlowerCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Подготовка истории обучения
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
confusion_matrices = []  # будет хранить матрицы ошибок по эпохам
epoch_times = []

# Вспомогательные функции
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total if total > 0 else 0.0

def evaluate_with_confusion(model, loader, num_classes):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            total_samples += labels.size(0)
    #матрица ошибок
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[int(t), int(p)] += 1
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    acc = sum(int(t == p) for t, p in zip(all_true, all_pred)) / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc, cm

# -------------------------
# Обучение
# -------------------------
start_time = time.time()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct += (preds == labels).sum().item()

    train_loss_epoch = running_loss / total_samples
    train_acc_epoch = correct / total_samples

    # Оценка на тестовом наборе после эпохи
    test_loss_epoch, test_acc_epoch, cm_epoch = evaluate_with_confusion(model, test_loader, num_classes)

    # Сохраняем метрики
    train_losses.append(train_loss_epoch)
    train_accuracies.append(train_acc_epoch)
    test_losses.append(test_loss_epoch)
    test_accuracies.append(test_acc_epoch)
    confusion_matrices.append(cm_epoch)
    epoch_time = time.time() - start_time if epoch == 0 else time.time() - (start_time + sum(epoch_times))
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss_epoch:.4f}  train_acc={train_acc_epoch:.4f}  "
          f"test_loss={test_loss_epoch:.4f}  test_acc={test_acc_epoch:.4f}")

# Сохраняем модель
torch.save(model.state_dict(), "flower_cnn.pth")

# -------------------------
# Визуализация кривых и матрицы ошибок
# -------------------------
# 1) кривые обучения (loss)
plt.figure(figsize=(8,6))
plt.plot(range(1, epochs+1), train_losses, label='Train loss')
plt.plot(range(1, epochs+1), test_losses, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning curves: loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curves.png")
plt.close()

# 2) кривые точности
plt.figure(figsize=(8,6))
plt.plot(range(1, epochs+1), train_accuracies, label='Train accuracy')
plt.plot(range(1, epochs+1), test_accuracies, label='Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning curves: accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curves_accuracy.png")
plt.close()

# 3) матрица ошибок для последней эпохи
import seaborn as sns  
class_names = flower_names
cm_last = confusion_matrices[-1]


plt.figure(figsize=(8,6))
plt.imshow(cm_last, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(num_classes), class_names, rotation=45, ha='right')
plt.yticks(np.arange(num_classes), class_names)

fmt = 'd'
thresh = cm_last.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        value = cm_last[i, j]
        plt.text(j, i, format(value, fmt),
                 ha="center", va="center",
                 color="white" if value > thresh else "black")

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Last Epoch)')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("Графики и матрица ошибок сохранены: learning_curves.png, learning_curves_accuracy.png, confusion_matrix.png")
