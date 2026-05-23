from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets, models

# --- 1. Настройка и загрузка модели ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "flower_restNet.pth"

model = models.resnet18(weights=None) 

# Адаптируем последний слой под 5 классов (daisy, dandelion, rose, sunflower, tulip)
num_classes = 5
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Загружаем сохранённые веса
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  

# --- 2. Настройка датасета (валидация) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowers"), transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  

# --- 3. Сбор меток ---
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

# --- 4. Построение матрицы ошибок ---
cm = confusion_matrix(true_labels, predicted_labels)

# --- 5. Визуализация через matplotlib ---
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# Добавляем числа в ячейки
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# Подписи
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (Confusion Matrix)')
plt.colorbar()
plt.tight_layout()
plt.show()
