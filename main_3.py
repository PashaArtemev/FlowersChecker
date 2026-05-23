import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
# Алгоритм обучает нейросеть (на базе предобученной модели ResNet‑18) распознавать виды цветов. 

# Определяем, на каком устройстве будет работать модель (GPU, если доступно, иначе CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к датасету
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowers")

# RandomResizedCrop(224) — случайное кадрирование до 224x224 (увеличивает разнообразие).
# RandomHorizontalFlip() — случайный переворот по горизонтали.
# ColorJitter - случайные изменения яркости, контраста и оттенка.
# transforms.ToTensor() - Преобразует картинку из обычного формата в многомерный массив.
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

# Собственный датасет
class ImagesFromFoldersDataset(Dataset):
    # Конструктор класса: инициализирует объект датасета
    def __init__(self, samples, root, transform=None):
        self.samples = samples  # Сохраняет список кортежей (путь_к_файлу, метка_класса)
        self.root = root        # Сохраняет путь к корневой папке датасета
        self.transform = transform  # Сохраняет преобразования (трансформации) для изображений (может быть None)

    # Возвращает общее количество элементов в датасете (нужно для DataLoader)
    def __len__(self):
        return len(self.samples)

    # Основной метод: возвращает изображение и его метку по заданному индексу
    def __getitem__(self, idx):
        path, label = self.samples[idx]  # Извлекает путь к файлу и метку класса по индексу idx
        img = Image.open(path).convert('RGB')  # Открывает изображение по пути и конвертирует его в формат RGB
        if self.transform:  # Если заданы преобразования 
            img = self.transform(img)  # применяет их к изображению (например, обрезку, нормализацию и т. д.)
        return img, label  # Возвращает обработанное изображение и соответствующую метку класса


# Функция возвращает готовый список всех изображений с присвоенными метками классов
def gather_samples(root, class_list):
    # Создаём пустой список для хранения кортежей (путь_к_файлу, метка_класса)
    samples = []
    
    # Перебираем все классы из списка class_list
    # enumerate() даёт нам одновременно:
    # - label: порядковый номер класса (0, 1, 2, ...), который будет меткой
    # - cls_name: имя папки (например, 'daisy', 'dandelion')
    for label, cls_name in enumerate(class_list):
        # Формируем полный путь к папке с текущим классом
        # Например: root='./flowers', cls_name='daisy' → './flowers/daisy'
        cls_dir = os.path.join(root, cls_name)
        
        if not os.path.isdir(cls_dir):
            continue
        
        # Перебираем все файлы внутри папки текущего класса
        for fname in os.listdir(cls_dir):
            # Проверяем расширение файла — берём только графические форматы
            # fname.lower() приводит имя к нижнему регистру для надёжности проверки
            # endswith() проверяет, заканчивается ли имя на одно из указанных расширений
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Формируем полный путь к файлу и добавляем в список кортеж:
                # (полный_путь_к_картинке, метка_класса)
                # Например: ('./flowers/daisy/img1.jpg', 0)
                samples.append((os.path.join(cls_dir, fname), label))
    
    return samples



# Обязательная защита для многопроцессорности в Windows
# Гарантирует, что код выполнится только при прямом запуске скрипта (не при импорте)
if __name__ == '__main__':
    
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    num_classes = len(classes)
    all_samples = gather_samples(root_dir, classes)

    # берём 80 % в train, 20 % в val
    train_indices = []  # Список индексов для обучающей выборки
    val_indices = []   # Список индексов для валидационной выборки

    random.seed(42)
    for c in range(num_classes):  # Для каждого класса
        class_samples = [i for i, (_, lbl) in enumerate(all_samples) if lbl == c]
        random.shuffle(class_samples)
        split = int(0.8 * len(class_samples))
        # Добавляем первые 80 % индексов в обучающую выборку
        train_indices.extend(class_samples[:split])
        # Добавляем последние 20 % индексов в валидационную выборку
        val_indices.extend(class_samples[split:])

    # Формируем обучающую выборку из all_samples по индексам train_indices
    train_samples = [all_samples[i] for i in train_indices]
    # Формируем валидационную выборку из all_samples по индексам val_indices
    val_samples = [all_samples[i] for i in val_indices]

    train_dataset = ImagesFromFoldersDataset(train_samples, root_dir, transform=train_transform)
    val_dataset = ImagesFromFoldersDataset(val_samples, root_dir, transform=val_transform)

    # Создаём загрузчик данных для обучения: batch_size=64, перемешивание включено, 4 рабочих процесса
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    # Создаём загрузчик данных для валидации: batch_size=64, без перемешивания, 4 рабочих процесса
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Модель: предобученная ResNet-18 с весами ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Адаптация последнего слоя под количество классов в нашем датасете
    # Заменяем полносвязный слой (fc) на новый с нужным числом выходов
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Заморозка базовых слоёв: обучаем только новые/последние слои
    for name, param in model.named_parameters():
        # Если имя параметра содержит "fc" или "layer4", включаем обучение
        if "fc" in name or "layer4" in name:
            param.requires_grad = True
        else:
            # Остальные параметры замораживаем (не обучаем)
            param.requires_grad = False

    # Оптимизатор и критерий
    # Функция потерь для классификации — кросс‑энтропия (насколько предсказания модели отличаются от реальных ответов.)
    criterion = nn.CrossEntropyLoss()
    # Оптимизатор Adam: обучаем только параметры с requires_grad=True, скорость обучения 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Обучение
    epochs = 25  # Количество эпох обучения
    best_val_acc = 0.0  # Лучшая точность на валидации (инициализация)
    best_model_path = "flower_restNet.pth"  # Путь для сохранения лучшей модели

    for epoch in range(epochs):  # Цикл по эпохам
        # Обучение
        model.train()  # Переводим модель в режим обучения (активация dropout/batchnorm)
        running_loss = 0.0  # Суммарная потеря за эпоху
        total = 0  # Общее количество обработанных образцов
        correct = 0  # Количество правильно классифицированных образцов
        print(f"\n--- Эпоха {epoch+1}/{epochs} начало ---")  # Вывод информации о начале эпохи

        for images, labels in train_loader:  # Цикл по батчам обучающей выборки
            images = images.to(device)  # Перемещаем изображения на устройство
            labels = labels.to(device)  # Перемещаем метки на устройство

            optimizer.zero_grad()  # Обнуляем градиенты перед обратным проходом
            outputs = model(images)  # Прямой проход: получаем предсказания модели
            loss = criterion(outputs, labels)  # Считаем потерю между предсказаниями и истинными метками
            loss.backward()  # вычисляем градиенты
            optimizer.step()  # Обновляем веса модели

            # Накапливаем потерю (умножаем на размер батча для усреднения)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)  # Получаем предсказанные классы
            total += labels.size(0)  # Увеличиваем счётчик общего числа образцов
            # Увеличиваем счётчик правильных предсказаний
            correct += (preds == labels).sum().item()


        # Считаем среднюю потерю за эпоху
        train_loss = running_loss / total
        # Считаем точность на обучающей выборке
        train_acc = correct / total

        # Валидация
        model.eval()  # Переводим модель в режим валидации (отключаем dropout/batchnorm)
        val_running_loss = 0.0  # Суммарная потеря на валидации
        val_total = 0  # Общее число образцов на валидации
        val_correct = 0  # Число правильных предсказаний на валидации
        with torch.no_grad():  # Отключаем вычисление градиентов (экономия памяти и скорости)
            for images, labels in val_loader:  # Цикл по батчам валидационной выборки
                images = images.to(device)  # Перемещаем изображения на устройство
                labels = labels.to(device)  # Перемещаем метки на устройство
                outputs = model(images)  # Прямой проход
                loss = criterion(outputs, labels)  # Считаем потерю

                # Накапливаем потерю на валидации
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)  # Получаем предсказанные классы
                val_total += labels.size(0)  # Увеличиваем счётчик образцов
                # Увеличиваем счётчик правильных предсказаний
                val_correct += (preds == labels).sum().item()

        # Считаем среднюю потерю на валидации за эпоху
        val_loss = val_running_loss / val_total
        # Считаем точность на валидации
        val_acc = val_correct / val_total

        # Выводим метрики за эпоху: потери и точность для train/val
        print(f"Epoch {epoch+1}/{epochs} результаты:  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Сохраняем лучшую модель (с наивысшей точностью на валидации)
        if val_acc > best_val_acc:
            best_val_acc = val_acc  # Обновляем лучшую точность
            # Сохраняем веса модели в файл
            torch.save(model.state_dict(), best_model_path) 

    print(f"\nЛучшая точность: {best_val_acc:.4f}")
    print(f"Сохранил лучшую модель тут: {best_model_path}")
