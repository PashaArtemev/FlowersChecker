import pandas as pd
import matplotlib.pyplot as plt

# 1. Считываем CSV-файл
df = pd.read_csv('training_log.csv')

# 2. Создаем фигуру с двумя графиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 3. График потерь (Loss)
ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', marker='o')
ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', marker='s')
ax1.set_title('Потери (Loss)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4. График точности (Accuracy)
ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', color='blue', marker='o')
ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', color='red', marker='s')
ax2.set_title('Точность (Accuracy)')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 5. Выводим график на экран
plt.tight_layout()  # Подгоняем отступы, чтобы графики не перекрывались
plt.show()
