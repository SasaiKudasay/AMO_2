import numpy as np
import pandas as pd
import os

# Создаём папки train и test
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)


# Функция генерации данных
def generate_data(num_days, noise=False, anomaly=False):
    days = np.arange(1, num_days + 1)

    # Базовая функция: посетителей больше в выходные (дни 6 и 7, 13 и 14 и т.д.)
    visitors = 100 + 30 * np.sin(days * (2 * np.pi / 7))  # недельный цикл

    if noise:
        visitors += np.random.normal(0, 10, size=num_days)  # добавляем шум (ошибки измерений)

    if anomaly:
        anomaly_indices = np.random.choice(num_days, size=2, replace=False)
        visitors[anomaly_indices] += np.random.choice([50, -50], size=2)  # очень много или очень мало посетителей

    return pd.DataFrame({'day': days, 'visitors': visitors})


# Генерируем 3 train файла
for i in range(1, 4):
    df = generate_data(30, noise=(i == 2), anomaly=(i == 3))
    df.to_csv(f'train/train_data_{i}.csv', index=False)

# Генерируем 2 test файла
for i in range(1, 3):
    df = generate_data(30, noise=(i == 1), anomaly=(i == 2))
    df.to_csv(f'test/test_data_{i}.csv', index=False)

print("Данные успешно созданы!")
