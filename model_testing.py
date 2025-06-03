import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib  # если scaler был сохранён

# Папка с предсказаниями
pred_dir = 'predictions'
test_dir = 'test'

# Если scaler нужно загрузить (если сохраняли):
# scaler = joblib.load('scaler.pkl')
# или просто переиспользуй из предыдущего скрипта, если в той же сессии

# Иначе создаём scaler, обученный на train
train_files = [f for f in os.listdir('train') if f.endswith('.csv') and not f.endswith('_scaled.csv')]
train_dfs = [pd.read_csv(os.path.join('train', f)) for f in train_files]
train_data_concat = pd.concat(train_dfs, ignore_index=True)
scaler = StandardScaler()
scaler.fit(train_data_concat[['visitors']])

# Проходим по каждому файлу с предсказаниями
for filename in os.listdir(pred_dir):
    if filename.endswith('_scaled.csv'):
        pred_df = pd.read_csv(os.path.join(pred_dir, filename))

        # Обратное масштабирование
        pred_df['predicted_visitors'] = scaler.inverse_transform(pred_df[['predicted_visitors_scaled']])

        # Оригинальные данные
        test_file = filename.replace('pred_', '')
        test_df = pd.read_csv(os.path.join(test_dir, test_file.replace('_scaled.csv', '.csv')))

        # Метрики
        mae = mean_absolute_error(test_df['visitors'], pred_df['predicted_visitors'])
        mse = mean_squared_error(test_df['visitors'], pred_df['predicted_visitors'])
        r2 = r2_score(test_df['visitors'], pred_df['predicted_visitors'])

        print(f"Файл: {filename}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2: {r2:.2f}")
        print()

        # График
        plt.figure(figsize=(10, 6))
        plt.plot(test_df['day'], test_df['visitors'], label='Истинные значения')
        plt.plot(test_df['day'], pred_df['predicted_visitors'], label='Предсказанные значения')
        plt.xlabel('День')
        plt.ylabel('Число посетителей')
        plt.title(f'Файл: {test_file}')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'predictions/plot_{filename.replace("_scaled.csv", ".png")}')
        plt.close()

print("Метрики рассчитаны, графики сохранены!")
