import os
import pandas as pd
from sklearn.linear_model import LinearRegression

train_dir = 'train'
test_dir = 'test'

# Читаем все масштабированные train файлы
train_dfs = []
for filename in os.listdir(train_dir):
    if filename.endswith('_scaled.csv'):
        df = pd.read_csv(os.path.join(train_dir, filename))
        train_dfs.append(df)

train_data = pd.concat(train_dfs, ignore_index=True)

# Читаем test файлы
test_dfs = []
for filename in os.listdir(test_dir):
    if filename.endswith('_scaled.csv'):
        df = pd.read_csv(os.path.join(test_dir, filename))
        test_dfs.append((filename, df))

# Модель
model = LinearRegression()

# Обучаем модель на day → visitors_scaled
model.fit(train_data[['day']], train_data['visitors_scaled'])

# Предсказания
os.makedirs('predictions', exist_ok=True)

for filename, test_df in test_dfs:
    preds = model.predict(test_df[['day']])
    result_df = test_df.copy()
    result_df['predicted_visitors_scaled'] = preds
    result_df.to_csv(f'predictions/pred_{filename}', index=False)

print("Предсказания сохранены в папке 'predictions'.")
