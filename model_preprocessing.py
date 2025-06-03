import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

train_dir = 'train'
test_dir = 'test'

# Список всех train файлов
train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]

# Читаем все train данные в один DataFrame, чтобы fit scaler на всех train данных
train_dfs = [pd.read_csv(f) for f in train_files]
train_data_concat = pd.concat(train_dfs, ignore_index=True)

# Масштабируем только visitors
scaler = StandardScaler()
train_data_concat['visitors_scaled'] = scaler.fit_transform(train_data_concat[['visitors']])

# Теперь разделим обратно по файлам
start_idx = 0
for i, df in enumerate(train_dfs):
    rows = len(df)
    scaled_visitors = train_data_concat['visitors_scaled'].iloc[start_idx:start_idx + rows].values
    df_scaled = pd.DataFrame({
        'day': df['day'],
        'visitors_scaled': scaled_visitors
    })
    output_file = train_files[i].replace('.csv', '_scaled.csv')
    df_scaled.to_csv(output_file, index=False)
    print(f'Файл сохранён: {output_file}')
    start_idx += rows

# Для test файлов применяем transform
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

for file in test_files:
    df = pd.read_csv(file)
    df_scaled = pd.DataFrame({
        'day': df['day'],
        'visitors_scaled': scaler.transform(df[['visitors']]).flatten()
    })
    output_file = file.replace('.csv', '_scaled.csv')
    df_scaled.to_csv(output_file, index=False)
    print(f'Файл сохранён: {output_file}')

print("Все файлы успешно масштабированы!")
