# Імпортуємо необхідні бібліотеки
import pandas as pd

# Завантажимо дані з файлів
data_file = 'vodafone_music_subset.csv'
dictionary_file = 'Feature_dictionary.xlsx'

# Читаємо дані з файлів
data = pd.read_csv(data_file)
feature_dict = pd.read_excel(dictionary_file, sheet_name=None)

# Переглядаємо перші рядки даних та список листів у файлі словника
data_preview = data.head()
feature_dict_sheets = feature_dict.keys()


