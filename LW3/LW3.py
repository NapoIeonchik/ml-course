import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
pd.set_option('display.max_rows', None)  # Виводити всі рядки
pd.set_option('display.max_columns', None)  # Виводити всі стовпці

df = pd.read_csv("C:/project/machine learning/ml-course/LW3/adult.csv", delimiter=',')
data_cleaned = df.replace('?', np.nan)
data_cleaned['income'] = data_cleaned['income'].apply(lambda x: 1 if x == '>50K' else 0)
data_dropped = data_cleaned.dropna()
data_cleaned = data_dropped.drop(columns=['educational-num'])

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in data_cleaned.select_dtypes(include=np.number).columns:
    #data_cleaned = remove_outliers(data_cleaned, col)
    
    df_encoded = pd.get_dummies(data_cleaned, drop_first=True)
    
correlation_matrix = df_encoded.corr()
# Виведення кореляції між кожною ознакою та таргетною змінною
correlation_with_target = correlation_matrix['income'].sort_values(ascending=False)
print(correlation_with_target)

# Вибір ознак з високою кореляцією з 'income'
selected_features = correlation_with_target[correlation_with_target.abs() > 0.2].index.tolist()
selected_features.remove('income')  # Видаляємо таргетну змінну зі списку ознак
print("Вибрані ознаки для моделі:", selected_features)