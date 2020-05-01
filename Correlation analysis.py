# Импорт нужных библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()  # более красивый внешний вид графиков по умолчанию

df = pd.read_csv('2019.csv', sep=',')

# df.info()
# print(df.head())
# print(df.describe())

for x in df.columns[3:]:
    sns.jointplot(x=x, y='Score', data=df)
    plt.figure(figsize=(15, 8))  # увеличим размер картинки
    # plt.show()


print(df[df.columns[2:]].corr())


