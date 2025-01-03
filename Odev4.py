import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests

#region Boston verisi sklearn kütüphanesinden silindiği için verdikleri linkten manuel aldım.
url = "https://lib.stat.cmu.edu/datasets/boston"
response = requests.get(url)
data = response.text

lines = data.split('\n')

column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", 
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# İlk 22 satır veri açıklamasıdır, bunları atlıyoruz
lines = lines[22:]

# Veriyi iki sütunlu bir şekilde ayırarak okuyoruz
columns = []
for i in range(0, len(lines), 2):  # Her iki satır bir veri grubu oluşturur
    if i+1 < len(lines):  # Satır çiftleri mevcutsa
        col1 = lines[i].split()  # İlk satır
        col2 = lines[i+1].split()  # İkinci satır
        columns.append(col1 + col2)  # İkisini birleştir

# Pandas DataFrame oluştur
columns = np.array(columns, dtype=float)  # Numerik olarak işliyoruz
df = pd.DataFrame(columns, columns=column_names)
#endregion


X = df.drop(columns="MEDV")
y = df["MEDV"]
print(df.head())

plt.figure(figsize=(12,10))
cor = df.corr() 
#print(cor)
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["MEDV"])
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

# Ödevde istenen yeni veri seti df2 içerisinde
df2 = df[relevant_features.index]  # relevant_features.index kodu features'ların kolon isimlerini döndürür
print(df2.head())
