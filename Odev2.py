import pandas as pd
import numpy as np


# Aşağıdaki fonksiyon sklearn kütüphanesindeki train_test_split fonksiyonunun görevini yaptırmaya çalıştığım benzer bir fonksiyondur.
def train_test_split(X, Y, test_size = 0.30):
    # Veri setindeki satır sayısını alıyoruz.
    n = X.shape[0]

    # Test verisi için ne kadar veri alacağımızı belirledik.
    test_size = int(n * test_size)

    # Veri setindeki satırları karıştırıyoruz.
    indices = np.random.permutation(n)

    # Train ve test indekslerini ayırıyoruz
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    # X ve Y'yi bu indekslere göre bölüyoruz
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
    
    return X_train, X_test, Y_train, Y_test



data = pd.read_csv("E:\\okul\\MakineÖğrenmesi\\5Holdout\\breastcancer.csv")
data = pd.DataFrame(data)
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
print(data.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


print(f"X Train Data:\n{X_train.head(5)}")
print(f"X Test Data:\n{X_test.head(5)}")

print("All Data: ", data.shape)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)