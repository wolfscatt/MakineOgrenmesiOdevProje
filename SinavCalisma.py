from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import numpy as np 
import pandas as pd

X = np.array([[1,2], [3,4],[5,6],[7,8], [3,4], [5,6], [5,6], [7,8], [3,4], [20,6]])

df = pd.DataFrame(X)
print(f'Orjinal veri:\n{df}')

# # Min Max Scaler
# mms = MinMaxScaler(feature_range=(-1,1))
# x_normalized_with_mms = mms.fit_transform(x)
# df = pd.DataFrame(x_normalized_with_mms)
# print(f'Min Max Scaler ile normalize edilmiş veri:\n{df}')


# # Standard Scaler
# ss = StandardScaler()
# x_normalized_with_ss = ss.fit_transform(x)
# df = pd.DataFrame(x_normalized_with_ss)
# print(f'StandardScaler ile normalize edilmiş veri:\n{df}')


# PCA modeli
pca = PCA(n_components=1)  # 1 ana bileşene indirgeniyor
X_pca = pca.fit_transform(X)

print("Orijinal Veri:\n", X)
print("Dönüştürülmüş Veri (PCA):\n", X_pca)
print("Açıklanan Varyans Oranı:", pca.explained_variance_ratio_)
