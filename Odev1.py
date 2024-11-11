from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()            # Slaytta bulunan PCA için kullanılan veri seti
data = iris.data
feature_names = iris.feature_names
y = iris.target

# Aldığımız veriyi yukarıda data ve özellik isimleri olarak ayırdık.
# Aşağıda ise pandas ile dataframe formatına çeviriyoruz.
# Ve verinin target kısmını da dataframe içerisinde oluşturduğumuz sınıf isimli kolon içerisine atıyoruz.
df = pd.DataFrame(data, columns=feature_names)
df["sinif"] = y


# Grafik alanını hazırlama
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# LDA modelini oluştur ve bileşen sayısını ikiye indir
for i in range(1, 3):  
    lda = LinearDiscriminantAnalysis(n_components=i)
    x_lda = lda.fit_transform(data, y)
    
    # Varyans oranını hesaplayarak göster
    explained_variance_ratio = lda.explained_variance_ratio_
    print(f"\nn_components={i}")
    print("Variance ratio=", explained_variance_ratio)
    print("Sum ratio=", sum(explained_variance_ratio))

    # Cumulative Variance grafiği
    axs[0].plot(range(1, i + 1), np.cumsum(explained_variance_ratio), marker='o', label=f'n_components={i}')
    axs[0].set(xlabel='Number of Components', ylabel='Cumulative Variance Ratio')
    axs[0].legend()

# Son bileşenlerin dağılımını gösterme
df_sns = pd.DataFrame({'variance': explained_variance_ratio, 'Component': [f'LD{i+1}' for i in range(len(explained_variance_ratio))]})
sns.barplot(x='Component', y='variance', data=df_sns, color="c", ax=axs[1])
axs[1].set_title("Variance Ratio per LDA Component")

plt.tight_layout()
plt.show()