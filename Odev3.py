import pandas as pd
import numpy as np


df = pd.DataFrame({'group':list('aaaab'), 
                   'val':[1,3,3,np.NaN,5],
                   'id_group':[1,np.NaN,np.NaN,np.NaN,np.NaN]})

# Tüm Değerleri aynı ve farklı olan değerlerin datadan çıkarılmış hali
df2 = df.loc[:, ~((df.nunique() == 1) | (df.nunique() == df.count()))]
print(df2)
# nunique() metodu her sütunun benzersiz değer sayısını döndürür.
# ~ işareti ise koşulun tersini alır. Basitçe, ters alma işaretidir.