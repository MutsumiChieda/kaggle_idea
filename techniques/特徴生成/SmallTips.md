# SmallTips

記事にならない小さなテクニックを記述する．

## グループ内偏差

特徴$f_2$でのグループ内における，特徴$f_1$の値の偏差を求める．  
`df[new1] = df[f1] - df.groupby(df(f2)[f1].mean())`  
`df[new2] = df.groupby(f2)[f1].mean()`  

## PCA+角度による特徴生成  

PCAによって2要素に圧縮されたもの同士の関係を抽出する．  

```python
def calc_radian(pca_result):
    rad = []
    for r in pca_result:
        rad.append(math.atan(r[0]/r[1]))
    return rad
pca = PCA(n_components=2).fit(X_train)
X_train['time_rad'] = calc_radian(pca.transform(X_train))
X_test['time_rad']  = calc_radian(pca.transform(X_test))
```  
