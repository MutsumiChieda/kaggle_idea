# Calibration

確率予測時に，モデルの出力値を教師信号のクラス分布に近づけること．

Output|Truth|Calibrated
|-|-|-|
0.4|1|0.5
0.4|0|0.5
0.9|1|1
0.9|1|1

## Calibration Curve

確率予測の信頼性を可視化した曲線．

描画方法  

1. データを予測値でビニング  
2. ビニングしたデータの予測値の平均と，対応するPositiveデータの出現率でプロット  

直線に近いほど確率予測として信頼性が高い．

## Calibrationの方法  

これらは`sklearn.calibration`にも実装されている．

```python  
cl_clf = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
cl_clf.fit()
pred = cl_clf.predict_proba(X_test)[:, 1]
clf_score = brier_score_loss(y_test, pred) # MSE
frac_of_positives, mean_pred_value = calibration_curve(y_test, pred, n_bins=10)
fig = plt.scatter(mean_pred_value, frac_of_positives)
```

### Sigmoid  

Calibration CurveがS字カーブになる場合(__SVMに多い__)に有効．  

入力をモデル出力$f(x)$，教師信号を正解ラベルとしてSigmoidのパラメータ$A, B$を訓練．  
予測時はSigmoidに通した値を出力する．
$$ P(y=1|x) = \frac{1}{1 + \exp(Af(x) + B)} $$

### Isotonic Regression  

Calibration CurveがS字カーブにならない場合(__Naive bayesなど__)に使われる．  

$$ y_i = isotonic(f(x_i)) + \epsilon_i $$

### Calibration for Undersampling  

不均衡データをUndersamplingした場合に有効．  
少数派クラスの確率が大きくなるバイアスを除去できる．  

$$ p = \frac{\beta p_s}{\beta p_s + p_s + 1} $$

### Log Loss 最適化  

LightGBMに有効．

