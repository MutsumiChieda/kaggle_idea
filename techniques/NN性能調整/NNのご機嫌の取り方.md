# [NNのご機嫌の取り方](https://www.slideshare.net/TakujiTahara/20190713-kaggle-tokyo-meetup-lt-nn-no-gokigentori-tawara-155334755)

本記事では，以下のケースに対応する．

- kernelに工夫を加えたらscoreが低下した場合
- 論文の実装時に学習が進まない

---  

## ミニバッチサイズ

__資源の許す限り大きめに設定する__ (画像の場合，128程度)  

ミニバッチ$B$での損失は$B$内のサンプルでの損失の平均である．
$$L(B) = \frac{1}{B} \sum_{s \in B} L(y_s, t_s)$$

一般に，バッチサイズが小さいほどLossは大きく(極端なデータに引っ張られる)，  
バッチサイズが大きいほどLossは小さい．  

---  

## 学習率  

__LR-RangeTestで，最適な値を求める__(fastaiのlr_findを利用する)  
__学習率スケジューリングを導入する__(CosineShiftなど)  

LR-RangeTestは，学習率を線形に増加させながら短いステップ数(e.g. 1 ep.)学習する．  
このときのLossをplot(x: lr, y: loss)して，グラフを見て学習率の最大値・最小値を求める．  
- 急激に改善した部分のlr -> lrの最大値
- 改善が緩やかになった部分のlr -> lrの最小値

学習率スケジューリングには，以下のものがある．
|名称|方式|
|-|-|
MultiStepShift|ep50%,75%,90%で0.2倍
CosineShift|徐々に減衰
ReduceLROnPlateau|Loss停滞時に減衰
TriangularLR|減衰・増加を繰り返す  

減衰系のスケジューラーを使う場合，最初にLinearWarmUpを行う．  

手動スケジューリングでは見極めが困難なので，行う場合はアンサンブルを前提に過学習をある程度許容する．

---  

## Initializer
__画像コンペの場合，scaleは小さめに設定する__

## 最適化手法  

__とりあえずSGD+NesterovAG+スケジューリング__  

---  

## 初期値  

初期値に関する[記事](./初期値.md)を参照．