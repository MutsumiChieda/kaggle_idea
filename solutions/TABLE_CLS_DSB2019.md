# [Data Science Bowl 2019](https://www.kaggle.com/c/data-science-bowl-2019)

[まとめ](https://naotaka1128.hatenadiary.jp/entry/dsb-2019-top-solution)  
[色んなランクでの反省会](https://umi-log.com/kaggle-dsb-mtg/)

## どんなコンペだったか  

児童向けアプリの受講者が何回の試験で合格できるかのレベル(順序尺度)を予測する．  
評価基準はQWKだが，不安定だった．  
Shake-up/downの激しさが際立ったが，上位者は入るべくして入ったという印象．

## 提供されたデータ  

アプリ内でのユーザの行動ログ．  
行動ごとに1レコード提供される．

# [1st](https://www.kaggle.com/c/data-science-bowl-2019/discussion/127469) (Pri: 0.568, pub: 0.563)

## モデル

単一のLightGBMモデル．  

## CV  

__基本方針: LB無視__  
不安定でCVとの相関が低かったため．  
代わりに，2つのValidationSetを採用した．

5folds (seedをfoldごとに変更)  
- GroupKFold (installation_id / 5x5Fold)
- スコアはQWKが不安定だったので加重平均RMSEとした．  
加重平均の重みはAssessmentの回数の逆数とした．

Nested CV
- Trainを分割して疑似的にtrain/testに分割した．  
疑似train: 全ログを使った1400ユーザー  
疑似test: ログを一部打ち切った2200ユーザー  
これを50回行った平均をスコアとした．  

## 特徴量エンジニアリング  
### 生成した特徴
- 同じAssessmentか、類似したゲームに関連する特徴量  
(ゲーム内の順序をもとに、ゲームがどのAssessmentと似ているかをマッピング)
- mean/sum/last/std/max/slopeをtrue attempt, correct true, correct feedback に対して算出
- ログデータを以下のように分割して特徴量を作成
  - 全履歴
  - 過去5/12/48時間
  - 前回のAssessmentから現在まで
- Eventインターバル特徴量を作り、mean/lastをevent_idやevent_codeでグルーピングして算出
- ビデオをスキップしたかどうか  
(clip eventインターバル / clip時間)で算出  

### 特徴選択
- 重複の排除
- adversarial AUCが.5になるように特徴を排除
- [null importance](../techniques/特徴選択/null_importance.md) 上位500のみを使用

## データオーグメンテーション
なし．

## アンサンブル
期間中は行っていない  
Late Subでは0.8xLightGBM+0.2xCatBoostが最高のPrivateスコアとなった．

## その他の工夫  
- testでaccuracy_groupがわかるものはtrainに使った．  
- trainにはRMSEを使い、validationには加重平均RMSEを使った．  
- QWK最適化用のしきい値は[OptimizedRounder](https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved)を用いた．