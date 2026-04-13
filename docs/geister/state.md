# このファイルについて
このファイルはopen_spielを用いて実装するガイスターの状態遷移や合法手生成を担当する、GeisterStateの設計をまとめたマークダウンファイルである。

# 概要
GeisterStateはopen_spielに存在するStateクラスを継承し、ガイスターを進行する際の状態遷移や合法手生成を行うクラスである。
仕様ともなるゲーム進行ルールは[rule.md](rule.md)にまとめられている。

設計はTicTacToeStateを参考に行う。盤面については高速化のため、BitBoardを用いる。BitBoardの動作については[bitboard.md](bitboard.md)にまとめてある。

# クラス分け
- OnePlayerBoardクラス
  - 一方のプレイヤ側の駒を管理する
- GeisterStateクラス
  - Stateクラスを継承したルール管理クラス
  - 2プレイヤのOnePlayerBoardインスタンスを持つ
  
詳しくは[bitboard.md](bitboard.md)に示してある。

# 状態遷移
ガイスターのルールに従い、遷移を行う。
DoApplyAction(Action action_id)を適用することで行動を適用する。

Action型の中身はintであるが、あくまでidなのでstructとか別の形に変換できる。
詳しくは[action.md](action.md)へ。

# 合法手生成
[bitboard.md](bitboard.md)で示した計算方法により、[action.md](action.md)の方式に則って生成。
ただし出力する際にはその3次元配列を1次元配列に直し、trueのindexだけを保存したvectorを用いる。

# ゲームフェイズ
現在のフェイズ（配置フェイズと対戦フェイズ）についてはGeisterPhaseFragというenumで管理する。
詳しくは[select_phase.md](select_phase.md)へ。

# 初期配置時の動作
配置フェーズ時にはDoApplyActionが配置のための処理となるため、行動入力の形式は変えなくて良い
ただし、アクションデータの読み取りの仕方は変わるため、そこは注意する必要がある
詳しくは[select_phase.md](select_phase.md)へ。

# 盤面のtensor表現
- ObservationTensor()でAIに入力する用のTensorを出力する
- 味方の赤駒、青駒の位置と、相手駒の位置をそれぞれTensorに入れる
  - 全て0,1で表現
  - 合計3層
- 取った敵の青・赤駒の数をスカラーで表現してtensorに入れる
  - 取った駒数/最大数(4)の値
  - 合計2層
- 相手側ゴールの位置
  - ゴール:1、それ以外:0
  - 1層
- 現在のゲームフェーズをスカラーで表現してtehsorに入れる
  - 配置フェーズ:0、対戦フェーズ:1
  - 1層
- 現在の手数を引き分け手数(1000)で割ったスカラー情報
  - 1層
- 全て合わせて8層
