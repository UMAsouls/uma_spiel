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

# 勝敗判定
- 手番開始時に自分の青駒が相手側の脱出口にいるとき、合法手が脱出させる行動だけになり、それを実行することで勝利となる
  - 脱出口にいるかどうかはbitboardで判定
- 手番終了時に相手側陣地の駒数も数え、赤駒が無かったら負け、青駒が無かったら勝ちとなる
  - こっちもbitboardで判定

# 状態遷移
ガイスターのルールに従い、遷移を行う。
DoApplyAction(Action action_id)を適用することで行動を適用する。

Action型の中身はintであるが、あくまでidなのでstructとか別の形に変換できる。
詳しくは[action.md](action.md)へ。

# 合法手生成
[bitboard.md](bitboard.md)で示した計算方法により、[action.md](action.md)の方式に則って生成。
ただし出力する際にはその3次元配列を1次元配列に直し、trueのindexだけを保存したvectorを用いる。
青駒が脱出口にいる際は脱出させる手だけを返す

# AutoReverseMode
State生成時にAutoReverseMode = trueとすることで、プレイヤ2が手番の時の入力行動や取得盤面・合法手が自動的に点対象に反転するようになる。
AI作成時には手前側が自陣であることだけ考慮すればいい。

# 行動入力
出力された合法手(1次元配列)から値を選んで行動とする。
脱出口に青駒がある際は行動の中身に関わらず、入力された時点で勝利処理

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
  - 相手駒は赤駒、青駒を区別せずに入れる
  - 合計3層
- 取った敵の青・赤駒の数をスカラーで表現してtensorに入れる
  - 取った駒数/最大数(4)の値
  - 合計2層
- 相手側脱出口の位置
  - 脱出口:1、それ以外:0
  - 1層
- 現在のゲームフェーズをスカラーで表現してtehsorに入れる
  - 配置フェーズ:0、対戦フェーズ:1
  - 1層
- 現在の手数を引き分け手数(1000)で割ったスカラー情報
  - 1層
- 過去20手の履歴
  - 詳しくは[record.md](record.md)へ
  - 40層
- 全て合わせて48層

# 手番管理
- TictacToeStateと同じ
- 手番終了時に current_player = 1 - current_player;