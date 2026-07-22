# このファイルについて
ガイスターの初期配置の設計についてまとめたファイル

# 初期配置について
ゲームの最初に赤駒と青駒の配置を設定することができる。
設定可能な空間は自陣側の中央寄り2行4列（計8マス）である。

# 初期配置時の動作
## OnePlayerBoard　
- 位置(x,y)とプレイヤ、そしてどの駒を設置するかを引数で与えて設置する
- 位置やその駒の設置限界数を超えるとfalseを返し、何も進めない
  - どの位置にどの駒を設置できるかをGetSettablePlace()で取得する
    - 3次元配列で合法手を表す
    - 1,2軸は位置を、3軸目は駒の種類を表す
      - A[x,y,0] = trueなら(x,y)に青駒が設置可能
      - A[x,y,1] = trueなら(x,y)に赤駒が設置可能
    - これらをまた1次元配列に直し、trueのindexを合法手として返す
    - その形で入力しても設置できる関数を用意する(SetPiece(int action_id))
- 仕様上、全部の駒が設置されたら設置不可能になる
## GeisterPhaseFrag
- ゲームが配置フェイズか対戦フェイズかを表すenum
  - GeisterPhaseFrag::SelectPhase = 配置フェイズ
  - GeisterPhaseFrag::PlayPhase = 対戦フェイズ
## GeisterState
- 今の状態が配置フェイズか対戦フェイズか GeisterPhaseFrag GetGamePhase() で取得可能
- 配置フェイズの際はDoApplyAction(int action_id)が配置のための関数に切り替わる
  - action_idはOnePlayerBoardで返された合法手の中から選ばれる
- 対戦フェイズになった時点で駒移動のための関数にDoApplyActionが切り替わる
- 今の状態が対戦フェイズか配置フェイズかを0,1で表し、盤面表現tensorに入れておく