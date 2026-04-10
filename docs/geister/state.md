# このファイルについて
このファイルはopen_spielを用いて実装するガイスターの状態遷移や合法手生成を担当する、GeisterStateの設計をまとめたマークダウンファイルである。

# 概要
GeisterStateはopen_spielに存在するStateクラスを継承し、ガイスターを進行する際の状態遷移や合法手生成を行うクラスである。
設計はTicTacToeStateを参考に行う。盤面については高速化のため、BitBoardを用いる。BitBoardの動作については[bitboard.md](bitboard.md)にまとめてある。

# 状態遷移
ガイスターのルールに従い、遷移を行う。
DoApplyAction(Action action_id)を適用することで行動を適用する。

Action型の中身はintであるが、あくまでidなのでstructとか別の形に変換できる。
詳しくは[action.md](action.md)へ。