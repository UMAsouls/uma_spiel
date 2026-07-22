// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_GEISTER_H_
#define OPEN_SPIEL_GAMES_GEISTER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace geister {

// ガイスターの基本定数
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 6;
inline constexpr int kNumCols = 6;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kMaxRedPieces = 4;
inline constexpr int kMaxBluePieces = 4;
inline constexpr int kMaxPieces = kMaxRedPieces + kMaxBluePieces;
inline constexpr int kPlacementActionBase = kNumCells * 4;
inline constexpr int kNumPlacementActions = 70;
inline constexpr int kNumDistinctActions =
    kPlacementActionBase + kNumPlacementActions;
inline constexpr int kMaxGameLength = 1000;  // 引き分け手数
inline constexpr int kNumObservationLayers = 8; // state.mdに基づくTensorの総層数
inline constexpr int kNumInfoStateLayers = 8;

// =============================================================================
// BitBoard用の定数とインライン関数 (bitboard.md に基づく)
// =============================================================================
inline constexpr uint64_t kFullBoardMask = 0xFFFFFFFFFULL; // 6x6 = 36 bits

// 各列のマスク (A列=左端=0, F列=右端=5)
inline constexpr uint64_t kColAMask = 0x041041041ULL; 
inline constexpr uint64_t kColFMask = 0x820820820ULL; 

// 各行のマスク (0行=奥, 5行=手前)
inline constexpr uint64_t kRow0Mask = 0x00000003FULL; 
inline constexpr uint64_t kRow5Mask = 0xFC0000000ULL; 

// ゴールの場所のマスク
inline constexpr uint64_t kGoalMask = 0b100001ULL;

// BitBoardのシフト操作 (インデックスは奥から手前、左から右へ 0~35 と仮定)
inline constexpr uint64_t ShiftUp(uint64_t b) { return (b >> kNumCols) & kFullBoardMask; }
inline constexpr uint64_t ShiftDown(uint64_t b) { return (b << kNumCols) & kFullBoardMask; }
inline constexpr uint64_t ShiftLeft(uint64_t b) { return (b >> 1) & ~kColFMask & kFullBoardMask; }
inline constexpr uint64_t ShiftRight(uint64_t b) { return (b << 1) & ~kColAMask & kFullBoardMask; }

// 単一ビット操作
inline constexpr bool HasBit(uint64_t b, int pos) { return (b & (1ULL << pos)) != 0; }
inline constexpr void SetBit(uint64_t& b, int pos) { b |= ((1ULL << pos) & kFullBoardMask); }
inline constexpr void ClearBit(uint64_t& b, int pos) { b &= ~(1ULL << pos); }

// 盤面を点対称に反転した際のインデックスを取得 (AutoReverseMode用)
inline constexpr int ReversePos(int pos) { return (kNumCells - 1) - pos; }

// アクション方向を点対象に反転（AutoReverseMode用）
inline constexpr int ReverseActionDirection(int dir) { return (dir + 1) % 2 + (dir / 2) * 2; }

// アクションを点対象に反転(AutoReverseMode用)
inline constexpr Action ReverseAction(Action action_id) {
  int pos = action_id % 36;
  int dir = action_id / 36;
  int rev_pos = ReversePos(pos);
  int rev_dir = ReverseActionDirection(dir);
  return rev_pos + rev_dir * 36;
}

inline constexpr uint64_t ReverseBoard(uint64_t b) {
#if defined(__clang__) && __has_builtin(__builtin_bitreverse64)
  return __builtin_bitreverse64(b) >> 28;
#else
  b = ((b & 0x5555555555555555ULL) << 1) |
      ((b >> 1) & 0x5555555555555555ULL);
  b = ((b & 0x3333333333333333ULL) << 2) |
      ((b >> 2) & 0x3333333333333333ULL);
  b = ((b & 0x0F0F0F0F0F0F0F0FULL) << 4) |
      ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  b = ((b & 0x00FF00FF00FF00FFULL) << 8) |
      ((b >> 8) & 0x00FF00FF00FF00FFULL);
  b = ((b & 0x0000FFFF0000FFFFULL) << 16) |
      ((b >> 16) & 0x0000FFFF0000FFFFULL);
  b = (b << 32) | (b >> 32);
  return b >> 28;
#endif
}

// ビット(駒)を数える関数
inline int CountBits(uint64_t b) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(b);
#else
  int count = 0; while (b) { b &= b - 1; count++; } return count;
#endif
}

inline std::vector<int> GetPiecePositions(uint64_t b) {
  std::vector<int> positions;
  while(b > 0) {
    uint64_t pos = __builtin_ctzll(b);
    b &= b - 1;
    positions.push_back(pos);
  }

  return positions;
}

// 現在のゲームフェイズ
enum class GeisterPhaseFrag {
  kPlacement,  // 配置フェイズ
  kPlaying,    // 対戦フェイズ
};

// 一方のプレイヤ側の駒を管理するクラス
// bitboard.md の仕様に基づく
class OnePlayerBoard {
public:
  // 盤面上の駒の位置 (BitBoard)
  uint64_t blue_pieces = 0;
  uint64_t red_pieces = 0;

  // 相手から取った駒の数
  int captured_blue = 0;
  int captured_red = 0;

  // BitBoard操作メソッド
  bool HasPiece(int pos) const { return HasBit(blue_pieces | red_pieces, pos); }
  bool HasBlue(int pos) const { return HasBit(blue_pieces, pos); }
  bool HasRed(int pos) const { return HasBit(red_pieces, pos); }

  void SetBlue(int pos) { SetBit(blue_pieces, pos); }
  void SetRed(int pos) { SetBit(red_pieces, pos); }
  void Remove(int pos) { 
    ClearBit(blue_pieces, pos);
    ClearBit(red_pieces, pos);
  }
  
  uint64_t AllPieces() const { return blue_pieces | red_pieces; }
};

// アクションの構造体表現 (action.mdに基づく)
struct GeisterActionStruct : public ActionStruct {
  int x;
  int y;
  int direction; // 0:↑, 1:↓, 2:→, 3:←
  SPIEL_STRUCT_BOILERPLATE(GeisterActionStruct, x, y, direction);
};

class GeisterGame;
class GeisterObserver;

// ガイスターの状態管理クラス
class GeisterState : public State {
 public:
  GeisterState(
    std::shared_ptr<const Game> game, 
    bool auto_reverse_mode,
    bool action_result_input_mode
  );
  
  GeisterState(const GeisterState&) = default;
  GeisterState& operator=(const GeisterState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  void InformationStateTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  
  std::unique_ptr<ActionStruct> ActionToStruct(
      Player player, Action action_id) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;
  
  // 勝敗結果の取得
  Player outcome() const { return outcome_; }

  // Boardの参照を返す
  const OnePlayerBoard& GetBoard(Player player) const {
    return boards_[player];
  }

  OnePlayerBoard GetBoardCopy(Player player) const {
    return boards_[player];
  }

  GeisterPhaseFrag GetPhaseFrag() const { return phase_; }

  int GetNumMoves() const { return num_moves_; }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  Player current_player_ = 0;
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;

  GeisterPhaseFrag phase_ = GeisterPhaseFrag::kPlacement;
  OnePlayerBoard boards_[kNumPlayers];
  
  // プレイヤ2が手番の時の入力行動や取得盤面・合法手を点対象に反転するフラグ
  bool auto_reverse_mode_;
  bool action_result_input_mode_;

  void SelectPhaseApplyAction(Player player, Action action_id);
  void PlayingPhaseApplyAction(Player player, Action action_id);
};

// ガイスターのゲームオブジェクト
class GeisterGame : public Game {
 public:
  explicit GeisterGame(const GameParameters& params);
  
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;
  int MaxGameLength() const override { return kMaxGameLength; }
  std::string ActionToString(Player player, Action action_id) const override;

  /*
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
  */

  std::shared_ptr<GeisterObserver> default_observer_;
  std::shared_ptr<GeisterObserver> info_state_observer_;
};

}  // namespace geister
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GEISTER_H_
