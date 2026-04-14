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
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace geister {

// ガイスターの基本定数
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 6;
inline constexpr int kNumCols = 6;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kMaxGameLength = 1000;  // 引き分け手数
inline constexpr int kNumObservationLayers = 48; // state.mdに基づくTensorの総層数

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
};

// ガイスターの状態管理クラス
class GeisterState : public State {
 public:
  GeisterState(std::shared_ptr<const Game> game, bool auto_reverse_mode);
  
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
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  
  // 勝敗結果の取得
  Player outcome() const { return outcome_; }

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
  int MaxGameLength() const override { return kMaxGameLength; }
  std::string ActionToString(Player player, Action action_id) const override;
};

}  // namespace geister
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GEISTER_H_