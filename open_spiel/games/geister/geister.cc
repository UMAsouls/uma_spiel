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

#include "open_spiel/games/geister/geister.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace geister {
namespace {

// ガイスターのゲームタイプ定義
const GameType kGameType{
    /*short_name=*/"geister",
    /*long_name=*/"Geister",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation, // 相手の駒の色が見えないため不完全情報ゲーム
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"auto_reverse_mode", GameParameter(true)} // state.mdに基づく反転モードフラグ
    }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GeisterGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// =============================================================================
// GeisterState の実装
// =============================================================================

GeisterState::GeisterState(std::shared_ptr<const Game> game, bool auto_reverse_mode)
    : State(game), auto_reverse_mode_(auto_reverse_mode) {
  // TODO: 初期状態のセットアップ（配置フェイズの初期化など）
}

std::string GeisterState::ActionToString(Player player, Action action_id) const {
  return game_->ActionToString(player, action_id);
}

std::string GeisterState::ToString() const {
  std::string str = "";
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      int pos = r * kNumCols + c;
      
      // P0(大文字) と P1(小文字) の駒を描画
      if (boards_[0].HasBlue(pos)) str += "B";
      else if (boards_[0].HasRed(pos)) str += "R";
      else if (boards_[1].HasBlue(pos)) str += "b";
      else if (boards_[1].HasRed(pos)) str += "r";
      else str += ".";
    }
    if (r < kNumRows - 1) str += "\n";
  }
  
  return str;
}

bool GeisterState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || num_moves_ >= kMaxGameLength;
}

std::vector<double> GeisterState::Returns() const {
  if (outcome_ == 0) return {1.0, -1.0};
  if (outcome_ == 1) return {-1.0, 1.0};
  return {0.0, 0.0}; // 引き分け時
}

std::string GeisterState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string GeisterState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void GeisterState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  
  // TODO: state.md の全48層のTensor構築ロジックを実装
  std::fill(values.begin(), values.end(), 0.0);
}

std::unique_ptr<State> GeisterState::Clone() const {
  return std::unique_ptr<State>(new GeisterState(*this));
}

std::vector<Action> GeisterState::LegalActions() const {
  if (IsTerminal()) return {};
  // TODO: bitboard.md や action.md に基づく合法手生成ロジックの実装
  return {0}; // 一時的なプレースホルダ
}

void GeisterState::DoApplyAction(Action action_id) {
  // TODO: アクションの適用（配置フェイズと対戦フェイズでの分岐、BitBoardの更新）
  num_moves_++;
  current_player_ = 1 - current_player_;
}

// =============================================================================
// GeisterGame の実装
// =============================================================================

GeisterGame::GeisterGame(const GameParameters& params)
    : Game(kGameType, params) {}

int GeisterGame::NumDistinctActions() const {
  // TODO: action.md に基づく正確な全行動数を設定する
  return 144; // 例: 6x6=36マス * 4方向 = 144 等
}

std::unique_ptr<State> GeisterGame::NewInitialState() const {
  return std::unique_ptr<State>(new GeisterState(
      shared_from_this(), ParameterValue<bool>("auto_reverse_mode", true)));
}

std::vector<int> GeisterGame::ObservationTensorShape() const {
  // state.md に基づく 48層 * 6行 * 6列 のTensor
  return {kNumObservationLayers, kNumRows, kNumCols};
}

std::string GeisterGame::ActionToString(Player player, Action action_id) const {
  // TODO: action.mdに基づく文字列フォーマットに修正する
  return absl::StrCat("Action(", action_id, ")");
}

}  // namespace geister
}  // namespace open_spiel