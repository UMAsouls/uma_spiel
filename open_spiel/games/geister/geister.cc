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

std::vector<Action> SelectPhaseLegalActions(uint64_t int_board, int blue_count, int red_count) {
  std::vector<Action> actions;

  uint64_t set_able_pos = (uint64_t(std::pow(2,4))<<7) & (uint64_t(std::pow(2,4))<<13);
  set_able_pos &= int_board;

  while(int_board != 0) {
    uint64_t pos = __builtin_ctzll(int_board);
    int_board &= int_board - 1;
    if(blue_count > 0) actions.push_back(pos);
    if(red_count > 0) actions.push_back(pos + 36);
  }

  return actions;
}


std::vector<Action> BattlePhaseLegalActions(uint64_t int_board) {
  uint64_t shift_up_board = ShiftUp(int_board);
  uint64_t shift_down_board = ShiftDown(int_board);
  uint64_t shift_right_board = ShiftRight(int_board);
  uint64_t shift_left_board = ShiftLeft(int_board);

  std::vector<Action> actions;

  auto able_up = int_board & ~ ShiftDown(shift_up_board & int_board);
  auto able_down = int_board & ~ ShiftUp(shift_down_board & int_board);
  auto able_right = int_board & ~ ShiftLeft(shift_right_board & int_board);
  auto able_left = int_board & ~ ShiftRight(shift_left_board & int_board);

  auto set_able_move = [](uint64_t able_move, std::vector<Action>& actions, int direction) {
    while (able_move != 0)
    {
      uint64_t pos = __builtin_ctzll(able_move);
      able_move &= able_move - 1;
      actions.push_back(pos + direction * 36);
    }
  };

  set_able_move(able_up, actions, 0);
  set_able_move(able_down, actions, 1);
  set_able_move(able_right, actions, 2);
  set_able_move(able_left, actions, 3);

  return actions; 
}

std::vector<Action> GeisterState::LegalActions() const {
  if (IsTerminal()) return {};
  // TODO: bitboard.md や action.md に基づく合法手生成ロジックの実装

  //ReveseMode = trueならint_boardの反転処理を行う
  auto int_board = boards_[current_player_].AllPieces();
  if(auto_reverse_mode_) int_board = ReverseBoard(int_board);

  auto red_count = CountBits(boards_[current_player_].red_pieces);
  auto blue_count = CountBits(boards_[current_player_].blue_pieces);

  std::vector<Action> actions;

  if(phase_ == GeisterPhaseFrag::kPlacement) {
    return SelectPhaseLegalActions(int_board, blue_count, red_count);
  }
  else {
    return BattlePhaseLegalActions(int_board);
  }

  return actions;
}


std::unique_ptr<ActionStruct> GeisterState::ActionToStruct(
    Player player, Action action_id) const {
  auto action_struct = std::make_unique<GeisterActionStruct>();
  action_struct->x = action_id % 6;
  action_struct->y = (action_id / 6) % 6;
  action_struct->direction = action_id / 36;
  return action_struct;
}

std::vector<Action> GeisterState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* a = SafeActionCast<GeisterActionStruct>(action_struct);
  SPIEL_CHECK_GE(a->x, 0);
  SPIEL_CHECK_LT(a->x, kNumCols);
  SPIEL_CHECK_GE(a->y, 0);
  SPIEL_CHECK_LT(a->y, kNumRows);
  SPIEL_CHECK_GE(a->direction, 0);
  SPIEL_CHECK_LT(a->direction, 4);
  return {a->x + a->y * 6 + a->direction * 36};
}

void GeisterState::DoApplyAction(Action action_id) {
  // TODO: アクションの適用（配置フェイズと対戦フェイズでの分岐、BitBoardの更新）
  switch (phase_)
  {
  case GeisterPhaseFrag::kPlacement:
    SelectPhaseApplyAciton(current_player_, action_id);
    break;
  case GeisterPhaseFrag::kPlaying:
    PlayingPhaseApplyAction(current_player_, action_id);
    break;
  default:
    break;
  }

  num_moves_++;
  current_player_ = 1 - current_player_;
}

void GeisterState::SelectPhaseApplyAciton(Player player, Action action_id) {
  // TODO:　select_phase.mdを参考に初期配置フェイズのアクション適用
  OnePlayerBoard& board = boards_[player];
  OnePlayerBoard& opponent_board = boards_[1 - player];

  int x = action_id % 6;
  int y = (action_id / 6) % 6;
  int pos = y * kNumCols + x;
  int kind = action_id / 36;

  if(player == 1 && auto_reverse_mode_) pos = ReversePos(pos);

  if(kind == 0) board.SetBlue(pos);
  else if(kind == 1) board.SetRed(pos);

  int pawn_count = CountBits(board.AllPieces());
  int opponent_pawn_count = CountBits(opponent_board.AllPieces());

  if(pawn_count >= kMaxPieces && opponent_pawn_count >= kMaxPieces) {
    phase_ = GeisterPhaseFrag::kPlaying;
    return;
  }

}

void GeisterState::PlayingPhaseApplyAction(Player player, Action action_id) {
  // TODO: action.mdやstate.mdを参考に対戦フェイズでのアクション適用
  OnePlayerBoard& board = boards_[player];
  OnePlayerBoard& opponent_board = boards_[1 - player];

  // プレイヤ2の自動反転
  if(player == 1 && auto_reverse_mode_) action_id = ReverseAction(action_id);

  //アクションの中身解読
  int x = action_id % 6;
  int y = (action_id / 6) % 6;
  int pos = y * kNumCols + x;
  int dir = action_id / 36;

  //脱出による移動判定
  if(board.HasBlue(pos)) {
    if((player == 0 && (pos == 0 || pos == 5) && dir == 0) ||
       (player == 1 && (pos == 30 || pos == 35) && dir == 1)) {
      outcome_ = player;
      return;
    }
  }

  int next_pos = pos;
  switch (dir)
  {
  case 0:
    next_pos = pos - kNumCols;
    break;
  case 1:
    next_pos = pos + kNumCols;
    break;
  case 2:
    next_pos = pos + 1;
    break;
  case 3:
    next_pos = pos - 1;
    break;
  default:
    break;
  }

  //自分盤面へのアクション適用
  if(board.HasBlue(pos)) board.SetBlue(next_pos);
  else if(board.HasRed(pos)) board.SetRed(next_pos);
  board.Remove(pos);

  //相手盤面へのアクション適用
  if(opponent_board.HasBlue(next_pos)) board.captured_blue++;
  else if(opponent_board.HasRed(next_pos)) board.captured_red++;
  opponent_board.Remove(next_pos);

  // 駒全取りによる勝敗判定
  if(board.captured_blue >= kMaxBluePieces) outcome_ = player;
  else if(board.captured_red >= kMaxRedPieces) outcome_ = 1 - player;

}

// =============================================================================
// GeisterGame の実装
// =============================================================================

GeisterGame::GeisterGame(const GameParameters& params)
    : Game(kGameType, params) {}

int GeisterGame::NumDistinctActions() const {
  // action.md に基づく全行動数: 6(x) * 6(y) * 4(方向) = 144
  return 144;
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
  int x = action_id % 6;
  int y = (action_id / 6) % 6;
  int dir = action_id / 36;
  
  std::string dir_str;
  switch (dir) {
    case 0: dir_str = "Up"; break;
    case 1: dir_str = "Down"; break;
    case 2: dir_str = "Right"; break;
    case 3: dir_str = "Left"; break;
    default: dir_str = "Unknown"; break;
  }
  
  return absl::StrCat("Move(x=", x, ", y=", y, ", dir=", dir_str, ")");
}

}  // namespace geister
}  // namespace open_spiel