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
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
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
        {"auto_reverse_mode", GameParameter(true)}, // state.mdに基づく反転モードフラグ
        {"action_result_input_mode", GameParameter(false)},
        {"players", GameParameter(kNumPlayers)}
    }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GeisterGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

constexpr uint64_t kPlacementBoardMask =
    (uint64_t{0b1111} << 25) | (uint64_t{0b1111} << 31);

constexpr std::array<std::array<int, 2>, 12> kOrderedColumnPairs{{
    {{0, 1}}, {{0, 2}}, {{0, 3}},
    {{1, 0}}, {{1, 2}}, {{1, 3}},
    {{2, 0}}, {{2, 1}}, {{2, 3}},
    {{3, 0}}, {{3, 1}}, {{3, 2}},
}};

constexpr std::array<std::array<int, 2>, 6> kColumnCombinations{{
    {{0, 1}}, {{0, 2}}, {{0, 3}},
    {{1, 2}}, {{1, 3}}, {{2, 3}},
}};

// Player 0から見た配置領域内の位置。前側がrow 4、後ろ側がrow 5。
constexpr int PlacementPos(int column, bool rear) {
  return (rear ? 31 : 25) + column;
}

uint64_t PlacementRedMask(int placement_id) {
  SPIEL_CHECK_GE(placement_id, 0);
  SPIEL_CHECK_LT(placement_id, kNumPlacementActions);

  uint64_t red_pieces = 0;
  auto set_red = [&red_pieces](int column, bool rear) {
    SetBit(red_pieces, PlacementPos(column, rear));
  };

  if (placement_id < 16) {
    for (int column = 0; column < 4; ++column) {
      set_red(column, (placement_id & (1 << column)) != 0);
    }
  } else if (placement_id < 64) {
    const int local_id = placement_id - 16;
    const auto& pair = kOrderedColumnPairs[local_id / 4];
    const int red_column = pair[0];
    const int blue_column = pair[1];
    const int mixed_bits = local_id % 4;

    set_red(red_column, false);
    set_red(red_column, true);
    int mixed_column_index = 0;
    for (int column = 0; column < 4; ++column) {
      if (column == red_column || column == blue_column) continue;
      set_red(column,
              (mixed_bits & (1 << mixed_column_index)) != 0);
      ++mixed_column_index;
    }
  } else {
    const auto& red_columns = kColumnCombinations[placement_id - 64];
    for (int column : red_columns) {
      set_red(column, false);
      set_red(column, true);
    }
  }

  return red_pieces;
}

// auto_reverse_mode=falseでは列を盤面の絶対的な左から右として扱い、
// 前後だけをPlayer 1側へ反転する。
uint64_t PlacementMaskForPlayer(uint64_t mask, Player player,
                                bool auto_reverse_mode) {
  if (player == 0) return mask;
  if (auto_reverse_mode) return ReverseBoard(mask);

  uint64_t player_mask = 0;
  for (int column = 0; column < 4; ++column) {
    if (HasBit(mask, PlacementPos(column, false))) {
      SetBit(player_mask, 7 + column);
    }
    if (HasBit(mask, PlacementPos(column, true))) {
      SetBit(player_mask, 1 + column);
    }
  }
  return player_mask;
}

}  // namespace

inline void SetPiecesToTensor(u_int64_t p, SpanTensor& t) {
  while (p > 0) {
    uint64_t pos = __builtin_ctzll(p);
    p &= p - 1;
    int x = pos % kNumCols;
    int y = pos / kNumCols;
    t.at(0,y,x) = 1;
  }
}

inline void FillValueToTensor(float value, SpanTensor& t) {
  // data() で absl::Span を取得し、対象領域を std::fill で一気に埋める
  auto span = t.data();
  std::fill(span.begin(), span.begin() + kNumCells, value);
}

class GeisterObserver : public Observer {
public:
  GeisterObserver(const IIGObservationType & iig_obs_type)
      : Observer(/*has_string=*/false, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const GeisterState& state = open_spiel::down_cast<const GeisterState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);

    if(iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer){
      auto out_red = allocator->Get("player_red", {1,kNumRows,kNumCols});
      auto out_blue = allocator->Get("player_blue", {1,kNumRows,kNumCols});
      auto out_enemy = allocator->Get("enemy", {1,kNumRows,kNumCols});

      auto out_got_red = allocator->Get("got_red", {1,kNumRows,kNumCols});
      auto out_got_blue = allocator->Get("got_blue", {1,kNumRows,kNumCols});

      auto out_goal_pos = allocator->Get("goal_pos", {1,kNumRows,kNumCols});
      auto out_game_phase = allocator->Get("game_phase", {1,kNumRows,kNumCols});

      auto out_left_step = allocator->Get("left_step", {1,kNumRows,kNumCols});

      auto board = state.GetBoard(player);
      uint64_t red_pieces = board.red_pieces;
      uint64_t blue_pieces = board.blue_pieces;
      auto en_board = state.GetBoard(1-player);
      uint64_t en_both_pieces = en_board.red_pieces | en_board.blue_pieces;
      uint64_t goal_pos = kGoalMask;

      if(player == 1) {
        red_pieces = ReverseBoard(red_pieces);
        blue_pieces = ReverseBoard(blue_pieces);
        en_both_pieces = ReverseBoard(en_both_pieces);
        goal_pos = ReverseBoard(goal_pos);
      }

      SetPiecesToTensor(board.red_pieces, out_red);
      SetPiecesToTensor(board.blue_pieces, out_blue);
      SetPiecesToTensor(en_both_pieces, out_enemy);
      FillValueToTensor(board.captured_red/kMaxRedPieces, out_got_red);
      FillValueToTensor(board.captured_blue/kMaxBluePieces, out_got_blue);
      
      SetPiecesToTensor(goal_pos, out_goal_pos);

      int phase = state.GetPhaseFrag() == GeisterPhaseFrag::kPlaying;
      FillValueToTensor(phase, out_game_phase);

      float left_step = state.GetNumMoves() / kMaxGameLength;
      FillValueToTensor(left_step, out_left_step);
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {}
  
private:
  const IIGObservationType iig_obs_type_;

};

// =============================================================================
// GeisterState の実装
// =============================================================================

GeisterState::GeisterState(std::shared_ptr<const Game> game, 
    bool auto_reverse_mode,
    bool action_result_input_mode
  ): State(game), 
  auto_reverse_mode_(auto_reverse_mode), 
  action_result_input_mode_(action_result_input_mode) {
  // TODO: 初期状態のセットアップ（配置フェイズの初期化など）
}

std::string GeisterState::ActionToString(Player player, Action action_id) const {
  if (action_id < kPlacementActionBase && auto_reverse_mode_ && player == 1) {
    action_id = ReverseAction(action_id);
  }
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

void GeisterState::ObservationTensor(
  Player player,
  absl::Span<float> values
) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  
  ContiguousAllocator allocator(values);
  const GeisterGame& game = open_spiel::down_cast<const GeisterGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

void GeisterState::InformationStateTensor(
  Player player,
  absl::Span<float> values
) const {
  ContiguousAllocator allocator(values);
  const GeisterGame& game = open_spiel::down_cast<const GeisterGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> GeisterState::Clone() const {
  return std::unique_ptr<State>(new GeisterState(*this));
}

std::vector<Action> SelectPhaseLegalActions() {
  std::vector<Action> actions;
  actions.reserve(kNumPlacementActions);
  for (Action action = kPlacementActionBase;
       action < kNumDistinctActions; ++action) {
    actions.push_back(action);
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
      actions.push_back(pos + direction * kNumCells);
    }
  };

  able_up &= ~(kRow0Mask^kGoalMask);
  set_able_move(able_up, actions, 0);
  able_down &= ~kRow5Mask;
  set_able_move(able_down, actions, 1);
  able_right &= ~kColFMask;
  set_able_move(able_right, actions, 2);
  able_left &= ~kColAMask;
  set_able_move(able_left, actions, 3);

  return actions; 
}

std::vector<Action> GeisterState::LegalActions() const {
  if (IsTerminal()) return {};
  // TODO: bitboard.md や action.md に基づく合法手生成ロジックの実装

  //ReveseMode = trueならint_boardの反転処理を行う
  auto int_board = boards_[current_player_].AllPieces();
  if(auto_reverse_mode_ && current_player_ == 1) int_board = ReverseBoard(int_board);

  if(phase_ == GeisterPhaseFrag::kPlacement) {
    return SelectPhaseLegalActions();
  }
  else {
    return BattlePhaseLegalActions(int_board);
  }
}


std::unique_ptr<ActionStruct> GeisterState::ActionToStruct(
    Player player, Action action_id) const {
  auto action_struct = std::make_unique<GeisterActionStruct>();
  action_struct->x = action_id % kNumCols;
  action_struct->y = (action_id / kNumCols) % kNumRows;
  action_struct->direction = action_id / kNumCells;
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
  return {a->x + a->y * kNumCols + a->direction * kNumCells};
}

void GeisterState::DoApplyAction(Action action_id) {
  // TODO: アクションの適用（配置フェイズと対戦フェイズでの分岐、BitBoardの更新）
  switch (phase_)
  {
  case GeisterPhaseFrag::kPlacement:
    SelectPhaseApplyAction(current_player_, action_id);
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

void GeisterState::SelectPhaseApplyAction(Player player, Action action_id) {
  OnePlayerBoard& board = boards_[player];
  OnePlayerBoard& opponent_board = boards_[1 - player];

  SPIEL_CHECK_GE(action_id, kPlacementActionBase);
  SPIEL_CHECK_LT(action_id, kNumDistinctActions);
  SPIEL_CHECK_EQ(board.AllPieces(), 0);

  const int placement_id = action_id - kPlacementActionBase;
  const uint64_t red_pieces = PlacementRedMask(placement_id);
  const uint64_t blue_pieces = kPlacementBoardMask ^ red_pieces;
  board.red_pieces = PlacementMaskForPlayer(
      red_pieces, player, auto_reverse_mode_);
  board.blue_pieces = PlacementMaskForPlayer(
      blue_pieces, player, auto_reverse_mode_);

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

  // action == -1ならpass
  if(action_id == -1) return;

  int act_result;

  // action_result_input_mode_のとき、9,10bit領域に行動結果が入力されている
  // 行動結果を取り出し、通常のaction_idを取り出す
  if(action_result_input_mode_) {
    constexpr int kActionResultShift = 9;
    constexpr int kActionResultMask = 0b11 << kActionResultShift;
    act_result = (action_id & kActionResultMask) >> kActionResultShift;
    action_id &= ~kActionResultMask;
  }

  // プレイヤ2の自動反転
  if(player == 1 && auto_reverse_mode_) action_id = ReverseAction(action_id);

  //アクションの中身解読
  int x = action_id % kNumCols;
  int y = (action_id / kNumCols) % kNumRows;
  int pos = y * kNumCols + x;
  int dir = action_id / kNumCells;

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

  // 相手盤面へのアクション適用
  // action入力の際は
  if(action_result_input_mode_) {
    if(act_result == 1) board.captured_red++;
    if(act_result == 2) board.captured_blue++;
  }else {
    if(opponent_board.HasBlue(next_pos)) board.captured_blue++;
    else if(opponent_board.HasRed(next_pos)) board.captured_red++;
  }
  opponent_board.Remove(next_pos);

  auto opponent_red_count = kMaxRedPieces - board.captured_red;
  auto opponent_blue_count = kMaxBluePieces - board.captured_blue;

  // 駒全取りによる勝敗判定
  if(opponent_blue_count <= 0) outcome_ = player;
  else if(opponent_red_count <= 0) outcome_ = 1 - player;

}

// =============================================================================
// GeisterGame の実装
// =============================================================================

GeisterGame::GeisterGame(const GameParameters& params)
    : Game(kGameType, params) {
      default_observer_ = std::make_shared<GeisterObserver>(kDefaultObsType);
      info_state_observer_ = std::make_shared<GeisterObserver>(kInfoStateObsType);
  }

int GeisterGame::NumDistinctActions() const {
  return kNumDistinctActions;
}

std::unique_ptr<State> GeisterGame::NewInitialState() const {
  return std::unique_ptr<State>(new GeisterState(
      shared_from_this(), 
      ParameterValue<bool>("auto_reverse_mode", true),
      ParameterValue<bool>("action_result_input_mode", false)
  ));
}

std::vector<int> GeisterGame::ObservationTensorShape() const {
  // state.md に基づく 48層 * 6行 * 6列 のTensor
  return {kNumObservationLayers, kNumRows, kNumCols};
}

std::vector<int> GeisterGame::InformationStateTensorShape() const {
  return {kNumInfoStateLayers, kNumRows, kNumCols};
}

std::string GeisterGame::ActionToString(Player player, Action action_id) const {
  if (action_id >= kPlacementActionBase &&
      action_id < kNumDistinctActions) {
    return absl::StrCat("Placement(id=",
                        action_id - kPlacementActionBase, ")");
  }

  int x = action_id % kNumCols;
  int y = (action_id / kNumCols) % kNumRows;
  int dir = action_id / kNumCells;
  
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
