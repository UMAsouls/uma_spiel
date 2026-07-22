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

#include <initializer_list>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace geister {
namespace {

namespace testing = open_spiel::testing;

uint64_t Mask(std::initializer_list<int> positions) {
  uint64_t mask = 0;
  for (int pos : positions) SetBit(mask, pos);
  return mask;
}

const GeisterState& GeisterStateFrom(const std::unique_ptr<State>& state) {
  return down_cast<const GeisterState&>(*state);
}

void PlacementLegalActionsTest() {
  const std::shared_ptr<const Game> game = LoadGame("geister");
  std::unique_ptr<State> state = game->NewInitialState();

  SPIEL_CHECK_EQ(game->NumDistinctActions(), kNumDistinctActions);
  std::vector<Action> expected_actions;
  for (Action action = kPlacementActionBase;
       action < kNumDistinctActions; ++action) {
    expected_actions.push_back(action);
  }
  SPIEL_CHECK_EQ(state->LegalActions(), expected_actions);
  SPIEL_CHECK_EQ(state->ActionToString(kPlacementActionBase),
                 "Placement(id=0)");
  SPIEL_CHECK_EQ(state->ActionToString(kNumDistinctActions - 1),
                 "Placement(id=69)");
}

void PlacementIdMappingTest() {
  const std::shared_ptr<const Game> game = LoadGame("geister");
  const uint64_t placement_mask =
      Mask({25, 26, 27, 28, 31, 32, 33, 34});
  std::set<uint64_t> red_piece_arrangements;

  for (int placement_id = 0;
       placement_id < kNumPlacementActions; ++placement_id) {
    std::unique_ptr<State> state = game->NewInitialState();
    state->ApplyAction(kPlacementActionBase + placement_id);
    const OnePlayerBoard& board = GeisterStateFrom(state).GetBoard(0);

    SPIEL_CHECK_EQ(CountBits(board.red_pieces), kMaxRedPieces);
    SPIEL_CHECK_EQ(CountBits(board.blue_pieces), kMaxBluePieces);
    SPIEL_CHECK_EQ(board.red_pieces & board.blue_pieces, 0);
    SPIEL_CHECK_EQ(board.AllPieces(), placement_mask);
    red_piece_arrangements.insert(board.red_pieces);
  }
  SPIEL_CHECK_EQ(red_piece_arrangements.size(), kNumPlacementActions);

  const std::vector<std::pair<int, uint64_t>> expected_red_masks = {
      {0, Mask({25, 26, 27, 28})},
      {15, Mask({31, 32, 33, 34})},
      {16, Mask({25, 31, 27, 28})},
      {19, Mask({25, 31, 33, 34})},
      {20, Mask({25, 31, 26, 28})},
      {63, Mask({31, 32, 28, 34})},
      {64, Mask({25, 31, 26, 32})},
      {69, Mask({27, 33, 28, 34})},
  };
  for (const auto& [placement_id, expected_red_mask] :
       expected_red_masks) {
    std::unique_ptr<State> state = game->NewInitialState();
    state->ApplyAction(kPlacementActionBase + placement_id);
    SPIEL_CHECK_EQ(GeisterStateFrom(state).GetBoard(0).red_pieces,
                   expected_red_mask);
  }
}

void PlacementPhaseTransitionTest() {
  const std::shared_ptr<const Game> game = LoadGame("geister");
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(kPlacementActionBase);
  SPIEL_CHECK_TRUE(GeisterStateFrom(state).GetPhaseFrag() ==
                   GeisterPhaseFrag::kPlacement);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->LegalActions().size(), kNumPlacementActions);

  state->ApplyAction(kNumDistinctActions - 1);
  const GeisterState& geister_state = GeisterStateFrom(state);
  SPIEL_CHECK_TRUE(geister_state.GetPhaseFrag() ==
                   GeisterPhaseFrag::kPlaying);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(geister_state.GetBoard(1).red_pieces,
                 ReverseBoard(Mask({27, 33, 28, 34})));
  SPIEL_CHECK_EQ(geister_state.GetBoard(1).blue_pieces,
                 ReverseBoard(Mask({25, 31, 26, 32})));
}

void PlacementWithoutAutoReverseTest() {
  const std::shared_ptr<const Game> game =
      LoadGame("geister(auto_reverse_mode=false)");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(kPlacementActionBase);
  state->ApplyAction(kPlacementActionBase);

  const OnePlayerBoard& board = GeisterStateFrom(state).GetBoard(1);
  SPIEL_CHECK_EQ(board.red_pieces, Mask({7, 8, 9, 10}));
  SPIEL_CHECK_EQ(board.blue_pieces, Mask({1, 2, 3, 4}));
}

void BasicGeisterTests() {
  testing::LoadGameTest("geister");
  testing::NoChanceOutcomesTest(*LoadGame("geister"));
  testing::RandomSimTest(*LoadGame("geister"), 100);
  PlacementLegalActionsTest();
  PlacementIdMappingTest();
  PlacementPhaseTransitionTest();
  PlacementWithoutAutoReverseTest();
}

}  // namespace
}  // namespace geister
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::geister::BasicGeisterTests(); }
