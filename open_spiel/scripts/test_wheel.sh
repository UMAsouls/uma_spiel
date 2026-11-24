#!/bin/bash

# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is called by the wheels workflow .github/workflows/wheels.yml.
set -e
set -x

if [ "$2" = "" ];
then
  echo "Usage: test_wheel <mode (geister|full|basic)> <project main dir> [python binary]"
  echo ""
  echo "Basic mode tests only the python functionaly (no ML libraries)"
  echo "Geister mode tests the installed wheel without CMake or ML dependencies."
  echo "Full mode installs the extra ML libraries and the wheel. (requires Python >= 3.11 for JAX)."
  exit -1
fi

MODE=$1
PROJDIR=$2

case "$MODE" in
  geister|full|basic) ;;
  *) echo "Unknown test mode: $MODE" >&2; exit 1 ;;
esac

uname -a
OS=`uname -a | awk '{print $1}'`

# Setting of PYBIN is complicated because of all the different environments this is run from.
if [[ "$3" != "" ]]; then
  PYBIN=$3
else
  PYBIN=${PYBIN:-"python3"}
fi

PYBIN=`which $PYBIN`

if [[ "$MODE" = "geister" ]]; then
  # cibuildwheel has already installed the wheel and its runtime dependencies.
  # Fail explicitly if Geister is missing, rather than accepting an empty API suite.
  "$PYBIN" - <<'PY'
import pyspiel
from open_spiel.python import rl_environment

game = pyspiel.load_game("geister")
assert game.get_type().default_loadable
env = rl_environment.Environment(game)
time_step = env.reset()
for _ in range(4):
    player = time_step.observations["current_player"]
    actions = time_step.observations["legal_actions"][player]
    assert actions, "Geister must provide legal actions"
    time_step = env.step([actions[0]])
print("Installed Geister wheel: load, reset and step passed")
PY
  "$PYBIN" "$PROJDIR/open_spiel/integration_tests/api_test.py" \
    '--test_only_games=^geister$'
  if command -v ccache >/dev/null 2>&1; then
    ccache --show-stats
  fi
  exit 0
fi

$PYBIN -m pip install --upgrade setuptools
$PYBIN -m pip install --upgrade -r $PROJDIR/requirements.txt -q

if [[ "$MODE" = "full" ]]; then
  echo "Full mode. Installing Python extra deps libraries."
  source $PROJDIR/open_spiel/scripts/python_extra_deps.sh $PYBIN
  $PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_JAX_DEPS
  $PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_PYTORCH_DEPS
  $PYBIN -m pip install --upgrade $OPEN_SPIEL_PYTHON_MISC_DEPS
fi

if [[ "$MODE" = "full" ]]; then
  if [[ "$OS" = "Linux" && "$OS_PYTHON_VERSION" = "3.11" ]]; then
    file=`ls wheelhouse/open_spiel-*-cp311-cp311-manylinux*.whl`
    ${PYBIN} -m pip install $file
  elif [[ "$OS" = "Darwin" && "$OS_PYTHON_VERSION" = "3.11" ]]; then
    file=`ls wheelhouse/open_spiel-*-cp311-cp311-*.whl`
    ${PYBIN} -m pip install $file
  else
    echo "Config not found for full tests: $OS / $OS_PYTHON_VERSION"
    exit -1
  fi
fi

export OPEN_SPIEL_BUILDING_WHEEL="ON"
export OPEN_SPIEL_BUILD_WITH_HANABI="ON"
export OPEN_SPIEL_BUILD_WITH_ACPC="ON"

rm -rf build && mkdir build && cd build
cmake -DPython3_EXECUTABLE=${PYBIN} $PROJDIR/open_spiel

NPROC="nproc"
if [[ "$OS" == "darwin"* || "$OS" == "Darwin"* ]]; then
  NPROC="sysctl -n hw.physicalcpu"
fi

MAKE_NUM_PROCS=$(${NPROC})
let TEST_NUM_PROCS=4*${MAKE_NUM_PROCS}

ctest -j$TEST_NUM_PROCS --output-on-failure -R "^python/*" ../open_spiel
