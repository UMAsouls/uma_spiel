from __future__ import annotations

import os
import sys

f_path = os.path.abspath(__file__).split("/")
pj_path = ""
for p in f_path[:-4]:
    pj_path += f"/{p}"
    
sys.path.append(pj_path)

import argparse
import random
import socket
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyspiel
import torch

from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.pytorch.two_player_ppo import TwoPlayerPPO

MAX_COL = 6
MAX_ROW = 6
BOARD_SIZE = MAX_COL * MAX_ROW
NUM_GHOSTS_PER_PLAYER = 8

GAME_NAME = "geister"
DEFAULT_MODEL_PATH = Path(pj_path+"/geister_research/data/model/ver_1")

OWN_GHOST_CODES = tuple("ABCDEFGH")
OPPONENT_GHOST_CODES = tuple("abcdefgh")
ALL_GHOST_CODES = OWN_GHOST_CODES + OPPONENT_GHOST_CODES

# プロトコル上の初期位置。pos = y * 6 + x。
INITIAL_POSITIONS: dict[str, int] = {
    "A": 4 * MAX_COL + 1,
    "B": 4 * MAX_COL + 2,
    "C": 4 * MAX_COL + 3,
    "D": 4 * MAX_COL + 4,
    "E": 5 * MAX_COL + 1,
    "F": 5 * MAX_COL + 2,
    "G": 5 * MAX_COL + 3,
    "H": 5 * MAX_COL + 4,
    "a": 1 * MAX_COL + 4,
    "b": 1 * MAX_COL + 3,
    "c": 1 * MAX_COL + 2,
    "d": 1 * MAX_COL + 1,
    "e": 0 * MAX_COL + 4,
    "f": 0 * MAX_COL + 3,
    "g": 0 * MAX_COL + 2,
    "h": 0 * MAX_COL + 1,
}
POSITION_TO_OWN_GHOST = {
    pos: code for code, pos in INITIAL_POSITIONS.items() if code.isupper()
}

MOVE_DIR_NAMES = ("NORTH", "SOUTH", "EAST", "WEST")
DIRECTION_DELTAS = (-MAX_COL, MAX_COL, 1, -1)
DELTA_TO_DIRECTION = {delta: direction for direction, delta in enumerate(DIRECTION_DELTAS)}

ACTION_RESULT_SHIFT = 9
RESULT_NONE = 0
RESULT_RED = 1
RESULT_BLUE = 2

BUFSIZE = 4096
ENCODING = "utf-8"


class ProtocolError(RuntimeError):
    pass


class ClientPhase(Enum):
    WAIT_SET_REQUEST = auto()
    WAIT_SET_RESULT = auto()
    WAIT_BOARD = auto()
    WAIT_MOVE_RESULT = auto()
    FINISHED = auto()


class GhostStatus(Enum):
    ON_BOARD = auto()
    CAPTURED = auto()
    ESCAPED = auto()


@dataclass(frozen=True)
class GhostInfo:
    code: str
    color: str
    status: GhostStatus
    position: int | None

    @property
    def x(self) -> int | None:
        return None if self.position is None else self.position % MAX_COL

    @property
    def y(self) -> int | None:
        return None if self.position is None else self.position // MAX_COL


BoardState = dict[str, GhostInfo]


class LineSocket:
    """TCPストリームから CRLF 単位でコマンドを切り出す。"""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buffer = bytearray()

    def recv_line(self) -> str:
        while True:
            delimiter_index = self.buffer.find(b"\r\n")
            if delimiter_index >= 0:
                raw = bytes(self.buffer[:delimiter_index])
                del self.buffer[: delimiter_index + 2]
                return raw.decode(ENCODING)

            chunk = self.sock.recv(BUFSIZE)
            if not chunk:
                # 最後の行にCRLFが無い実装にも最低限対応する。
                if self.buffer:
                    raw = bytes(self.buffer)
                    self.buffer.clear()
                    return raw.decode(ENCODING)
                raise ConnectionError("サーバーとのTCP接続が切断されました")
            self.buffer.extend(chunk)

    def send_line(self, command: str) -> None:
        command = command.rstrip("\r\n")
        self.sock.sendall((command + "\r\n").encode(ENCODING))


def normalize_response(command: str) -> str:
    # 仕様書にある "OK " のような余分な空白を吸収する。
    return command.strip()


def reverse_position(position: int) -> int:
    return BOARD_SIZE - 1 - position


def reverse_direction(direction: int) -> int:
    # C++側 ReverseActionDirection と同じ変換: 上↔下、右↔左。
    return (direction + 1) % 2 + (direction // 2) * 2


def reverse_action(action: int) -> int:
    position = action % BOARD_SIZE
    direction = action // BOARD_SIZE
    return reverse_position(position) + reverse_direction(direction) * BOARD_SIZE


def add_action_result(action: int, result: int) -> int:
    if result not in (RESULT_NONE, RESULT_RED, RESULT_BLUE):
        raise ValueError(f"不正な行動結果です: {result}")
    return action | (result << ACTION_RESULT_SHIFT)


def parse_board_command(command: str) -> tuple[str, BoardState]:
    """MOV/WON/LSTコマンドを駒コード単位の状態へ変換する。"""
    if "?" not in command:
        raise ProtocolError(f"盤面コマンドに '?' がありません: {command!r}")

    kind, payload = command.split("?", 1)
    if kind not in {"MOV", "WON", "LST"}:
        raise ProtocolError(f"盤面コマンドではありません: {command!r}")

    expected_length = len(ALL_GHOST_CODES) * 3
    if len(payload) != expected_length:
        raise ProtocolError(
            f"盤面文字列の長さが不正です: expected={expected_length}, "
            f"actual={len(payload)}, payload={payload!r}"
        )

    board: BoardState = {}
    for index, code in enumerate(ALL_GHOST_CODES):
        token = payload[index * 3 : index * 3 + 3]
        x_char, y_char, color = token
        if not (x_char.isdigit() and y_char.isdigit()):
            raise ProtocolError(f"座標が不正です: code={code}, token={token!r}")

        x = int(x_char)
        y = int(y_char)
        if (x, y) == (9, 9):
            status = GhostStatus.CAPTURED
            position = None
        elif (x, y) == (8, 8):
            status = GhostStatus.ESCAPED
            position = None
        elif 0 <= x < MAX_COL and 0 <= y < MAX_ROW:
            status = GhostStatus.ON_BOARD
            position = y * MAX_COL + x
        else:
            raise ProtocolError(f"範囲外の座標です: code={code}, token={token!r}")

        if color not in {"R", "B", "r", "b", "u"}:
            raise ProtocolError(f"不正な色コードです: code={code}, color={color!r}")

        board[code] = GhostInfo(code, color, status, position)

    return kind, board


def make_initial_expected_board(own_red_codes: set[str]) -> BoardState:
    board: BoardState = {}
    for code in ALL_GHOST_CODES:
        if code.isupper():
            color = "R" if code in own_red_codes else "B"
        else:
            color = "u"
        board[code] = GhostInfo(
            code=code,
            color=color,
            status=GhostStatus.ON_BOARD,
            position=INITIAL_POSITIONS[code],
        )
    return board


def make_set_command(own_red_codes: Iterable[str]) -> str:
    red_codes = sorted(set(own_red_codes))
    if len(red_codes) != 4 or any(code not in OWN_GHOST_CODES for code in red_codes):
        raise ValueError(f"赤駒はA-Hから4個必要です: {red_codes}")
    return "SET:" + "".join(red_codes)


def ghost_at(board: Mapping[str, GhostInfo], position: int, *, own: bool) -> str:
    codes = OWN_GHOST_CODES if own else OPPONENT_GHOST_CODES
    matches = [
        code
        for code in codes
        if board[code].status is GhostStatus.ON_BOARD
        and board[code].position == position
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"位置{position}の駒を一意に特定できません: own={own}, matches={matches}"
        )
    return matches[0]


def make_move_command(action: int, board: Mapping[str, GhostInfo]) -> str:
    direction = action // BOARD_SIZE
    position = action % BOARD_SIZE
    if not 0 <= direction < len(MOVE_DIR_NAMES):
        raise ProtocolError(f"方向を解釈できないactionです: {action}")
    ghost_code = ghost_at(board, position, own=True)
    return f"MOV:{ghost_code},{MOVE_DIR_NAMES[direction]}"


def infer_opponent_action(
    previous: Mapping[str, GhostInfo],
    current: Mapping[str, GhostInfo],
    own_colors: Mapping[str, str],
) -> tuple[int, int]:
    """盤面差分から相手のローカルactionと捕獲結果を復元する。"""
    moved: list[tuple[str, GhostInfo, GhostInfo]] = []
    for code in OPPONENT_GHOST_CODES:
        before = previous[code]
        after = current[code]
        if before.status != after.status or before.position != after.position:
            moved.append((code, before, after))

    if len(moved) != 1:
        raise ProtocolError(
            "相手の移動駒を一意に推定できません: "
            + ", ".join(
                f"{code}:{before.position}/{before.status.name}"
                f"->{after.position}/{after.status.name}"
                for code, before, after in moved
            )
        )

    code, before, after = moved[0]
    if before.status is not GhostStatus.ON_BOARD or before.position is None:
        raise ProtocolError(f"移動前の相手駒が盤上にありません: {code}")
    if after.status is not GhostStatus.ON_BOARD or after.position is None:
        # 相手の脱出直後には通常こちらへMOV要求は来ないため、ここは異常系扱い。
        raise ProtocolError(f"相手駒の通常移動ではありません: {code}, {after.status.name}")

    delta = after.position - before.position
    if delta not in DELTA_TO_DIRECTION:
        raise ProtocolError(
            f"相手駒の移動量が不正です: {code}, {before.position}->{after.position}"
        )

    global_direction = DELTA_TO_DIRECTION[delta]
    global_action = before.position + global_direction * BOARD_SIZE
    # OpenSpielのplayer 1へは、auto_reverse_mode前提のローカル視点actionを渡す。
    local_action = reverse_action(global_action)

    captured_codes = [
        own_code
        for own_code in OWN_GHOST_CODES
        if previous[own_code].status is GhostStatus.ON_BOARD
        and current[own_code].status is GhostStatus.CAPTURED
    ]
    if len(captured_codes) > 1:
        raise ProtocolError(f"1手で複数の自駒が取られています: {captured_codes}")

    result = RESULT_NONE
    if captured_codes:
        captured_code = captured_codes[0]
        previous_position = previous[captured_code].position
        if previous_position != after.position:
            raise ProtocolError(
                f"捕獲された駒の位置と相手移動先が一致しません: "
                f"captured={captured_code}@{previous_position}, moved_to={after.position}"
            )
        color = own_colors[captured_code]
        result = RESULT_RED if color == "R" else RESULT_BLUE

    return local_action, result


def apply_own_move_to_snapshot(
    board: Mapping[str, GhostInfo], action: int, result: int
) -> BoardState:
    """次回の相手手推定用に、直前のサーバー盤面へ自分の手を反映する。"""
    updated = dict(board)
    position = action % BOARD_SIZE
    direction = action // BOARD_SIZE
    mover_code = ghost_at(board, position, own=True)
    mover = board[mover_code]

    is_escape = direction == 0 and position in (0, 5)
    if is_escape:
        updated[mover_code] = GhostInfo(
            mover_code, mover.color, GhostStatus.ESCAPED, None
        )
        return updated

    next_position = position + DIRECTION_DELTAS[direction]
    updated[mover_code] = GhostInfo(
        mover_code, mover.color, GhostStatus.ON_BOARD, next_position
    )

    opponent_on_destination = [
        code
        for code in OPPONENT_GHOST_CODES
        if board[code].status is GhostStatus.ON_BOARD
        and board[code].position == next_position
    ]

    if result == RESULT_NONE:
        if opponent_on_destination:
            raise ProtocolError(
                f"結果はOKですが移動先に相手駒があります: {opponent_on_destination}"
            )
    else:
        if len(opponent_on_destination) != 1:
            raise ProtocolError(
                f"捕獲結果を受信しましたが対象駒を特定できません: "
                f"destination={next_position}, matches={opponent_on_destination}"
            )
        captured_code = opponent_on_destination[0]
        captured_color = "r" if result == RESULT_RED else "b"
        updated[captured_code] = GhostInfo(
            captured_code, captured_color, GhostStatus.CAPTURED, None
        )

    return updated


def positions_equal(left: Mapping[str, GhostInfo], right: Mapping[str, GhostInfo]) -> bool:
    return all(
        left[code].status == right[code].status
        and left[code].position == right[code].position
        for code in ALL_GHOST_CODES
    )


def current_player(time_step: Any) -> int:
    observations = time_step.observations
    return int(observations["current_player"])


def legal_actions(time_step: Any, player: int) -> list[int]:
    actions = time_step.observations["legal_actions"]
    if isinstance(actions, Mapping):
        return list(actions[player])
    return list(actions[player])


def extract_agent_action(output: Any) -> int:
    if isinstance(output, (int, np.integer)):
        return int(output)
    if hasattr(output, "action"):
        return int(output.action)
    if isinstance(output, Sequence) and output:
        first = output[0]
        if isinstance(first, (int, np.integer)):
            return int(first)
        if hasattr(first, "action"):
            return int(first.action)
    raise TypeError(f"PPOAgentの出力からactionを取得できません: {type(output)!r}")


def choose_ppo_action(agent: Any, time_step: Any) -> int:
    try:
        output = agent.step([time_step], is_evaluation=True)
    except TypeError:
        output = agent.step([time_step])
    action = extract_agent_action(output)

    player = current_player(time_step)
    legal = legal_actions(time_step, player)
    if action not in legal:
        raise RuntimeError(f"PPOが非合法手を返しました: action={action}, legal={legal}")
    return action


def create_agent(env: rl_environment.Environment, model_path: Path) -> TwoPlayerPPO:
    info_state_shape = tuple(np.asarray(env.observation_spec()["info_state"]).flatten())
    num_actions = env.game.num_distinct_actions()
    agent = TwoPlayerPPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=env.game.num_players(),
        num_envs=1,
        agent_fn=PPOAgent,
        device="cpu",
    )
    state_dict = torch.load(model_path, map_location="cpu")
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def run_placement_phase(
    env: rl_environment.Environment,
    agent: Any,
    rng: random.Random,
) -> tuple[Any, set[str], dict[str, str]]:
    time_step = env.reset()
    own_red_codes: set[str] = set()

    # 両者が8駒ずつ置くまで、合計16手進める。
    for _ in range(NUM_GHOSTS_PER_PLAYER * 2):
        player = current_player(time_step)
        legal = legal_actions(time_step, player)
        if not legal:
            raise RuntimeError("配置フェーズ中に合法手が無くなりました")

        if player == 0:
            action = choose_ppo_action(agent, time_step)
            kind = action // BOARD_SIZE
            position = action % BOARD_SIZE
            if kind == 1:
                try:
                    own_red_codes.add(POSITION_TO_OWN_GHOST[position])
                except KeyError as exc:
                    raise RuntimeError(f"初期配置位置が不正です: {position}") from exc
        else:
            action = rng.choice(legal)

        time_step = env.step([action])

    if len(own_red_codes) != 4:
        raise RuntimeError(f"赤駒が4個になっていません: {sorted(own_red_codes)}")

    own_colors = {
        code: ("R" if code in own_red_codes else "B") for code in OWN_GHOST_CODES
    }
    return time_step, own_red_codes, own_colors


def response_to_result(response: str) -> int:
    normalized = normalize_response(response)
    if normalized == "OK":
        return RESULT_NONE
    if normalized == "OKR":
        return RESULT_RED
    if normalized == "OKB":
        return RESULT_BLUE
    if normalized == "NG":
        raise ProtocolError("サーバーが送信した行動をNGとして拒否しました")
    raise ProtocolError(f"未知の移動結果です: {response!r}")


def main(host: str, port: int, model_path: Path, seed: int | None = None) -> None:
    rng = random.Random(seed)
    phase = ClientPhase.WAIT_SET_REQUEST

    game = pyspiel.load_game(
        GAME_NAME,
        {
            "auto_reverse_mode": True,
            "action_result_input_mode": True,
        },
    )
    env = rl_environment.Environment(game=game)
    agent = create_agent(env, model_path)

    with socket.create_connection((host, port)) as raw_socket:
        connection = LineSocket(raw_socket)

        command = connection.recv_line()
        if normalize_response(command) != "SET?":
            raise ProtocolError(f"最初のコマンドがSET?ではありません: {command!r}")

        time_step, own_red_codes, own_colors = run_placement_phase(env, agent, rng)
        connection.send_line(make_set_command(own_red_codes))
        phase = ClientPhase.WAIT_SET_RESULT

        set_result = normalize_response(connection.recv_line())
        if set_result == "NG":
            raise ProtocolError("初期配置SETコマンドが拒否されました")
        if set_result != "OK":
            raise ProtocolError(f"SET結果が不正です: {set_result!r}")
        phase = ClientPhase.WAIT_BOARD

        expected_initial = make_initial_expected_board(own_red_codes)
        previous_board: BoardState | None = None
        first_board = True

        while phase is not ClientPhase.FINISHED:
            command = connection.recv_line()
            normalized = normalize_response(command)

            if normalized.startswith("WON:") or normalized.startswith("LST:"):
                # 相手の手で終局した場合も、サーバーは追加の行動を要求しない。
                phase = ClientPhase.FINISHED
                break

            kind, board = parse_board_command(normalized)
            if kind != "MOV":
                raise ProtocolError(f"WAIT_BOARD中にMOV以外を受信しました: {kind}")

            if first_board:
                if positions_equal(expected_initial, board):
                    # AIが先攻。配置後のcurrent_playerは0のままになっている想定。
                    pass
                else:
                    # AIが後攻。内部では先攻固定なのでpassしてから相手初手を反映する。
                    time_step = env.step([-1])
                    opponent_action, opponent_result = infer_opponent_action(
                        expected_initial, board, own_colors
                    )
                    time_step = env.step(
                        [add_action_result(opponent_action, opponent_result)]
                    )
                first_board = False
            else:
                if previous_board is None:
                    raise RuntimeError("前回盤面が未設定です")
                opponent_action, opponent_result = infer_opponent_action(
                    previous_board, board, own_colors
                )
                time_step = env.step(
                    [add_action_result(opponent_action, opponent_result)]
                )

            if time_step.last():
                # 通常はサーバー側のWON/LSTが先に来るが、内部判定が終局なら送信しない。
                phase = ClientPhase.FINISHED
                break

            if current_player(time_step) != 0:
                raise RuntimeError(
                    f"自分の行動選択時にcurrent_playerが0ではありません: "
                    f"{current_player(time_step)}"
                )

            action = choose_ppo_action(agent, time_step)
            move_command = make_move_command(action, board)
            connection.send_line(move_command)
            phase = ClientPhase.WAIT_MOVE_RESULT

            result_response = connection.recv_line()
            result = response_to_result(result_response)
            time_step = env.step([add_action_result(action, result)])
            previous_board = apply_own_move_to_snapshot(board, action, result)

            if time_step.last():
                # サーバーからWON/LSTが続けて送られる可能性があるため、1行だけ受け取る。
                final_command = normalize_response(connection.recv_line())
                if not (final_command.startswith("WON:") or final_command.startswith("LST:")):
                    raise ProtocolError(
                        f"内部終局後にWON/LST以外を受信しました: {final_command!r}"
                    )
                phase = ClientPhase.FINISHED
            else:
                phase = ClientPhase.WAIT_BOARD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenSpiel Geister AI用TCP通信クライアント"
    )
    parser.add_argument("host", help="クライアントアプリから渡される接続先IP")
    parser.add_argument("port", type=int, help="クライアントアプリから渡されるポート")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"学習済みモデルのパス（default: {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument("--seed", type=int, default=None, help="相手初期配置用乱数seed")
    args = parser.parse_args()

    main(args.host, args.port, args.model_path, args.seed)
