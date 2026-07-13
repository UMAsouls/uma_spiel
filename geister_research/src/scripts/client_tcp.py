import numpy as np
import torch
import pyspiel

import socket

from numpy.typing import NDArray

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.pytorch.two_player_ppo import TwoPlayerPPO

MAX_COL = 6
MAX_ROW = 6
BOARD_SIZE = MAX_COL*MAX_ROW

GAME_DATA = "geister"
DATA_DIR = "geister_research/data"
MODEL_DATA_DIR = "model"
MODEL_NAME = "ver_1"

GHOST_POS = np.zeros((BOARD_SIZE))
GHOST_POS[np.arange(4*MAX_COL+1,4*MAX_COL+5)] = np.arange(1,5)
GHOST_POS[np.arange(5*MAX_COL+1,5*MAX_COL+5)] = np.arange(5,9)

GHOST_DICT = [chr(ord("A") + i) for i in range(8)]

MOVE_DIR_NAME_DICT = [
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST"
]

MOVE_DIR_DICT = [
    [1,0],
    [-1,0],
    [0,1],
    [0,-1]
]

BUFSIZE = 4096
FORMAT = "utf-8"

def make_set_command(pos:NDArray[np.bool_]):
    assert pos.sum() == 4
    
    ghosts = GHOST_POS[np.where(pos)[0]]
    
    cmd = ""
    for g in ghosts:
        cmd += GHOST_DICT[g-1]
        
    return cmd

def make_move_command(action: int, board_info: list[list[str]]):
    mv_dir = action // BOARD_SIZE
    mv_dir_name = MOVE_DIR_NAME_DICT[mv_dir]
    mv_pos = action % BOARD_SIZE
    mv_pos_x = mv_pos % MAX_COL
    mv_pos_y = mv_pos // MAX_COL
    
    ghost = board_info[mv_pos_y][mv_pos_x]
    
    cmd = f"MOV:{ghost},{mv_dir_name}\r\n"
    
    return cmd

def make_board_from_server_cmd(cmd:str):
    board: list[list[str]] = [[""]*MAX_COL for _ in range(MAX_ROW)]
    
    

def main(host, port):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host,port))
    
    data = client.recv(BUFSIZE)
    cmd = data.decode(FORMAT)
    
    game = pyspiel.load_game(GAME_DATA)
    env = rl_environment.Environment(game=game)
    
    info_state_shape = tuple(np.array(env.observation_spec()["info_state"]).flatten())
    num_actions = game.num_distinct_actions()
    
    # 評価用なので num_envs=1 でPPOエージェントを初期化
    ppo_agent = TwoPlayerPPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=game.num_players(),
        num_envs=1,
        agent_fn=PPOAgent,
        device="cpu"
    )
    
    # 学習済みモデルの重みをロード
    model_path = f"{DATA_DIR}/{MODEL_DATA_DIR}/{MODEL_NAME}"
    ppo_agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    # ネットワークを評価モード(DropoutやBatchNormの固定)に切り替え
    ppo_agent.eval()
    
    time_step = env.reset() 
    
    
    client.close()