import numpy as np
import torch
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.pytorch.two_player_ppo import TwoPlayerPPO

GAME_DATA = "geister"
DATA_DIR = "geister_research/data"
MODEL_DATA_DIR = "model"
MODEL_NAME = "ver_1"

NUM_EVAL_GAMES = 10000  # 対戦回数

def evaluate():
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
    
    # 対戦相手となるランダムエージェントの準備 (便宜上 player_id=1 に固定)
    rand_agent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)
    
    ppo_wins = 0
    rand_wins = 0
    draws = 0

    print(f"Starting {NUM_EVAL_GAMES} evaluation games: PPO(Player 0) vs Random(Player 1)")

    for i in range(NUM_EVAL_GAMES):
        time_step = env.reset()
        
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            
            if current_player == 0:
                # PPOのターン
                # is_evaluation=True を指定することで、確率的なサンプリングではなく
                # 最も確率の高い行動を選ぶ（決定論的アプローチ）、または勾配計算をスキップします
                agent_output = ppo_agent.step([time_step], is_evaluation=True)
                action = agent_output[0].action
            elif current_player == 1:
                # ランダムエージェントのターン
                action = rand_agent.step(time_step).action
            else:
                raise ValueError("Unknown player")
                
            # 環境を1ステップ進める
            time_step = env.step([action])
            
        # 終局時の報酬で勝敗を判定
        rewards = time_step.rewards
        if rewards[0] > 0:
            ppo_wins += 1
        elif rewards[1] > 0:
            rand_wins += 1
        else:
            draws += 1
            
        if (i + 1) % 10 == 0:
            print(f"Played {i + 1} games... (PPO Wins: {ppo_wins}, Random: {rand_wins}, Draws: {draws})")
            
    win_rate = ppo_wins / NUM_EVAL_GAMES * 100
    print("-" * 30)
    print("Evaluation Results")
    print("-" * 30)
    print(f"PPO Win Rate   : {win_rate:.1f}%")
    print(f"Total PPO Wins : {ppo_wins}")
    print(f"Total Random   : {rand_wins}")
    print(f"Total Draws    : {draws}")

if __name__ == "__main__":
    evaluate()