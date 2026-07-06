import random
from absl.testing import absltest
import numpy as np
import torch
import os

from torch.utils.tensorboard import SummaryWriter  

from open_spiel.python import rl_environment
import pyspiel
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.two_player_ppo import TwoPlayerPPO
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.vector_env import SyncVectorEnv

# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
GAME_DATA = "geister"

MAX_PROCESS = 3
TOTAL_TIME_STEP = 1000000
STEPS_PER_BATCH = 10000

DATA_DIR = "geister_research/data"
MODEL_NAME = "ver_1"
LOG_DATA_DIR = "log"
MODEL_DATA_DIR = "model"


def learn():
    game = pyspiel.load_game(GAME_DATA)
    env = rl_environment.Environment(game=game)
    envs = SyncVectorEnv(
      [rl_environment.Environment(game=game) for i in range(MAX_PROCESS)]
    )
    agent_fn = PPOAgent
    anneal_lr = True

    info_state_shape = tuple(
        np.array(env.observation_spec()["info_state"]).flatten())
    
    writer = SummaryWriter(f"{DATA_DIR}/{LOG_DATA_DIR}/{MODEL_NAME}")

    total_timesteps = TOTAL_TIME_STEP
    steps_per_batch = STEPS_PER_BATCH
    batch_size = int(len(envs) * steps_per_batch)
    num_updates = total_timesteps // batch_size
    agent = TwoPlayerPPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        num_envs=MAX_PROCESS,
        agent_fn=agent_fn,
        steps_per_batch = STEPS_PER_BATCH,
        writer = writer
      )

    time_step = envs.reset()
    idx = 0
    for update in range(num_updates):
      for _ in range(steps_per_batch):
        agent_output = agent.step(time_step)
        time_step, reward, done, unreset_time_steps = envs.step(
            agent_output, reset_if_done=True)
        
        agent.post_step(unreset_time_steps, done)

      if anneal_lr:
        agent.anneal_learning_rate(update, num_updates)

      agent.learn(unreset_time_steps)
      print("learn", idx*steps_per_batch*len(envs))
      idx += 1
      
    writer.close()  # 最後に閉じる
    os.makedirs(f"{DATA_DIR}/{MODEL_DATA_DIR}", exist_ok=True)
    torch.save(agent.state_dict(), f"{DATA_DIR}/{MODEL_DATA_DIR}/{MODEL_NAME}")

if __name__ == "__main__":
  learn()





