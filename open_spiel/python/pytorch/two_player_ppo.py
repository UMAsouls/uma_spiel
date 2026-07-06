import torch
from torch import nn
from torch import optim
import time
import numpy as np
from open_spiel.python.rl_agent import StepOutput

from open_spiel.python.rl_environment import TimeStep

# ppo.py で定義されている PPO クラスや関数がインポートされている前提
from open_spiel.python.pytorch.ppo import PPO, legal_actions_to_mask

class TwoPlayerPPO(PPO):
    """2人対戦ゲーム（自己対戦）に対応するためのPPO拡張クラス"""

    def __init__(self, *args, **kwargs):
        # player_id は不要になるため、もし kwargs に含まれていれば削除
        if 'player_id' in kwargs:
            del kwargs['player_id']
            
        # 親クラスの初期化。これでニューラルネットやバッファが構築される
        super().__init__(*args, **kwargs)
        
        # 追加: 各ステップで「誰のターンだったか」を記録するトラッカー
        self.step_players = torch.zeros(
            (self.steps_per_batch, self.num_envs), dtype=torch.long
        ).to(self.device)
        
        # 将来的にプレイヤーごとのトラジェクトリ（軌跡）を分離して
        # learn() を呼び出す場合は、ここに独自のバッファを追加していくと綺麗です

    def step(self, time_step, is_evaluation=False):
        """
        各環境の現在のターンプレイヤーの視点で状態を取得し、行動を決定する。
        """
        # 1. 各環境の現在手番のプレイヤーIDを取得
        # OpenSpielでは終端状態の時に current_player() が負の値(-4など)を返すため、
        # 配列のインデックスエラーを防ぐための安全策として、負の場合は便宜上 0 を使うようにします
        current_players = [
            ts.current_player() if ts.current_player() >= 0 else 0 
            for ts in time_step
        ]

        if is_evaluation:
            with torch.no_grad():
                # self.player_id ではなく、各環境の current_players[i] の情報を抽出
                legal_actions_mask = legal_actions_to_mask([
                    ts.observations["legal_actions"][p] 
                    for ts, p in zip(time_step, current_players)
                ], self.num_actions).to(self.device)
                
                obs = torch.Tensor(np.array([
                    np.reshape(ts.observations["info_state"][p], self.input_shape) 
                    for ts, p in zip(time_step, current_players)
                ])).to(self.device)
                
                action, _, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask)
                    
                return [
                    StepOutput(action=a.item(), probs=p)
                    for (a, p) in zip(action, probs)
                ]
        else:
            with torch.no_grad():
                # 訓練時も同様に現在のプレイヤー視点で情報を抽出
                legal_actions_mask = legal_actions_to_mask([
                    ts.observations["legal_actions"][p] 
                    for ts, p in zip(time_step, current_players)
                ], self.num_actions).to(self.device)
                
                obs = torch.Tensor(np.array([
                    np.reshape(ts.observations["info_state"][p], self.input_shape) 
                    for ts, p in zip(time_step, current_players)
                ])).to(self.device)
                
                action, logprob, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask)

                # 親クラスが用意しているバッファに保存
                self.legal_actions_mask[self.cur_batch_idx] = legal_actions_mask
                self.obs[self.cur_batch_idx] = obs
                self.actions[self.cur_batch_idx] = action
                self.logprobs[self.cur_batch_idx] = logprob
                self.values[self.cur_batch_idx] = value.flatten()
                
                # ここが追加ポイント: 行動したプレイヤーIDを記憶
                self.step_players[self.cur_batch_idx] = torch.tensor(current_players).to(self.device)

                agent_output = [
                    StepOutput(action=a.item(), probs=p)
                    for (a, p) in zip(action, probs)
                ]
                return agent_output
            
    # post_stepの引数を、単一のrewardではなく time_step を受け取るように変更します
    def post_step(self, time_step:list[TimeStep], done):
        """
        環境が1ステップ進んだ後の状態(time_step)を受け取り、
        直前に行動したプレイヤーの視点での報酬を抽出して保存します。
        """
        # 直前のステップで行動したプレイヤーのリストを取得
        acted_players = self.step_players[self.cur_batch_idx]
        
        # 終局時など、各プレイヤーの報酬が配列で入っているので、行動したプレイヤーの分だけを抽出
        rewards = [
            ts.rewards[p.item()] if len(ts.rewards) > 0 else 0.0
            for ts, p in zip(time_step, acted_players)
        ]
        
        self.rewards[self.cur_batch_idx] = torch.tensor(rewards).to(self.device).view(-1)
        self.dones[self.cur_batch_idx] = torch.tensor(done).to(self.device).view(-1)

        self.total_steps_done += self.num_envs
        self.cur_batch_idx += 1
        
    def learn(self, time_step):
        """
        論理バッファ分離を用いたTD誤差（GAE）の計算とネットワーク更新
        """
        # 最新の状態の現在プレイヤーを取得
        final_players = [
            ts.current_player() if ts.current_player() >= 0 else 0 
            for ts in time_step
        ]
        
        next_obs = torch.Tensor(np.array([
            np.reshape(ts.observations["info_state"][p], self.input_shape) 
            for ts, p in zip(time_step, final_players)
        ])).to(self.device)

        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            
            # --- 論理バッファ分離によるGAE計算 ---
            # プレイヤーごと(0, 1)に、次の状態の価値とGAEを追跡するトラッカーを作成
            next_val_tracker = torch.zeros((self.num_players, self.num_envs)).to(self.device)
            last_gae_tracker = torch.zeros((self.num_players, self.num_envs)).to(self.device)
            next_nonterm_tracker = torch.ones((self.num_players, self.num_envs)).to(self.device)
            
            # 最新状態の価値を、該当するプレイヤーのトラッカーにセット
            for i, p in enumerate(final_players):
                next_val_tracker[p, i] = next_value[0, i]

            # 時間を遡って各ステップのGAEを計算
            env_indices = torch.arange(self.num_envs).to(self.device)
            for t in reversed(range(self.steps_per_batch)):
                p_t = self.step_players[t] # そのステップで行動したプレイヤー
                
                # 該当プレイヤーの「次の価値」と「次の終了判定」を抽出
                nv = next_val_tracker[p_t, env_indices]
                nt = next_nonterm_tracker[p_t, env_indices]
                lg = last_gae_tracker[p_t, env_indices]
                
                delta = self.rewards[t] + self.gamma * nv * nt - self.values[t]
                
                if self.gae:
                    adv = delta + self.gamma * self.gae_lambda * nt * lg
                else:
                    # GAEを使わない標準TD収益の場合のフォールバック
                    adv = delta
                
                advantages[t] = adv
                
                # トラッカーを「今のステップの価値」で更新 (次のループの過去から見れば、これが"未来"になる)
                next_val_tracker[p_t, env_indices] = self.values[t]
                last_gae_tracker[p_t, env_indices] = adv
                next_nonterm_tracker[p_t, env_indices] = 1.0 - self.dones[t]

            returns = advantages + self.values

        # --- 以下、元の PPO クラスの最適化処理をそのまま実行 ---
        b_legal_actions_mask = self.legal_actions_mask.reshape((-1, self.num_actions))
        b_obs = self.obs.reshape((-1,) + self.input_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
                    b_obs[mb_inds],
                    legal_actions_mask=b_legal_actions_mask[mb_inds],
                    action=b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.writer is not None:
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.total_steps_done)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.total_steps_done)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.total_steps_done)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.total_steps_done)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.total_steps_done)
            self.writer.add_scalar("charts/SPS", int(self.total_steps_done / (time.time() - self.start_time)), self.total_steps_done)

        self.updates_done += 1
        self.cur_batch_idx = 0