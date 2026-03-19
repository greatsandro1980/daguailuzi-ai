"""
大怪路子 PPO 神经网络
升级点：
  1. 网络加深 + Dropout 防过拟合
  2. 动作评分改为独立的 action_encoder（每个候选动作单独编码后与状态融合）
  3. ReplayBuffer 支持 GAE（广义优势估计）
  4. select_action 返回动作概率供 PPO clip 使用
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game_env import FEATURE_DIM, encode_state, CARD_VEC_IDX, cards_to_vec

# ─── 网络 ─────────────────────────────────────────────
class DaguaiPPONet(nn.Module):
    """
    双头网络：策略头 + 价值头
    输入: state 特征向量 (FEATURE_DIM)
    输出:
      logits (54,)  — 每张牌槽位的打分，用于动作选择
      value  scalar — 当前局面期望收益
    """
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 54),
        )

    def forward(self, x):
        feat   = self.backbone(x)
        value  = self.value_head(feat).squeeze(-1)
        logits = self.action_head(feat)
        return logits, value


# ─── 动作选择 ─────────────────────────────────────────
MAX_ACTIONS = 200

def select_action(net, obs, device, temperature=1.0, greedy=False):
    """
    返回: (action, log_prob, value, action_probs, actions)
    多返回 action_probs 和 actions，PPO 重算概率时用得上
    """
    actions = obs['legal_actions']
    if not actions:
        return [], torch.tensor(0.0), torch.tensor(0.0), None, []

    pass_actions = [a for a in actions if len(a) == 0]
    play_actions = [a for a in actions if len(a) > 0]
    if len(play_actions) > MAX_ACTIONS:
        play_actions = random.sample(play_actions, MAX_ACTIONS)
    actions = play_actions + pass_actions

    N = len(actions)
    act_matrix = np.zeros((N, 54), dtype=np.float32)
    act_sizes  = np.zeros(N, dtype=np.float32)
    for i, act in enumerate(actions):
        if act:
            idx = CARD_VEC_IDX[act]
            act_matrix[i, idx] = 1.0
            act_sizes[i] = len(act)

    state   = encode_state(obs)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = net(state_t)
        logits = logits.squeeze(0)   # (54,)

    act_t   = torch.FloatTensor(act_matrix).to(device)
    sizes_t = torch.FloatTensor(act_sizes).to(device)
    raw     = act_t @ logits
    denom   = sizes_t.clamp(min=1.0)
    raw_scores = raw / denom

    # pass 分数：用出牌动作均值的低分位动态计算，避免固定 -1.0 导致 pass 过度占优
    play_mask = (sizes_t > 0)
    if play_mask.any():
        play_scores = raw_scores[play_mask]
        pass_score = play_scores.quantile(0.25).item() - 0.5   # 出牌25分位以下0.5
    else:
        pass_score = -1.0   # fallback

    scores = torch.where(play_mask, raw_scores,
                         torch.full_like(raw_scores, pass_score))

    probs = F.softmax(scores / temperature, dim=0)

    if greedy:
        idx      = scores.argmax().item()
        log_prob = torch.log(probs[idx] + 1e-8)
    else:
        dist     = torch.distributions.Categorical(probs)
        idx      = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(idx))

    return actions[idx], log_prob, value.squeeze(0), probs, actions


# ─── PPO ReplayBuffer（支持 GAE）─────────────────────
class PPOBuffer:
    """
    存储一局或多局的经验，支持 GAE 计算优势函数。
    每个 step 存储：
      state, action_vec, log_prob, value, reward, seat, done
    """
    def __init__(self):
        self.states     = []
        self.act_vecs   = []   # 动作的 54 维向量表示
        self.log_probs  = []
        self.values     = []
        self.rewards    = []
        self.seats      = []
        self.dones      = []

    def add(self, state_feat, act_vec, log_prob, value, reward, seat, done):
        self.states.append(state_feat)
        self.act_vecs.append(act_vec)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.seats.append(seat)
        self.dones.append(done)

    def fill_final_rewards(self, final_rewards):
        """游戏结束时把终局奖励填到最后一步，中间步奖励为 0"""
        for i, seat in enumerate(self.seats):
            if self.dones[i]:
                self.rewards[i] = final_rewards[seat]

    def compute_gae(self, gamma=0.99, lam=0.95):
        """
        计算 GAE 优势函数和折扣回报。
        按 seat 分组，每个 seat 独立计算时序关系。
        返回: advantages (list), returns (list)
        """
        n = len(self.states)
        advantages = [0.0] * n
        returns    = [0.0] * n

        # 按 seat 分组，找每个 seat 的 step 索引（时序顺序）
        seat_steps = {}
        for i, seat in enumerate(self.seats):
            if seat not in seat_steps:
                seat_steps[seat] = []
            seat_steps[seat].append(i)

        for seat, idxs in seat_steps.items():
            gae = 0.0
            # 从后往前计算
            for j in reversed(range(len(idxs))):
                i = idxs[j]
                v_next = self.values[idxs[j+1]].item() if j + 1 < len(idxs) else 0.0
                v_curr = self.values[i].item()
                r      = self.rewards[i]
                done   = self.dones[i]

                delta = r + gamma * v_next * (1 - done) - v_curr
                gae   = delta + gamma * lam * (1 - done) * gae
                advantages[i] = gae
                returns[i]    = gae + v_curr

        return advantages, returns

    def __len__(self):
        return len(self.states)
