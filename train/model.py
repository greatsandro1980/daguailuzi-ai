"""
大怪路子强化学习神经网络
架构：双头网络（策略头 + 价值头）
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game_env import FEATURE_DIM, encode_state, legal_actions, cards_to_vec, CARD_VEC_IDX

# ─── 网络结构 ─────────────────────────────────────────
class DaguaiNet(nn.Module):
    """
    输入: state 特征向量 (FEATURE_DIM=224)
    输出:
      - value: 标量，预测当前局面对本玩家的期望收益
      - action_logits: 对每张牌"是否出"的评分（54维，用于动作选择）
    """
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=512):
        super().__init__()
        # 共享主干
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
        )
        # 价值头：预测期望奖励
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),   # 输出在 [-1, 1]
        )
        # 动作头：对 54 个牌位打分
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 54),
        )

    def forward(self, x):
        feat  = self.backbone(x)
        value = self.value_head(feat).squeeze(-1)
        logits= self.action_head(feat)
        return logits, value


# ─── 动作选择 ─────────────────────────────────────────
def action_to_vec(action):
    """把一个动作（card list）转成 54 维 0/1 向量"""
    return cards_to_vec(action)


MAX_ACTIONS = 200   # 最多保留200个候选动作，超出则随机采样（加速训练）

def select_action(net, obs, device, temperature=1.0, greedy=False):
    """
    给定观测，用网络选择一个动作。
    返回: (action, log_prob, value)

    优化：将所有动作编码成矩阵，一次矩阵乘法代替循环，速度提升 100x。
    """
    actions = obs['legal_actions']
    if not actions:
        return [], torch.tensor(0.0), torch.tensor(0.0)

    # 动作数量太多时随机剪枝（保留 pass 动作）
    pass_actions  = [a for a in actions if len(a) == 0]
    play_actions  = [a for a in actions if len(a) > 0]
    if len(play_actions) > MAX_ACTIONS:
        play_actions = random.sample(play_actions, MAX_ACTIONS)
    actions = play_actions + pass_actions

    # 批量构建动作矩阵 (N, 54)
    N = len(actions)
    act_matrix = np.zeros((N, 54), dtype=np.float32)
    act_sizes  = np.zeros(N, dtype=np.float32)
    for i, act in enumerate(actions):
        if act:
            idx = CARD_VEC_IDX[act]
            act_matrix[i, idx] = 1.0
            act_sizes[i] = len(act)

    # 编码状态，一次前向传播
    state   = encode_state(obs)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = net(state_t)
        logits = logits.squeeze(0)   # (54,)

    # 批量评分：(N, 54) @ (54,) → (N,)，再按动作长度归一化
    act_t    = torch.FloatTensor(act_matrix).to(device)
    sizes_t  = torch.FloatTensor(act_sizes).to(device)
    raw_scores = act_t @ logits                             # (N,)
    # pass 动作得分固定为 -1；出牌动作取均值
    denom    = sizes_t.clamp(min=1.0)
    scores_t = torch.where(sizes_t > 0, raw_scores / denom,
                           torch.full_like(raw_scores, -1.0))

    if greedy:
        idx = scores_t.argmax().item()
        log_prob = torch.tensor(0.0)
    else:
        probs = F.softmax(scores_t / temperature, dim=0)
        dist  = torch.distributions.Categorical(probs)
        idx   = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(idx))

    return actions[idx], log_prob, value.squeeze(0)


# ─── 经验池 ──────────────────────────────────────────
class ReplayBuffer:
    """存储一局游戏中每一步的经验"""
    def __init__(self):
        self.states     = []   # feature vectors
        self.log_probs  = []
        self.values     = []
        self.rewards    = []   # 每步奖励（游戏结束时填入）
        self.seats      = []   # 哪个玩家的经验

    def add(self, state_feat, log_prob, value, seat):
        self.states.append(state_feat)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(0.0)   # 占位，结束后填
        self.seats.append(seat)

    def fill_rewards(self, final_rewards):
        """游戏结束时，把最终奖励填入每步（简化版：所有步共享终局奖励）"""
        for i, seat in enumerate(self.seats):
            self.rewards[i] = final_rewards[seat]

    def __len__(self):
        return len(self.states)


# ─── 简单测试 ─────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    net = DaguaiNet().to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"网络参数量: {total_params:,}")

    # 测试前向传播
    dummy = torch.randn(4, FEATURE_DIM).to(device)
    logits, value = net(dummy)
    print(f"logits shape: {logits.shape}")   # (4, 54)
    print(f"value shape: {value.shape}")     # (4,)

    # 测试动作选择
    from game_env import DaguaiEnv
    env = DaguaiEnv()
    obs = env.reset()
    action, lp, val = select_action(net, obs, device)
    print(f"选择动作: {len(action)} 张牌, log_prob={lp:.4f}, value={val:.4f}")
