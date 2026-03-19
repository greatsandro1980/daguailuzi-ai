"""
大怪路子改进版训练脚本
整合：增强版特征 + 奖励塑形 + 对手池多样化

训练流程：
  阶段1 (0~20%)  : PPO vs 随机/贪心对手
  阶段2 (20~50%) : PPO vs 智能规则对手
  阶段3 (50~100%): 纯自我对弈
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 导入增强版环境
from game_env_enhanced import (
    DaguaiEnvEnhanced, encode_state_enhanced,
    compute_step_reward
)
from game_env import TEAM_MAP, recognize, TYPE_ORDER

# 动态获取实际特征维度
_env_test = DaguaiEnvEnhanced()
_obs_test = _env_test.reset()
FEATURE_DIM_ACTUAL = encode_state_enhanced(_obs_test, _env_test.game_history).shape[0]
del _env_test, _obs_test

# 导入对手池
from opponents import (
    OpponentPool, RandomPlayer, GreedyPlayer,
    ConservativePlayer, AggressivePlayer, TeamAwarePlayer, SmartPlayer
)

# ─── 网络定义 ─────────────────────────────────────────
class DaguaiNet(nn.Module):
    """改进版网络，适配增强特征"""
    def __init__(self, input_dim=FEATURE_DIM_ACTUAL, hidden_dim=512):
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
        feat = self.backbone(x)
        value = self.value_head(feat).squeeze(-1)
        logits = self.action_head(feat)
        return logits, value


# ─── 经验池 ──────────────────────────────────────────
class PPOBuffer:
    """PPO经验池，支持GAE"""
    def __init__(self):
        self.states = []
        self.act_vecs = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.seats = []
        self.dones = []

    def add(self, state_feat, act_vec, log_prob, value, reward, seat, done):
        self.states.append(state_feat)
        self.act_vecs.append(act_vec)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.seats.append(seat)
        self.dones.append(done)

    def compute_gae(self, gamma=0.99, lam=0.95):
        n = len(self.states)
        advantages = [0.0] * n
        returns = [0.0] * n

        seat_steps = {}
        for i, seat in enumerate(self.seats):
            if seat not in seat_steps:
                seat_steps[seat] = []
            seat_steps[seat].append(i)

        for seat, idxs in seat_steps.items():
            gae = 0.0
            for j in reversed(range(len(idxs))):
                i = idxs[j]
                v_next = self.values[idxs[j+1]].item() if j + 1 < len(idxs) else 0.0
                v_curr = self.values[i].item()
                r = self.rewards[i]
                done = self.dones[i]

                delta = r + gamma * v_next * (1 - done) - v_curr
                gae = delta + gamma * lam * (1 - done) * gae
                advantages[i] = gae
                returns[i] = gae + v_curr

        return advantages, returns

    def __len__(self):
        return len(self.states)


# ─── 动作选择 ─────────────────────────────────────────
MAX_ACTIONS = 200  # 最多保留200个候选动作

def cards_to_vec(cards):
    """牌组转54维向量"""
    vec = np.zeros(54, dtype=np.float32)
    for c in cards:
        if c.get('is_big_joker'):
            vec[53] = 1.0
        elif c.get('is_small_joker'):
            vec[52] = 1.0
        else:
            from game_env import SUITS, RANKS
            si = SUITS.index(c['suit'])
            ri = RANKS.index(c['rank'])
            vec[si * 13 + ri] = 1.0
    return vec


def evaluate_action(cards, trump_rank=None):
    """快速评估动作质量，用于筛选"""
    if not cards:
        return -100
    from game_env import recognize, TYPE_ORDER
    ct = recognize(cards, trump_rank)
    if not ct:
        return -50
    return TYPE_ORDER[ct[0]] * 100 + ct[1]


def select_action(net, obs, device, temperature=1.0):
    """选择动作，返回 (action, log_prob, value)"""
    actions = obs['legal_actions']
    if not actions:
        return [], torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # 分离 pass 和出牌动作
    pass_actions = [a for a in actions if len(a) == 0]
    play_actions = [a for a in actions if len(a) > 0]

    # 动作数量太多时智能剪枝
    if len(play_actions) > MAX_ACTIONS:
        # 按牌力快速评估，保留高价值动作
        trump_rank = obs.get('trump_rank')
        scored = [(a, evaluate_action(a, trump_rank)) for a in play_actions]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 保留前100个高价值 + 随机100个
        top_100 = [s[0] for s in scored[:100]]
        remaining = [s[0] for s in scored[100:]]
        if remaining:
            random_100 = random.sample(remaining, min(100, len(remaining)))
            play_actions = top_100 + random_100
        else:
            play_actions = top_100

    actions = play_actions + pass_actions

    N = len(actions)
    act_matrix = np.zeros((N, 54), dtype=np.float32)
    act_sizes = np.zeros(N, dtype=np.float32)
    for i, act in enumerate(actions):
        if act:
            act_matrix[i] = cards_to_vec(act)
            act_sizes[i] = len(act)

    # 编码状态
    state = encode_state_enhanced(obs, obs.get('game_history'))
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = net(state_t)
        logits = logits.squeeze(0)

    act_t = torch.FloatTensor(act_matrix).to(device)
    sizes_t = torch.FloatTensor(act_sizes).to(device)

    raw_scores = act_t @ logits
    denom = sizes_t.clamp(min=1.0)
    scores = torch.where(sizes_t > 0, raw_scores / denom,
                         torch.full_like(raw_scores, -1.0))

    probs = F.softmax(scores / temperature, dim=0)
    dist = torch.distributions.Categorical(probs)
    idx = dist.sample()
    log_prob = dist.log_prob(idx)

    return actions[idx.item()], log_prob, value.squeeze(0)


# ─── 训练配置 ─────────────────────────────────────────
CFG = {
    'lr': 3e-4,
    'total_episodes': 50000,
    'save_interval': 1000,
    'log_interval': 100,
    'hidden_dim': 512,
    'ppo_epochs': 4,
    'clip_eps': 0.2,
    'batch_episodes': 16,
    'max_steps': 2000,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_coef': 0.5,
    'entropy_coef': 0.02,
    'team_reward_coef': 0.3,
    'temperature': 1.2,
    'min_temperature': 0.3,
    'stage1_end': 0.2,
    'stage2_end': 0.5,
}


# ─── 收集一局经验 ─────────────────────────────────────
def collect_episode(net, env, device, temperature, stage, opponent_pool):
    """
    跑一局，收集经验
    stage: 'random' / 'rule' / 'self'
    """
    obs = env.reset()
    buffer = PPOBuffer()
    steps = 0

    # 蓝队席位
    BLUE_SEATS = {1, 3, 5}

    # 根据阶段选择对手
    progress = random.random()  # 简化：用随机数决定对手
    if stage == 'random':
        blue_opponents = {s: opponent_pool.sample(0.1) for s in BLUE_SEATS}
    elif stage == 'rule':
        blue_opponents = {s: opponent_pool.sample(0.4) for s in BLUE_SEATS}
    else:
        blue_opponents = None  # 自我对弈

    while not env.done and steps < CFG['max_steps']:
        cp = obs['current_player']
        is_blue = cp in BLUE_SEATS

        if stage != 'self' and is_blue and blue_opponents:
            # 对手出牌
            opponent = blue_opponents.get(cp)
            if opponent:
                action = opponent.select_action(obs)
            else:
                action = random.choice(obs['legal_actions'])
            obs, rewards, done, info = env.step(action)
        else:
            # PPO出牌，记录经验
            state_feat = encode_state_enhanced(obs, env.game_history)
            action, log_prob, value = select_action(net, obs, device, temperature)

            # 计算中间奖励
            step_reward = compute_step_reward(env, action, obs, env.game_history)

            # 执行动作
            obs, rewards, done, info = env.step(action)

            # 如果游戏结束，加上最终奖励
            if done:
                final_reward = rewards[cp]
                step_reward += final_reward

            # 记录经验
            act_vec = cards_to_vec(action) if action else np.zeros(54, dtype=np.float32)
            buffer.add(
                state_feat=state_feat,
                act_vec=act_vec,
                log_prob=log_prob,
                value=value,
                reward=step_reward,
                seat=cp,
                done=done,
            )

        steps += 1

    return buffer, rewards


# ─── PPO损失计算 ─────────────────────────────────────
def compute_ppo_loss(net, buffer, device, clip_eps, value_coef, entropy_coef):
    if len(buffer) == 0:
        return None

    advantages, returns = buffer.compute_gae(CFG['gamma'], CFG['gae_lambda'])

    states = torch.FloatTensor(np.array(buffer.states)).to(device)
    act_vecs = torch.FloatTensor(np.array(buffer.act_vecs)).to(device)
    old_lp = torch.stack(buffer.log_probs).to(device).detach()
    returns_t = torch.FloatTensor(returns).to(device)
    adv_t = torch.FloatTensor(advantages).to(device)

    if adv_t.std() > 1e-8:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    logits, values = net(states)

    raw_scores = (act_vecs * logits).sum(dim=1)
    act_norms = act_vecs.sum(dim=1).clamp(min=1.0)
    scores = raw_scores / act_norms

    is_pass = (act_vecs.sum(dim=1) == 0)
    scores = torch.where(is_pass, torch.full_like(scores, -1.0), scores)

    # 计算新的 log_prob
    probs = F.softmax(scores.unsqueeze(1), dim=0).squeeze(1)
    new_lp = torch.log(probs + 1e-8)

    # PPO Clip
    ratio = torch.exp(new_lp - old_lp)
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
    actor_loss = -torch.min(surr1, surr2).mean()

    # 价值损失
    value_loss = F.mse_loss(values, returns_t)

    # 熵正则
    entropy = -(new_lp * torch.exp(new_lp)).mean()
    entropy_loss = -entropy

    total_loss = actor_loss + value_coef * value_loss + entropy_coef * entropy_loss

    return total_loss, actor_loss.item(), value_loss.item(), entropy.item()


# ─── 主训练循环 ───────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    net = DaguaiNet(hidden_dim=CFG['hidden_dim']).to(device)
    optimizer = Adam(net.parameters(), lr=CFG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)

    # 加载已有模型
    start_ep = 0
    if args.resume:
        if os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            net.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_ep = ckpt.get('episode', 0)
            print(f"✅ 从 episode {start_ep} 恢复训练")
        else:
            print(f"⚠️  模型文件不存在: {args.resume}")

    # 对手池
    opponent_pool = OpponentPool()

    # 保存目录
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_improved')
    os.makedirs(save_dir, exist_ok=True)

    env = DaguaiEnvEnhanced()
    reward_history = []
    win_history = []
    loss_history = []
    accumulated = []
    t0 = time.time()
    last_stage = None

    print(f"\n🚀 开始改进版训练，共 {args.episodes} 局")
    print(f"   特征维度: {FEATURE_DIM_ACTUAL}")
    print(f"   阶段1 (0~{int(args.episodes*CFG['stage1_end'])}局): PPO vs 随机/贪心对手")
    print(f"   阶段2 ({int(args.episodes*CFG['stage1_end'])}~{int(args.episodes*CFG['stage2_end'])}局): PPO vs 智能规则对手")
    print(f"   阶段3 ({int(args.episodes*CFG['stage2_end'])}局~结束): 纯自我对弈\n")

    for ep in range(start_ep, args.episodes):
        progress = ep / args.episodes
        temperature = max(CFG['min_temperature'], CFG['temperature'] * (1 - progress * 0.7))
        entropy_coef = max(0.005, CFG['entropy_coef'] * (1 - progress * 0.5))

        # 确定训练阶段
        if progress < CFG['stage1_end']:
            stage = 'random'
        elif progress < CFG['stage2_end']:
            stage = 'rule'
        else:
            stage = 'self'

        if stage != last_stage:
            stage_name = {'random': '阶段1: PPO vs 随机对手',
                          'rule': '阶段2: PPO vs 智能规则',
                          'self': '阶段3: 纯自我对弈'}
            print(f"\n>>> 切换到 {stage_name[stage]} (ep {ep+1})\n")
            last_stage = stage

        # 收集一局经验
        net.eval()
        buffer, final_rewards = collect_episode(
            net, env, device, temperature, stage, opponent_pool
        )
        accumulated.append(buffer)

        # 统计
        red_reward = np.mean([final_rewards[s] for s in [0, 2, 4]])
        reward_history.append(red_reward)
        red_team_win = final_rewards[0] > final_rewards[1]
        win_history.append(1.0 if red_team_win else 0.0)

        # 批量更新
        if (ep + 1) % CFG['batch_episodes'] == 0:
            net.train()
            total_loss_val = 0.0
            n_updates = 0

            for _ in range(CFG['ppo_epochs']):
                optimizer.zero_grad()
                for buf in accumulated:
                    result = compute_ppo_loss(
                        net, buf, device,
                        clip_eps=CFG['clip_eps'],
                        value_coef=CFG['value_coef'],
                        entropy_coef=entropy_coef,
                    )
                    if result:
                        loss, al, vl, ent = result
                        loss.backward()
                        total_loss_val += loss.item()
                        n_updates += 1

                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

            scheduler.step()
            if n_updates > 0:
                loss_history.append(total_loss_val / n_updates)
            accumulated = []
            net.eval()

        # 日志
        if (ep + 1) % CFG['log_interval'] == 0:
            elapsed = time.time() - t0
            avg_reward = np.mean(reward_history[-CFG['log_interval']:])
            win_rate = np.mean(win_history[-CFG['log_interval']:])
            avg_loss = np.mean(loss_history[-10:]) if loss_history else 0.0
            eps_per_sec = CFG['log_interval'] / elapsed
            t0 = time.time()

            print(f"[ep {ep+1:>6}/{args.episodes}] [{stage}] "
                  f"均值奖励={avg_reward:+.3f}  "
                  f"胜率={win_rate:.1%}  "
                  f"loss={avg_loss:.4f}  "
                  f"temp={temperature:.2f}  "
                  f"速度={eps_per_sec:.1f}局/s")

        # 保存模型
        if (ep + 1) % CFG['save_interval'] == 0:
            path = os.path.join(save_dir, f'improved_ep{ep+1}.pt')
            torch.save({
                'episode': ep + 1,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': CFG,
            }, path)

            # 同时保存为最新模型
            latest = os.path.join(save_dir, 'latest.pt')
            torch.save({
                'episode': ep + 1,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': CFG,
            }, latest)

            print(f"  💾 已保存: {path}")

    # 训练完成，保存最终模型
    final_path = os.path.join(save_dir, 'final.pt')
    torch.save({
        'episode': args.episodes,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'cfg': CFG,
    }, final_path)

    print(f"\n✅ 训练完成！最终模型已保存: {final_path}")
    print(f"   最终胜率: {np.mean(win_history[-1000:]):.1%}")

    return net


# ─── 入口 ───────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--episodes', type=int, default=CFG['total_episodes'], help='训练局数')
    args = parser.parse_args()

    train(args)
