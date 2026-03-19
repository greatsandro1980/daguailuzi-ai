"""
大怪路子超高速训练脚本
使用numba加速的游戏环境
目标：>50局/秒
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from game_env_fast_v2 import FastDaguaiEnvV2 as FastDaguaiEnv, recognize_cards, can_beat_cards

# ─── 配置 ─────────────────────────────────────────────
CFG = {
    'max_steps': 300,
    'batch_episodes': 50,     # 更多局才更新
    'ppo_epochs': 3,
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'temperature': 1.0,
    'max_actions': 30,        # 限制动作数量
}


# ─── 快速状态编码 ────────────────────────────────────
def fast_encode(hand, played, current, hand_sizes, trump_rank, my_team):
    """快速状态编码 - 224维"""
    feat = np.zeros(224, dtype=np.float32)
    feat[0:54] = hand.astype(np.float32)
    feat[54:108] = played.astype(np.float32) / 3.0
    feat[108:162] = current.astype(np.float32)
    feat[162:216] = (hand > 0).astype(np.float32)
    feat[216:222] = hand_sizes.astype(np.float32) / 27.0
    feat[222] = trump_rank / 14.0
    feat[223] = float(my_team)
    return feat


# ─── 简化策略网络 ────────────────────────────────────
class UltraLightNet(nn.Module):
    """超轻量级网络"""
    def __init__(self, state_dim=224, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.action = nn.Linear(hidden_dim, 54)
    
    def forward(self, x):
        feat = self.net(x)
        return self.action(feat), self.value(feat).squeeze(-1)


# ─── 经验池 ──────────────────────────────────────────
class FastBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


# ─── 快速动作选择 ────────────────────────────────────
def fast_select_action(net, obs, device, temperature=1.0):
    """快速动作选择"""
    actions = obs['legal_actions']
    if not actions:
        return None, 0.0, 0.0
    
    # 限制动作数量
    if len(actions) > CFG['max_actions']:
        # 优先选择出牌而非pass
        play_actions = [a for a in actions if np.sum(a) > 0]
        pass_actions = [a for a in actions if np.sum(a) == 0]
        
        if len(play_actions) > CFG['max_actions'] - 1:
            # 随机采样
            play_actions = random.sample(play_actions, CFG['max_actions'] - 1)
        
        actions = play_actions + (pass_actions[:1] if pass_actions else [])
    
    # 编码状态
    state = fast_encode(
        obs['hand'], obs['played_all'], obs['current_played'],
        obs['hand_sizes'], obs['trump_rank'], obs['my_team']
    )
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, value = net(state_t)
    
    # 计算每个动作的分数
    scores = []
    for action in actions:
        if np.sum(action) > 0:
            # 出牌分数 = 牌型权重 + 网络输出
            action_vec = torch.FloatTensor(action).to(device)
            net_score = (action_vec * logits).sum().item()
            # 简单启发式加分
            cards_played = int(np.sum(action))
            heuristic = cards_played * 0.5  # 出牌越多越好
            scores.append(net_score + heuristic)
        else:
            scores.append(-10.0)  # pass 低分
    
    # Softmax采样
    scores_t = torch.FloatTensor(scores)
    probs = F.softmax(scores_t / temperature, dim=0)
    idx = torch.multinomial(probs, 1).item()
    
    selected = actions[idx]
    log_prob = torch.log(probs[idx] + 1e-8).item()
    
    return selected, log_prob, value.item()


# ─── 收集经验 ────────────────────────────────────────
def collect_episode(net, env, device, temperature):
    """收集一局经验"""
    obs = env.reset()
    buffer = FastBuffer()
    
    while not env.done and env.steps < CFG['max_steps']:
        cp = obs['current_player']
        
        # 红队(0,2,4)用PPO，蓝队(1,3,5)用简单策略
        if cp % 2 == 0:
            action, log_prob, value = fast_select_action(net, obs, device, temperature)
            
            if action is None:
                action = np.zeros(54, dtype=np.int8)
            
            state = fast_encode(
                obs['hand'], obs['played_all'], obs['current_played'],
                obs['hand_sizes'], obs['trump_rank'], obs['my_team']
            )
            
            obs, rewards, done, _ = env.step(action)
            
            # 中间奖励
            step_reward = 0.01 * int(np.sum(action)) if np.sum(action) > 0 else -0.01
            if done:
                step_reward += rewards[cp]
            
            buffer.add(state, action.copy(), step_reward, value, done)
        else:
            # 蓝队：简单贪心
            actions = obs['legal_actions']
            if actions:
                # 优先出牌数多的
                play_actions = [a for a in actions if np.sum(a) > 0]
                if play_actions:
                    action = max(play_actions, key=lambda x: int(np.sum(x)))
                else:
                    action = actions[0] if actions else np.zeros(54, dtype=np.int8)
            else:
                action = np.zeros(54, dtype=np.int8)
            
            obs, _, done, _ = env.step(action)
    
    return buffer, rewards if env.done else np.zeros(6)


# ─── PPO更新 ────────────────────────────────────────
def ppo_update(net, optimizer, buffers, device):
    """PPO更新"""
    if not buffers:
        return 0.0
    
    total_loss = 0.0
    n_samples = 0
    
    # 合并所有buffer
    all_states = []
    all_actions = []
    all_returns = []
    all_values = []
    
    for buf in buffers:
        if len(buf) == 0:
            continue
        
        # 计算GAE
        rewards = np.array(buf.rewards)
        values = np.array(buf.values)
        dones = np.array(buf.dones)
        
        returns = np.zeros_like(rewards)
        adv = np.zeros_like(rewards)
        last_ret = 0
        last_adv = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + CFG['gamma'] * next_value * (1 - dones[t]) - values[t]
            adv[t] = delta + CFG['gamma'] * CFG['gae_lambda'] * (1 - dones[t]) * last_adv
            last_adv = adv[t]
            
            returns[t] = rewards[t] + CFG['gamma'] * (1 - dones[t]) * last_ret
            last_ret = returns[t]
        
        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for i in range(len(buf.states)):
            all_states.append(buf.states[i])
            all_actions.append(buf.actions[i])
            all_returns.append(returns[i])
            all_values.append(values[i])
    
    if not all_states:
        return 0.0
    
    # 批量训练
    states = torch.FloatTensor(np.array(all_states)).to(device)
    actions = torch.FloatTensor(np.array(all_actions)).to(device)
    returns_t = torch.FloatTensor(np.array(all_returns)).to(device)
    
    for _ in range(CFG['ppo_epochs']):
        logits, values = net(states)
        
        # 价值损失
        value_loss = F.mse_loss(values, returns_t)
        
        # 策略损失
        action_scores = (actions * logits).sum(dim=1)
        action_norms = actions.sum(dim=1).clamp(min=1.0)
        normalized_scores = action_scores / action_norms
        
        # 简化：直接用分数
        is_pass = (actions.sum(dim=1) == 0)
        normalized_scores = torch.where(is_pass, torch.full_like(normalized_scores, -1.0), normalized_scores)
        
        # 计算优势
        advantage = returns_t - values.detach()
        
        # 策略梯度损失
        policy_loss = -(normalized_scores * advantage).mean()
        
        loss = value_loss + 0.1 * policy_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        n_samples += 1
    
    return total_loss / max(n_samples, 1)


# ─── 主训练 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='.')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"✅ 使用设备: {device}")
    print(f"\n🚀 超高速训练模式，共 {args.episodes} 局")
    print(f"   目标速度: >50 局/秒")
    
    # 初始化
    net = UltraLightNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG['lr'])
    env = FastDaguaiEnv()
    
    # 统计
    reward_history = deque(maxlen=100)
    win_history = deque(maxlen=100)
    
    start_time = datetime.now()
    buffers = []
    last_print = 0
    
    for ep in range(args.episodes):
        # 动态温度
        progress = ep / args.episodes
        temperature = max(0.3, 1.0 * (1 - progress * 0.7))
        
        # 收集经验
        buffer, rewards = collect_episode(net, env, device, temperature)
        buffers.append(buffer)
        
        # 统计
        red_reward = np.mean([rewards[s] for s in [0, 2, 4]])
        reward_history.append(red_reward)
        win_history.append(1.0 if red_reward > 0 else 0.0)
        
        # 批量更新
        if (ep + 1) % CFG['batch_episodes'] == 0:
            loss = ppo_update(net, optimizer, buffers, device)
            buffers = []
        
        # 打印进度
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed - last_print >= 5:  # 每5秒打印一次
            avg_reward = np.mean(reward_history)
            win_rate = np.mean(win_history)
            eps = (ep + 1) / elapsed
            
            print(f"Ep {ep+1}/{args.episodes} | "
                  f"奖励: {avg_reward:+.3f} | "
                  f"胜率: {win_rate:.1%} | "
                  f"速度: {eps:.1f} eps/s")
            last_print = elapsed
        
        # 定期保存
        if (ep + 1) % 2000 == 0:
            save_path = os.path.join(args.save_dir, f'model_fast_ep{ep+1}.pt')
            torch.save({
                'model': net.state_dict(),
                'episode': ep + 1,
                'win_rate': np.mean(win_history),
            }, save_path)
            print(f"💾 已保存: {save_path}")
    
    # 最终保存
    final_path = os.path.join(args.save_dir, 'model_fast_final.pt')
    torch.save({
        'model': net.state_dict(),
        'episode': args.episodes,
        'win_rate': np.mean(win_history),
    }, final_path)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ 训练完成！")
    print(f"   总局数: {args.episodes}")
    print(f"   总耗时: {elapsed:.1f} 秒")
    print(f"   平均速度: {args.episodes/elapsed:.1f} 局/秒")
    print(f"   最终胜率: {np.mean(win_history):.1%}")
    print(f"   模型文件: {final_path}")


if __name__ == '__main__':
    main()
