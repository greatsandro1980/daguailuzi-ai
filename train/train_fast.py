"""
快速训练脚本 - 简化版
优化：减少特征维度、简化动作选择、降低游戏复杂度
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

from game_env import DaguaiEnv, cards_to_vec, encode_state, FEATURE_DIM

# ─── 配置 ─────────────────────────────────────────────
CFG = {
    'max_steps': 200,           # 减少步数
    'batch_episodes': 10,       # 更频繁更新
    'ppo_epochs': 2,            # 减少PPO迭代
    'lr': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'temperature': 1.0,
    'max_actions': 50,          # 限制动作数量
}


# ─── 简化网络 ────────────────────────────────────────
class SimpleNet(nn.Module):
    """轻量级策略网络"""
    def __init__(self, state_dim=224, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, 54)

    def forward(self, x):
        feat = self.backbone(x)
        value = self.value_head(feat).squeeze(-1)
        logits = self.action_head(feat)
        return logits, value


# ─── 经验池 ──────────────────────────────────────────
class SimpleBuffer:
    def __init__(self):
        self.data = []
    
    def add(self, state, act_vec, log_prob, value, reward, done):
        self.data.append((state, act_vec, log_prob, value, reward, done))
    
    def __len__(self):
        return len(self.data)
    
    def clear(self):
        self.data = []


# ─── 快速动作选择 ────────────────────────────────────
def fast_select_action(net, obs, device, temperature=1.0):
    """快速动作选择"""
    actions = obs['legal_actions']
    if not actions:
        return [], torch.tensor(0.0), torch.tensor(0.0)
    
    # 分离 pass 和出牌
    pass_actions = [a for a in actions if len(a) == 0]
    play_actions = [a for a in actions if len(a) > 0]
    
    # 剪枝：只保留前50个动作
    if len(play_actions) > CFG['max_actions']:
        # 简单启发式：优先长牌
        play_actions.sort(key=lambda x: len(x), reverse=True)
        play_actions = play_actions[:CFG['max_actions'] - 5]
        # 加一些随机
        remaining = [a for a in actions if a and a not in play_actions]
        if remaining:
            play_actions.extend(random.sample(remaining, min(5, len(remaining))))
    
    actions = play_actions + pass_actions
    N = len(actions)
    
    # 快速编码
    state = encode_state(obs)
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, value = net(state_t)
    
    # 计算分数
    scores = []
    for act in actions:
        if act:
            act_vec = cards_to_vec(act)
            score = (torch.FloatTensor(act_vec).to(device) * logits).sum()
            scores.append(score.item())
        else:
            scores.append(-100.0)  # pass 低分
    
    # Softmax采样
    scores = torch.FloatTensor(scores)
    probs = F.softmax(scores / temperature, dim=0)
    idx = torch.multinomial(probs, 1).item()
    
    selected = actions[idx]
    log_prob = torch.log(probs[idx] + 1e-8)
    
    return selected, log_prob, value.squeeze()


# ─── 收集经验 ────────────────────────────────────────
def collect_episode_fast(net, env, device, temperature):
    """快速收集一局"""
    obs = env.reset()
    buffer = SimpleBuffer()
    steps = 0
    
    while not env.done and steps < CFG['max_steps']:
        cp = obs['current_player']
        
        # 红队用PPO，蓝队用简单策略
        if cp % 2 == 0:  # 红队
            action, log_prob, value = fast_select_action(net, obs, device, temperature)
            act_vec = cards_to_vec(action) if action else np.zeros(54, dtype=np.float32)
            
            obs, rewards, done, _ = env.step(action)
            
            reward = rewards[cp] if done else 0.01 * len(action) if action else -0.01
            
            buffer.add(
                encode_state(obs),
                act_vec,
                log_prob.item(),
                value.item(),
                reward,
                done
            )
        else:  # 蓝队：简单贪心
            actions = obs['legal_actions']
            if actions:
                # 优先出长牌
                valid = [a for a in actions if a]
                if valid:
                    action = max(valid, key=lambda x: len(x))
                else:
                    action = []
                obs, _, done, _ = env.step(action)
            else:
                obs, _, done, _ = env.step([])
        
        steps += 1
    
    return buffer, rewards if done else [0] * 6


# ─── PPO更新 ────────────────────────────────────────
def ppo_update(net, optimizer, buffers, device):
    """简化PPO更新"""
    if not buffers:
        return 0.0
    
    total_loss = 0.0
    
    for buf in buffers:
        if len(buf) == 0:
            continue
        
        states = torch.FloatTensor(np.array([d[0] for d in buf.data])).to(device)
        rewards = torch.FloatTensor([d[4] for d in buf.data]).to(device)
        dones = torch.FloatTensor([d[5] for d in buf.data]).to(device)
        
        # 计算回报
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + 0.99 * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        
        # 前向
        logits, values = net(states)
        
        # 价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 策略损失（简化）
        advantage = returns - values.detach()
        policy_loss = -(advantage * advantage.sign()).mean()
        
        loss = value_loss + 0.1 * policy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(len(buffers), 1)


# ─── 主训练 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='.')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"✅ 使用设备: {device}")
    print(f"\n🚀 快速训练模式，共 {args.episodes} 局")
    
    # 初始化
    net = SimpleNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG['lr'])
    env = DaguaiEnv()
    
    # 统计
    reward_history = deque(maxlen=100)
    win_history = deque(maxlen=100)
    
    start_time = datetime.now()
    buffers = []
    
    for ep in range(args.episodes):
        # 动态温度
        progress = ep / args.episodes
        temperature = max(0.3, 1.0 * (1 - progress * 0.7))
        
        # 收集经验
        buffer, rewards = collect_episode_fast(net, env, device, temperature)
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
            avg_reward = np.mean(reward_history)
            win_rate = np.mean(win_history)
            eps = (ep + 1) / elapsed
            
            print(f"Ep {ep+1}/{args.episodes} | "
                  f"奖励: {avg_reward:+.3f} | "
                  f"胜率: {win_rate:.1%} | "
                  f"损失: {loss:.3f} | "
                  f"速度: {eps:.2f} eps/s")
        
        # 定期保存
        if (ep + 1) % 1000 == 0:
            save_path = os.path.join(args.save_dir, f'model_ep{ep+1}.pt')
            torch.save({
                'model': net.state_dict(),
                'episode': ep + 1,
                'win_rate': np.mean(win_history),
            }, save_path)
            print(f"💾 已保存: {save_path}")
    
    # 最终保存
    final_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'model': net.state_dict(),
        'episode': args.episodes,
        'win_rate': np.mean(win_history),
    }, final_path)
    print(f"\n✅ 训练完成！最终模型: {final_path}")
    print(f"   最终胜率: {np.mean(win_history):.1%}")


if __name__ == '__main__':
    main()
