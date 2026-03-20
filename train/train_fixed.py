"""
大怪路子训练脚本 - 修复版
正确的PPO实现 + 稳定训练
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from game_env_fast_v2 import FastDaguaiEnvV2, recognize_cards, can_beat_cards

# ─── 配置 ─────────────────────────────────────────────
CFG = {
    'max_steps': 300,
    'batch_size': 64,
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_episodes': 200000,
    'save_interval': 5000,
    'update_interval': 10,  # 每10局更新
}


# ─── 策略网络 ────────────────────────────────────────
class PolicyNet(nn.Module):
    """策略网络 - 输出动作价值"""
    def __init__(self, state_dim=224, hidden_dim=256, max_actions=200):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 动作评分网络
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 54, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action=None):
        """
        state: (B, 224)
        action: (B, 54) 或 None
        返回: action_scores (B, max_actions), value (B,), 或单个action的score
        """
        feat = self.encoder(state)
        value = self.value_head(feat).squeeze(-1)
        
        if action is not None:
            # 评估特定动作
            action_feat = torch.cat([feat, action], dim=-1)
            score = self.action_scorer(action_feat).squeeze(-1)
            return score, value
        
        return feat, value
    
    def get_action_scores(self, state, actions):
        """获取多个动作的分数"""
        feat = self.encoder(state)
        scores = []
        for action in actions:
            action_t = torch.FloatTensor(action).unsqueeze(0)
            action_feat = torch.cat([feat, action_t], dim=-1)
            score = self.action_scorer(action_feat).squeeze(-1)
            scores.append(score)
        return torch.stack(scores), feat


# ─── 状态编码 ────────────────────────────────────────
def encode_state(hand, played, current, hand_sizes, trump_rank, my_team, current_player):
    """状态编码"""
    feat = np.zeros(224, dtype=np.float32)
    feat[0:54] = hand.astype(np.float32)
    feat[54:108] = played.astype(np.float32) / 3.0
    feat[108:162] = current.astype(np.float32)
    feat[162:216] = (hand > 0).astype(np.float32)
    feat[216:222] = hand_sizes.astype(np.float32) / 27.0
    feat[222] = trump_rank / 14.0
    feat[223] = float(my_team)
    return feat


# ─── 对手策略 ────────────────────────────────────────
class RuleBot:
    """规则Bot：小牌优先策略"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return np.zeros(54, dtype=np.int8)
        
        play_actions = [(a, int(np.sum(a))) for a in actions if np.sum(a) > 0]
        if not play_actions:
            return actions[0]
        
        # 小牌优先：选牌数最多且点数最小的
        def get_rank(a):
            for i in range(54):
                if a[i] > 0:
                    return i % 13
            return 99
        
        play_actions.sort(key=lambda x: (-x[1], get_rank(x[0])))
        return play_actions[0][0]


class GreedyBot:
    """贪心Bot：大牌优先"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return np.zeros(54, dtype=np.int8)
        
        play_actions = [(a, int(np.sum(a))) for a in actions if np.sum(a) > 0]
        if not play_actions:
            return actions[0]
        
        # 大牌优先
        def get_rank(a):
            for i in range(53, -1, -1):
                if a[i] > 0:
                    return i % 13
            return 0
        
        play_actions.sort(key=lambda x: (x[1], get_rank(x[0])), reverse=True)
        return play_actions[0][0]


# ─── 收集经验 ────────────────────────────────────────
def collect_episode(net, env, device, temperature=1.0):
    """收集一局经验"""
    obs = env.reset()
    experiences = []
    
    # 蓝队用规则Bot
    blue_bots = {1: RuleBot(), 3: GreedyBot(), 5: RuleBot()}
    
    while not env.done and env.steps < CFG['max_steps']:
        cp = obs['current_player']
        actions = obs['legal_actions']
        
        if not actions:
            break
        
        state = encode_state(
            obs['hand'], obs['played_all'], obs['current_played'],
            obs['hand_sizes'], obs['trump_rank'], obs['my_team'], cp
        )
        
        if cp % 2 == 0:  # 红队：学习策略
            # 获取所有合法动作的分数
            play_actions = [a for a in actions if np.sum(a) > 0]
            if not play_actions:
                # 只能pass
                action = actions[0]
                experiences.append((state, action, 0.0, 0.0, cp))
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    scores, _ = net.get_action_scores(state_t, play_actions)
                    scores = scores.squeeze().cpu().numpy()
                
                # 加启发式分数
                heuristic_scores = []
                for a in play_actions:
                    n = int(np.sum(a))
                    ct, cr = recognize_cards(a, obs['trump_rank'])
                    # 出牌数越多越好，牌型越大越好
                    h = n * 2 + (ct * 3 if ct else 0)
                    heuristic_scores.append(h)
                
                total_scores = scores + np.array(heuristic_scores) * 0.5
                
                # Softmax采样
                probs = F.softmax(torch.FloatTensor(total_scores) / temperature, dim=0).numpy()
                probs = probs / probs.sum()
                idx = np.random.choice(len(play_actions), p=probs)
                
                action = play_actions[idx]
                log_prob = np.log(probs[idx] + 1e-8)
                value = float(scores[idx])
                
                experiences.append((state, action, log_prob, value, cp))
        else:  # 蓝队：规则Bot
            action = blue_bots[cp].select_action(obs)
        
        obs, rewards, done, _ = env.step(action)
    
    # 计算最终奖励
    if env.done:
        team_reward = np.mean([rewards[s] for s in [0, 2, 4]])  # 红队平均奖励
    else:
        team_reward = -0.5  # 未完成惩罚
    
    return experiences, team_reward, rewards if env.done else np.zeros(6)


# ─── PPO更新 ────────────────────────────────────────
def ppo_update(net, optimizer, experiences, device):
    """PPO更新"""
    if len(experiences) < 10:
        return 0.0
    
    # 解包
    states, actions, old_log_probs, old_values, players = zip(*experiences)
    
    n = len(states)
    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.FloatTensor(np.array(actions)).to(device)
    old_log_probs_t = torch.FloatTensor(np.array(old_log_probs)).to(device)
    
    # 计算回报（简化版：用最终奖励）
    # 这里我们用MC回报
    returns = []
    for i, (s, a, lp, v, p) in enumerate(experiences):
        # 给每个步骤分配奖励
        r = experiences[-1][1] if i == len(experiences)-1 else 0  # 只在最后给奖励
        returns.append(r)
    
    returns_t = torch.FloatTensor(returns).to(device)
    
    # 前向传播
    total_loss = 0.0
    
    for _ in range(3):  # PPO epochs
        new_scores = []
        new_values = []
        
        for i in range(n):
            state_t = states_t[i:i+1]
            action_t = actions_t[i:i+1]
            score, value = net(state_t, action_t)
            new_scores.append(score)
            new_values.append(value)
        
        new_scores_t = torch.stack(new_scores).squeeze()
        new_values_t = torch.stack(new_values).squeeze()
        
        # 计算新的log_prob
        # 这里简化：用score作为log_prob
        new_log_probs_t = new_scores_t
        
        # 概率比
        ratio = torch.exp(new_log_probs_t - old_log_probs_t)
        
        # 优势（简化）
        advantages = returns_t - new_values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CFG['clip_eps'], 1 + CFG['clip_eps']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = F.mse_loss(new_values_t, returns_t)
        
        # 熵奖励
        entropy = -(new_scores_t * torch.exp(new_scores_t.clamp(-10, 10))).mean()
        
        loss = policy_loss + CFG['value_coef'] * value_loss - CFG['entropy_coef'] * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / 3


# ─── 主训练 ──────────────────────────────────────────
def main():
    device = torch.device('cpu')
    
    net = PolicyNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG['lr'])
    env = FastDaguaiEnvV2()
    
    # 状态
    episode = 0
    win_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    all_experiences = []
    start_time = time.time()
    
    # 加载检查点
    save_dir = '/workspace/projects'
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('model_v2_ep') and f.endswith('.pt')]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split('ep')[1].split('.')[0]))
        ckpt = torch.load(os.path.join(save_dir, latest), map_location='cpu', weights_only=False)
        net.load_state_dict(ckpt['model'])
        episode = ckpt.get('episode', 0)
        print(f"加载检查点: {latest}, 从第{episode}局继续")
    
    print(f"🚀 开始训练 (修复版)")
    print(f"   目标: {CFG['max_episodes']}局")
    print(f"   学习率: {CFG['lr']}")
    
    temperature = 1.0
    
    for ep in range(episode, CFG['max_episodes']):
        # 温度衰减
        temperature = max(0.3, temperature * 0.9999)
        
        # 收集经验
        exp, team_reward, final_rewards = collect_episode(net, env, device, temperature)
        all_experiences.extend(exp)
        
        # 记录
        red_win = team_reward > 0
        win_history.append(1.0 if red_win else 0.0)
        reward_history.append(team_reward)
        
        # 定期更新
        if (ep + 1) % CFG['update_interval'] == 0:
            loss = ppo_update(net, optimizer, all_experiences, device)
            all_experiences = []
        
        # 保存状态
        if (ep + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (ep + 1 - episode) / elapsed if elapsed > 0 else 0
            win_rate = np.mean(win_history) * 100
            
            status = {
                'episode': ep + 1,
                'win_rate': round(win_rate, 1),
                'avg_reward': round(np.mean(reward_history), 3),
                'speed': round(speed, 1),
                'elapsed_hours': round(elapsed / 3600, 2),
                'lr': CFG['lr'],
                'temperature': round(temperature, 3),
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(os.path.join(save_dir, 'training_status.json'), 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"Ep {ep+1}/{CFG['max_episodes']} | 胜率: {win_rate:.1f}% | 奖励: {np.mean(reward_history):.3f} | 速度: {speed:.1f}/s")
        
        # 保存检查点
        if (ep + 1) % CFG['save_interval'] == 0:
            save_path = os.path.join(save_dir, f'model_v2_ep{ep+1}.pt')
            torch.save({
                'model': net.state_dict(),
                'episode': ep + 1,
                'win_rate': np.mean(win_history),
            }, save_path)
            print(f"✅ 保存: {save_path}")
    
    # 最终保存
    torch.save({'model': net.state_dict(), 'episode': CFG['max_episodes']}, 
               os.path.join(save_dir, 'model_v2_final.pt'))
    print("训练完成!")


if __name__ == '__main__':
    main()
