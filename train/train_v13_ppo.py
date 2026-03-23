#!/usr/bin/env python3
"""
V13 (P3优化) - PPO算法 + 扩展模型容量
策略：
1. PPO算法：更稳定的策略梯度，使用裁剪目标函数
2. Actor-Critic架构：同时学习策略和价值函数
3. 扩展模型容量：512神经元 × 4层
4. 混合对手训练：平衡vs规则和vs随机性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from collections import deque

import functools
print = functools.partial(print, flush=True)

# ============== 游戏环境 ==============
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        
    def reset(self):
        deck = [i for i in range(13) for _ in range(4)] + [13, 14]
        np.random.shuffle(deck)
        self.hands = np.zeros((6, 15), dtype=np.int32)
        idx = 0
        for p in range(6):
            count = 9 if p < 4 else 8
            for _ in range(count):
                self.hands[p, deck[idx]] += 1
                idx += 1
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        return self.get_state()
    
    def get_state(self):
        s = np.zeros(60, dtype=np.float32)
        s[:15] = self.hands[self.current]
        if self.last_play is not None:
            s[15:30] = self.last_play
        for i in range(6):
            s[30+i] = self.hands[i].sum()
        if self.last_play is not None and self.last_play.sum() > 0:
            cnt = int(self.last_play[self.last_play > 0][0])
            s[36 + min(cnt-1, 2)] = 1.0
        s[39] = 1.0 if self.last_play is None else 0.0
        hand = self.hands[self.current]
        s[46] = sum(1 for i in range(13) if hand[i] == 1)
        s[47] = sum(1 for i in range(13) if hand[i] == 2)
        s[48] = sum(1 for i in range(13) if hand[i] >= 3)
        s[49] = hand[13]
        s[50] = hand[14]
        s[51] = hand.sum()
        return s
    
    def get_actions(self):
        hand = self.hands[self.current]
        actions = []
        if self.last_play is None or self.last_player == self.current:
            for i in range(15):
                if hand[i] >= 1: actions.append((i, 1))
            for i in range(13):
                if hand[i] >= 2: actions.append((i, 2))
            for i in range(13):
                if hand[i] >= 3: actions.append((i, 3))
        else:
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt: actions.append((i, last_cnt))
            actions.append((-1, 0))
        return actions if actions else [(-1, 0)]
    
    def step(self, card, cnt):
        reward = 0.0
        hand = self.hands[self.current]
        
        if card >= 0:
            hand[card] -= cnt
            reward += 0.1 + cnt * 0.1
            if cnt >= 2:
                reward += 0.1
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            reward -= 0.05
            self.passes += 1
        
        for _ in range(6):
            self.current = (self.current + 1) % 6
            if not self.finished[self.current]:
                break
        
        if self.passes >= sum(1 for p in range(6) if not self.finished[p]):
            self.last_play = None
            self.last_player = -1
            self.passes = 0
        
        team0_done = all(self.finished[p] for p in [0, 2, 4])
        team1_done = all(self.finished[p] for p in [1, 3, 5])
        
        if team0_done or team1_done:
            winner = 0 if team0_done else 1
            if winner == 0:
                reward += 2.0
            return True, winner, reward
        
        return False, -1, reward


# ============== PPO Actor-Critic模型 ==============
class ActorCritic(nn.Module):
    """扩展容量的Actor-Critic网络"""
    def __init__(self, state_dim=60, action_dim=16, hidden_dim=512):
        super().__init__()
        
        # 共享特征提取层（更深更宽）
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor头（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Critic头（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, state, actions, deterministic=False):
        with torch.no_grad():
            logits, value = self(torch.FloatTensor(state))
        
        valid = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits, 0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        valid_probs = valid_probs / valid_probs.sum()
        
        if deterministic:
            a_idx = valid_probs.argmax().item()
        else:
            a_idx = torch.multinomial(valid_probs, 1).item()
        
        return a_idx, valid[a_idx], value.item()
    
    def evaluate_actions(self, states, actions):
        logits, values = self(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        dist_entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return action_log_probs, values.squeeze(), dist_entropy


# ============== 策略 ==============
def rule_action(game):
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx, _, _ = model.get_action(game.get_state(), actions, deterministic=True)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner, _ = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== PPO训练 ==============
def ppo_train(model, n_games=10000, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
              value_coef=0.5, entropy_coef=0.01):
    """
    PPO训练算法
    
    参数：
    - clip_ratio: PPO裁剪参数
    - gamma: 折扣因子
    - value_coef: 价值损失系数
    - entropy_coef: 熵正则化系数
    """
    game = Game()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_rule = 0
    best_random = 0
    best_combined = 0
    
    # 经验缓冲区
    memory = {
        'states': [], 'actions': [], 'rewards': [], 
        'values': [], 'log_probs': [], 'dones': []
    }
    
    print(f"\nPPO训练 ({n_games}局)...")
    print("=" * 60)
    print(f"参数: lr={lr}, gamma={gamma}, clip={clip_ratio}")
    print("=" * 60)
    
    for g in range(n_games):
        game.reset()
        episode_data = []
        
        while True:
            state = game.get_state()
            actions = game.get_actions()
            
            if game.current % 2 == 0:
                # 模型决策
                idx, action, value = model.get_action(state, actions, deterministic=False)
                
                with torch.no_grad():
                    logits, _ = model(torch.FloatTensor(state))
                log_prob = torch.log_softmax(logits, 0)[action].item()
                
                episode_data.append({
                    'state': state,
                    'action': action,
                    'value': value,
                    'log_prob': log_prob
                })
                
                card, cnt = actions[idx]
            else:
                # 对手决策（混合策略）
                actions_opp = game.get_actions()
                if np.random.random() < 0.6:
                    idx = np.random.randint(len(actions_opp))  # 随机
                else:
                    idx = rule_action(game)  # 规则
                card, cnt = actions_opp[idx]
            
            done, winner, reward = game.step(card, cnt)
            
            if game.current % 2 == 0 or done:
                episode_data[-1]['reward'] = reward if len(episode_data) > 0 else 0
                episode_data[-1]['done'] = done
            
            if done:
                # 计算回报（从后向前）
                returns = []
                R = 0
                for data in reversed(episode_data):
                    if 'reward' in data:
                        R = data['reward'] + gamma * R
                    returns.insert(0, R)
                
                # 存储经验
                for i, data in enumerate(episode_data):
                    memory['states'].append(data['state'])
                    memory['actions'].append(data['action'])
                    memory['rewards'].append(returns[i] if i < len(returns) else 0)
                    memory['values'].append(data['value'])
                    memory['log_probs'].append(data['log_prob'])
                    memory['dones'].append(data.get('done', False))
                break
        
        # PPO更新（每收集足够经验后）
        if len(memory['states']) >= 2048:
            states = torch.FloatTensor(np.array(memory['states']))
            actions = torch.LongTensor(memory['actions'])
            old_log_probs = torch.FloatTensor(memory['log_probs'])
            returns = torch.FloatTensor(memory['rewards'])
            old_values = torch.FloatTensor(memory['values'])
            
            # 计算优势
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 多轮更新
            for _ in range(4):
                new_log_probs, new_values, entropy = model.evaluate_actions(states, actions)
                
                # 策略损失（PPO裁剪）
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(new_values, returns)
                
                # 总损失
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            # 清空缓冲区
            memory = {k: [] for k in memory}
        
        # 测试和保存
        if (g + 1) % 1000 == 0:
            r1 = test(model, 400, 'rule')
            r2 = test(model, 400, 'random')
            combined = r1 + r2
            
            print(f"  {g+1:>5,}局 | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | 综合: {combined:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v13_ppo_best.pt')
                print(f"    ★ 新最佳模型!")
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V13 (P3) - PPO算法 + 扩展模型容量")
    print("=" * 60)
    
    model = ActorCritic(hidden_dim=512)
    
    # 尝试加载V12最佳模型作为初始化
    if os.path.exists('/workspace/projects/rl_v12_quick_random.pt'):
        print("加载V12最佳模型作为初始化...")
        old_model = torch.load('/workspace/projects/rl_v12_quick_random.pt', weights_only=True)
        # 部分迁移（只加载兼容的参数）
        model_state = model.state_dict()
        for name, param in old_model.items():
            if name in model_state and model_state[name].shape == param.shape:
                model_state[name] = param
        model.load_state_dict(model_state, strict=False)
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # PPO训练
    model, best_rule, best_random = ppo_train(model, n_games=10000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试:")
    
    if os.path.exists('/workspace/projects/rl_v13_ppo_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v13_ppo_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v13_ppo_final.pt')
    
    print("\n" + "=" * 60)
    print("P3优化完成!")
    print(f"最佳: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%")
    
    if r1 >= 60 and r2 >= 70:
        print("\n✅ 全部达标!")
    else:
        status = []
        if r1 < 60: status.append(f"vs规则 {r1:.1f}%")
        if r2 < 70: status.append(f"vs随机 {r2:.1f}%")
        print(f"\n⚠️ 未达标: {', '.join(status)}")


if __name__ == '__main__':
    main()
