#!/usr/bin/env python3
"""
V16 - DQN算法优化版
使用Deep Q-Network（off-policy）更适合微调
目标：vs规则 ≥90%, vs随机 ≥90%
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import functools
import random
from collections import deque
print = functools.partial(print, flush=True)

# ============== 增强游戏环境（更多状态特征） ==============
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.trick_count = 0
        self.team_cards = np.zeros((2, 15), dtype=np.int32)  # 队伍总牌数
        
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
        self.trick_count = 0
        self._update_team_cards()
        return self.get_state()
    
    def _update_team_cards(self):
        self.team_cards = np.zeros((2, 15), dtype=np.int32)
        for p in range(6):
            team = p % 2
            self.team_cards[team] += self.hands[p]
    
    def get_state(self):
        # 扩展状态到80维
        s = np.zeros(80, dtype=np.float32)
        
        # 0-14: 当前玩家手牌
        s[:15] = self.hands[self.current]
        
        # 15-29: 上家出的牌
        if self.last_play is not None:
            s[15:30] = self.last_play
        
        # 30-35: 各玩家手牌数
        for i in range(6):
            s[30+i] = self.hands[i].sum() / 27.0
        
        # 36-38: 牌型特征
        if self.last_play is not None and self.last_play.sum() > 0:
            cnt = int(self.last_play[self.last_play > 0][0])
            s[36 + min(cnt-1, 2)] = 1.0
        
        # 39: 是否首发
        s[39] = 1.0 if self.last_play is None else 0.0
        
        # 40-42: 队友状态
        team = self.current % 2
        for i, p in enumerate([p for p in range(6) if p % 2 == team and p != self.current]):
            s[40+i] = 1.0 if self.finished[p] else 0.0
        
        # 43-45: 对手状态
        for i, p in enumerate([p for p in range(6) if p % 2 != team]):
            s[43+i] = 1.0 if self.finished[p] else 0.0
        
        hand = self.hands[self.current]
        
        # 46-48: 手牌统计
        s[46] = sum(1 for i in range(13) if hand[i] == 1)  # 单张数
        s[47] = sum(1 for i in range(13) if hand[i] == 2)  # 对子数
        s[48] = sum(1 for i in range(13) if hand[i] >= 3)  # 三张数
        
        # 49-50: 王牌数
        s[49] = hand[13]  # 小王
        s[50] = hand[14]  # 大王
        
        # 51-52: 总牌数和轮次
        s[51] = hand.sum() / 27.0
        s[52] = self.trick_count / 15.0
        
        # 53-67: 队伍总牌数分布
        s[53:68] = self.team_cards[team] / 50.0
        
        # 68-79: 额外特征
        # 68-69: 队伍剩余总牌数
        s[68] = sum(self.hands[p].sum() for p in range(6) if p % 2 == team) / 81.0
        s[69] = sum(self.hands[p].sum() for p in range(6) if p % 2 != team) / 81.0
        
        # 70-72: 手牌价值分布
        s[70] = sum(hand[i] * (i + 2) for i in range(13)) / 200.0  # 普通牌价值
        s[71] = hand[13] * 15 / 200.0  # 小王价值
        s[72] = hand[14] * 16 / 200.0  # 大王价值
        
        # 73-79: 最近出牌历史（简化）
        if self.last_play is not None:
            s[73] = self.last_play.sum()
            s[74] = 1.0 if self.last_play[13] > 0 else 0.0  # 小王
            s[75] = 1.0 if self.last_play[14] > 0 else 0.0  # 大王
        
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
        team = self.current % 2
        
        if card >= 0:
            hand[card] -= cnt
            reward += 0.1 + cnt * 0.1
            
            # 策略奖励
            if self.last_play is None or self.last_player == self.current:
                if card <= 5:
                    reward += 0.05
                elif card >= 13:
                    reward -= 0.1
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            # pass奖励
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None else 0
                if last_val >= 10:
                    reward += 0.05
            self.passes += 1
        
        self._update_team_cards()
        
        for _ in range(6):
            self.current = (self.current + 1) % 6
            if not self.finished[self.current]:
                break
        
        if self.passes >= sum(1 for p in range(6) if not self.finished[p]):
            self.last_play = None
            self.last_player = -1
            self.passes = 0
            self.trick_count += 1
        
        team0_done = all(self.finished[p] for p in [0, 2, 4])
        team1_done = all(self.finished[p] for p in [1, 3, 5])
        
        if team0_done or team1_done:
            winner = 0 if team0_done else 1
            return True, winner, reward + (2.0 if winner == 0 else 0)
        
        return False, -1, reward


# ============== DQN网络 ==============
class DQNetwork(nn.Module):
    def __init__(self, state_dim=80, action_dim=16, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, actions, epsilon=0.0):
        """epsilon-greedy策略"""
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
        
        with torch.no_grad():
            q_values = self(torch.FloatTensor(state))
        
        # 只考虑有效动作
        valid = [15 if c < 0 else c for c, _ in actions]
        valid_q = torch.tensor([q_values[idx].item() for idx in valid])
        return valid_q.argmax().item()


# ============== 经验回放 ==============
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, valid_actions):
        self.buffer.append((state, action, reward, next_state, done, valid_actions))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, valid_actions = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            valid_actions
        )
    
    def __len__(self):
        return len(self.buffer)


# ============== 对手策略 ==============
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
                idx = model.get_action(game.get_state(), actions, epsilon=0.0)
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


# ============== DQN训练 ==============
def train_dqn(model, n_games=8000):
    game = Game()
    
    # 目标网络
    target_model = DQNetwork(state_dim=80, action_dim=16, hidden_dim=512)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer()
    
    best_rule = 0
    best_random = 0
    best_combined = 0
    
    # epsilon策略
    epsilon_start = 0.3
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    epsilon = epsilon_start
    
    # 训练参数
    batch_size = 64
    gamma = 0.99
    target_update = 500
    
    print(f"\nDQN训练 ({n_games}局)...")
    print("=" * 60)
    print("使用Deep Q-Network (off-policy)")
    print("=" * 60)
    
    total_steps = 0
    
    for g in range(n_games):
        game.reset()
        
        while True:
            state = game.get_state()
            actions = game.get_actions()
            
            if game.current % 2 == 0:
                # 我方使用DQN
                idx = model.get_action(state, actions, epsilon)
                card, cnt = actions[idx]
                action_key = 15 if card < 0 else card  # pass映射为15
            else:
                # 对方：70%随机 + 30%规则
                if np.random.random() < 0.7:
                    idx = np.random.randint(len(actions))
                else:
                    idx = rule_action(game)
                card, cnt = actions[idx]
                action_key = 15 if card < 0 else card
            
            done, winner, reward = game.step(card, cnt)
            next_state = game.get_state()
            next_actions = game.get_actions()
            
            # 存储经验
            if game.current % 2 == 0 or (not done and np.random.random() < 0.3):
                replay_buffer.push(
                    state, action_key, reward if game.current % 2 == 0 else -reward,
                    next_state, done, next_actions
                )
            
            total_steps += 1
            
            # 训练
            if len(replay_buffer) >= batch_size and total_steps % 4 == 0:
                states, actions_b, rewards, next_states, dones, valid_actions = replay_buffer.sample(batch_size)
                
                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions_b)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones)
                
                # 当前Q值
                current_q = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                
                # 目标Q值
                with torch.no_grad():
                    next_q = target_model(next_states_t)
                    max_next_q = next_q.max(1)[0]
                    target_q = rewards_t + gamma * max_next_q * (1 - dones_t)
                
                loss = nn.MSELoss()(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 更新目标网络
            if total_steps % target_update == 0:
                target_model.load_state_dict(model.state_dict())
            
            if done:
                break
        
        # 衰减epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 定期测试
        if (g + 1) % 500 == 0:
            r1 = test(model, 500, 'rule')
            r2 = test(model, 500, 'random')
            combined = r1 + r2
            
            print(f"  {g+1:>5,}局 (ε={epsilon:.3f}) | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | 综合: {combined:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v16_best.pt')
                print(f"    ★ 新最佳!")
            
            if r1 >= 90 and r2 >= 90:
                print(f"    🎯 目标达成!")
                break
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V16 - DQN算法优化版")
    print("目标：vs规则 ≥90%, vs随机 ≥90%")
    print("=" * 60)
    
    model = DQNetwork(state_dim=80, action_dim=16, hidden_dim=512)
    
    # 尝试加载预训练权重
    if os.path.exists('/workspace/projects/rl_v14b_best.pt'):
        print("尝试从V14b迁移学习...")
        try:
            old_state = torch.load('/workspace/projects/rl_v14b_best.pt', weights_only=True)
            # 映射部分权重
            new_state = model.state_dict()
            for key in old_state:
                if key in new_state and old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
            model.load_state_dict(new_state, strict=False)
            print("部分权重迁移成功!")
        except Exception as e:
            print(f"迁移失败: {e}")
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 训练
    model, best_rule, best_random = train_dqn(model, n_games=8000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试 (1000局):")
    
    if os.path.exists('/workspace/projects/rl_v16_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v16_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  综合得分: {r1+r2:.1f}%")
    
    if r1 >= 90 and r2 >= 90:
        print("\n🎯 目标达成!")


if __name__ == '__main__':
    main()
