#!/usr/bin/env python3
"""
V12c - 最终优化版
策略：
1. 从最佳模型开始
2. 100%随机Bot训练
3. 保守学习率
4. 目标：vs随机Bot ≥ 70%
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
from datetime import datetime
from collections import deque

import functools
print = functools.partial(print, flush=True)

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
            reward += 0.1  # 出牌奖励
            reward += cnt * 0.1  # 手牌减少奖励
            
            if cnt >= 2:
                reward += 0.1  # 关键牌型奖励
            
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 16)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, actions, epsilon=0.0):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
        best = -float('inf')
        best_idx = 0
        for i, idx in enumerate(valid):
            if logits[idx] > best:
                best = logits[idx]
                best_idx = i
        return best_idx


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


def train_v12c(model, n_games=8000):
    game = Game()
    opt = optim.Adam(model.parameters(), lr=2e-5)  # 非常保守的学习率
    best_rule = 0
    best_random = 0
    best_combined = 0
    best_model_state = None
    
    reward_history = deque(maxlen=1000)
    
    print(f"\nV12c训练 ({n_games}局)...")
    print("策略: 100%随机Bot对手，保守学习率")
    print("=" * 60)
    
    for g in range(n_games):
        game.reset()
        episode_rewards = []
        transitions = []
        
        while True:
            if game.current % 2 == 0:
                state = game.get_state()
                actions = game.get_actions()
                
                # 低探索率
                if np.random.random() < 0.03:
                    a_idx = np.random.randint(len(actions))
                else:
                    with torch.no_grad():
                        logits = model(torch.FloatTensor(state))
                    valid = [15 if c < 0 else c for c, _ in actions]
                    probs = torch.softmax(logits, 0)
                    valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
                    valid_probs = valid_probs / valid_probs.sum()
                    a_idx = torch.multinomial(valid_probs, 1).item()
                
                transitions.append((state, 15 if actions[a_idx][0] < 0 else actions[a_idx][0]))
                card, cnt = actions[a_idx]
            else:
                # 100%随机Bot对手
                actions = game.get_actions()
                a_idx = np.random.randint(len(actions))
                card, cnt = actions[a_idx]
            
            done, winner, instant_reward = game.step(card, cnt)
            episode_rewards.append(instant_reward)
            
            if done:
                total_reward = sum(episode_rewards)
                reward_history.append(total_reward)
                
                baseline = sum(reward_history) / len(reward_history) if reward_history else 0
                advantage = total_reward - baseline
                
                for s, a in transitions:
                    log_prob = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a]
                    loss = log_prob * advantage / 10
                    loss.backward()
                break
        
        if (g + 1) % 10 == 0:
            opt.step()
            opt.zero_grad()
        
        if (g + 1) % 1000 == 0:
            r1 = test(model, 400, 'rule')
            r2 = test(model, 400, 'random')
            combined = r1 + r2
            
            print(f"  {g+1:>5,}局 | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | 综合: {combined:.1f}%")
            
            # 只有综合得分提高时才保存
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), '/workspace/projects/rl_v12c_best.pt')
                print(f"    ★ 新最佳模型! vs规则: {r1:.1f}%, vs随机: {r2:.1f}%")
            
            if r1 >= 60 and r2 >= 70:
                print(f"\n✅ 目标达成！vs规则: {r1:.1f}%, vs随机: {r2:.1f}%")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V12c - 最终优化版")
    print("=" * 60)
    
    model = Net()
    
    # 加载最佳模型
    if os.path.exists('/workspace/projects/rl_v12_quick_rule.pt'):
        print("加载V12快速最佳模型...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v12_quick_rule.pt', weights_only=True))
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 训练
    model, best_rule, best_random = train_v12c(model, n_games=8000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试:")
    
    if os.path.exists('/workspace/projects/rl_v12c_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v12c_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v12c_final.pt')
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%")
    
    if r1 >= 60 and r2 >= 70:
        print("\n✅ 全部达标!")
    else:
        print(f"\n⚠️ vs规则 {'达标' if r1 >= 60 else '未达标'}, vs随机 {'达标' if r2 >= 70 else '未达标'}")


if __name__ == '__main__':
    main()
