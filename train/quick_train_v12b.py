#!/usr/bin/env python3
"""
V12b - 专门针对vs随机Bot优化
策略：
1. 更高的随机Bot训练比例
2. 奖励函数更偏向快速出完手牌
3. 更长的训练时间
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

# ============== 游戏环境 ==============
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        
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
        self.finish_order = []
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
        team = self.current % 2
        for i, p in enumerate([p for p in range(6) if p % 2 == team and p != self.current]):
            s[40+i] = 1.0 if self.finished[p] else 0.0
        for i, p in enumerate([p for p in range(6) if p % 2 != team]):
            s[43+i] = 1.0 if self.finished[p] else 0.0
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
            # 更激进的奖励：鼓励快速出牌
            reward += 0.05  # 出牌奖励
            reward += cnt * 0.05  # 手牌减少奖励
            
            # 关键牌型奖励
            if cnt == 2:
                reward += 0.03
            elif cnt == 3:
                reward += 0.08
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                self.finish_order.append(self.current)
                reward += 0.5  # 先出完奖励更高
        else:
            reward -= 0.02  # 过牌惩罚更重
            self.passes += 1
        
        # 下一个玩家
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
                reward += 1.5  # 胜利奖励更高
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


def test(model, n=300, opp='rule'):
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


def train_v12b(model, n_games=6000):
    game = Game()
    opt = optim.Adam(model.parameters(), lr=3e-5)  # 降低学习率
    best_rule = 0
    best_random = 0
    no_improve_count = 0  # 早停计数器
    best_model_state = None
    
    reward_history = deque(maxlen=500)
    
    print(f"\nV12b训练 ({n_games}局)...")
    print("=" * 60)
    
    for g in range(n_games):
        game.reset()
        episode_rewards = []
        transitions = []
        
        while True:
            if game.current % 2 == 0:
                state = game.get_state()
                actions = game.get_actions()
                
                # 降低探索率
                if np.random.random() < 0.05:
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
                actions = game.get_actions()
                # 70%随机Bot + 30%规则Bot
                if np.random.random() < 0.7:
                    a_idx = np.random.randint(len(actions))
                else:
                    a_idx = rule_action(game)
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
                    loss = log_prob * advantage / 5
                    loss.backward()
                break
        
        if (g + 1) % 5 == 0:
            opt.step()
            opt.zero_grad()
        
        if (g + 1) % 500 == 0:
            print(f"  {g+1:>4,}/{n_games}局完成", end='\r')
            sys.stdout.flush()
        
        if (g + 1) % 1000 == 0:
            r1 = test(model, 300, 'rule')
            r2 = test(model, 300, 'random')
            
            improved = False
            if r1 > best_rule:
                best_rule = r1
                torch.save(model.state_dict(), '/workspace/projects/rl_v12b_best_rule.pt')
                improved = True
            if r2 > best_random:
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v12b_best_random.pt')
                best_model_state = model.state_dict().copy()
                improved = True
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            print(f"\n  {g+1:>4,}局 | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | 最佳: {best_rule:.1f}%/{best_random:.1f}%")
            
            # 早停：连续3次无改善则停止
            if no_improve_count >= 3:
                print(f"\n早停：连续{no_improve_count}次无改善，停止训练")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break
            
            status = {
                'episode': g+1,
                'vs_rule': r1,
                'vs_random': r2,
                'best_rule': best_rule,
                'best_random': best_random,
                'training_type': 'v12b_optimized',
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V12b - 针对vs随机Bot优化")
    print("=" * 60)
    
    model = Net()
    
    if os.path.exists('/workspace/projects/rl_v12b_best_random.pt'):
        print("加载V12b最佳模型...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v12b_best_random.pt', weights_only=True))
    elif os.path.exists('/workspace/projects/rl_v9_best_rule.pt'):
        print("加载V9最佳模型...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v9_best_rule.pt', weights_only=True))
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 训练
    model, best_rule, best_random = train_v12b(model, n_games=6000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试:")
    
    if os.path.exists('/workspace/projects/rl_v12b_best_random.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v12b_best_random.pt', weights_only=True))
    
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v12b_final.pt')
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%")
    
    if r1 >= 60 and r2 >= 70:
        print("\n✅ 全部达标!")
    elif r1 >= 60:
        print(f"\n⚠️ vs规则达标，vs随机 {r2:.1f}% 未达70%目标")


if __name__ == '__main__':
    main()
