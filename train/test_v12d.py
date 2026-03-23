#!/usr/bin/env python3
"""
V12d - 稳健策略版
策略：
1. 混合对手训练（50%规则 + 50%随机）
2. 使用epsilon-greedy策略测试
3. 选择最稳健的出牌
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

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
        hand = self.hands[self.current]
        if card >= 0:
            hand[card] -= cnt
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            if hand.sum() == 0:
                self.finished[self.current] = True
        else:
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
            return True, 0 if team0_done else 1
        
        return False, -1


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
    
    def get_action(self, state, actions):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        best = -float('inf')
        best_idx = 0
        for i, idx in enumerate(valid):
            if logits[idx] > best:
                best = logits[idx]
                best_idx = i
        return best_idx


def rule_action(game):
    """规则策略：优先打小牌"""
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    
    # 优先打对子/三张
    multi = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if multi:
        multi.sort(key=lambda x: x[1])
        return multi[0][0]
    
    # 否则打最小的单张
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def aggressive_action(game):
    """激进策略：优先打大牌"""
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    
    # 优先打大的对子/三张
    multi = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if multi:
        multi.sort(key=lambda x: -x[1])
        return multi[0][0]
    
    # 否则打最小的单张
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
                idx = model.get_action(game.get_state(), actions)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


def test_with_policy(model, n=500):
    """测试使用混合策略"""
    game = Game()
    wins_rule = 0
    wins_random = 0
    
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.get_action(game.get_state(), actions)
            else:
                idx = np.random.randint(len(actions))
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins_random += 1
                break
    
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.get_action(game.get_state(), actions)
            else:
                idx = rule_action(game)
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins_rule += 1
                break
    
    return wins_rule / n * 100, wins_random / n * 100


def main():
    print("=" * 60)
    print("V12d - 稳健策略测试")
    print("=" * 60)
    
    # 加载模型
    models = {
        'V12快速': '/workspace/projects/rl_v12_quick_rule.pt',
        'V12快速随机': '/workspace/projects/rl_v12_quick_random.pt',
        'V12b最佳': '/workspace/projects/rl_v12b_best_random.pt',
    }
    
    best_combined = 0
    best_name = ''
    best_results = {}
    
    for name, path in models.items():
        if os.path.exists(path):
            model = Net()
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            
            r_rule, r_random = test_with_policy(model, 500)
            combined = r_rule + r_random
            
            print(f"\n{name}:")
            print(f"  vs规则Bot: {r_rule:.1f}%")
            print(f"  vs随机Bot: {r_random:.1f}%")
            print(f"  综合得分: {combined:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_name = name
                best_results = {'vs_rule': r_rule, 'vs_random': r_random}
        else:
            print(f"\n{name}: 模型不存在")
    
    print("\n" + "=" * 60)
    print(f"最佳模型: {best_name}")
    print(f"vs规则Bot: {best_results.get('vs_rule', 0):.1f}%")
    print(f"vs随机Bot: {best_results.get('vs_random', 0):.1f}%")
    print(f"综合得分: {best_combined:.1f}%")
    
    if best_results.get('vs_rule', 0) >= 60 and best_results.get('vs_random', 0) >= 70:
        print("\n✅ 全部达标!")
    else:
        print(f"\n⚠️ 部分未达标")


if __name__ == '__main__':
    main()
