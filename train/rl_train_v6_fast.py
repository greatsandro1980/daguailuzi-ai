#!/usr/bin/env python3
"""训练 V6 快速版 - 监督学习 + 自我博弈"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from datetime import datetime
from collections import deque

class SimpleGame:
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
    
    def step(self, card, count):
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            self.finish_order.append(self.current)
            if all(self.finished):
                return True, self.finish_order[0] % 2
        if action.sum() > 0:
            self.last_play = action.copy()
            self.last_player = self.current
            self.passes = 0
        else:
            self.passes += 1
        if self.passes >= 5:
            self.last_play = None
            self.last_player = -1
            self.passes = 0
        self.current = (self.current + 1) % 6
        while self.finished[self.current]:
            self.current = (self.current + 1) % 6
        return False, -1

class PolicyNet(nn.Module):
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
        valid_idx = [15 if c < 0 else c for c, _ in actions]
        scores = [(i, logits[idx].item()) for i, idx in enumerate(valid_idx)]
        return max(scores, key=lambda x: x[1])[0]

def rule_action(game):
    actions = game.get_actions()
    hand = game.hands[game.current]
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]

def strong_rule_action(game):
    actions = game.get_actions()
    team = game.current % 2
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    if mates_done >= 2:
        valid.sort(key=lambda x: -x[1])
    else:
        pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
        if pairs:
            pairs.sort(key=lambda x: x[1])
            return pairs[0][0]
        valid.sort(key=lambda x: x[1])
    return valid[0][0]

def generate_data(n=5000):
    game = SimpleGame()
    states, labels = [], []
    for _ in range(n):
        game.reset()
        while True:
            state = game.get_state()
            actions = game.get_actions()
            idx = rule_action(game)
            states.append(state)
            labels.append(15 if actions[idx][0] < 0 else actions[idx][0])
            done, _ = game.step(*actions[idx])
            if done: break
    return np.array(states), np.array(labels)

def train_supervised(model, states, labels, epochs=10):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        idx = np.random.permutation(len(states))
        loss_sum = 0
        for i in range(0, len(states), 256):
            b = idx[i:i+256]
            opt.zero_grad()
            loss = crit(model(torch.FloatTensor(states[b])), torch.LongTensor(labels[b]))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {ep+1}/{epochs}, Loss: {loss_sum/(len(states)//256):.4f}")

def test(model, n=100, opp='rule'):
    game = SimpleGame()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.get_action(game.get_state(), actions)
            else:
                idx = rule_action(game) if opp == 'rule' else np.random.randint(len(actions)) if opp == 'random' else strong_rule_action(game)
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100

def self_play(model, n=30000):
    game = SimpleGame()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    wins_hist = deque(maxlen=100)
    best = 0
    
    print("\n自我博弈训练...")
    for g in range(n):
        game.reset()
        trans = []
        while True:
            if game.current % 2 == 0:
                state = game.get_state()
                actions = game.get_actions()
                logits = model(torch.FloatTensor(state))
                valid = [15 if c < 0 else c for c, _ in actions]
                probs = torch.softmax(logits[valid], dim=0)
                a_idx = torch.multinomial(probs, 1).item()
                trans.append((state, valid[a_idx]))
                card, cnt = actions[a_idx]
            else:
                actions = game.get_actions()
                idx = rule_action(game)
                card, cnt = actions[idx]
            done, winner = game.step(card, cnt)
            if done:
                rw = 1.0 if winner == 0 else -0.3
                for s, a in trans:
                    opt.zero_grad()
                    loss = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a] * rw
                    loss.backward()
                    opt.step()
                wins_hist.append(1 if winner == 0 else 0)
                break
        
        if (g+1) % 2000 == 0:
            r1, r2 = test(model, 100, 'rule'), test(model, 100, 'random')
            if r1 > best:
                best = r1
                torch.save(model.state_dict(), '/workspace/projects/rl_v6_best.pt')
            print(f"{g+1:>5,} | vs规则: {r1:>5.1f}% | vs随机: {r2:>5.1f}% | 最佳: {best:>5.1f}%")
            status = {'episode': g+1, 'vs_rule': r1, 'vs_random': r2, 'best_rate': best, 
                     'training_type': 'v6', 'status': 'training', 'timestamp': datetime.now().isoformat()}
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    return model

def main():
    print("=" * 50)
    print("V6 快速训练版")
    print("=" * 50)
    
    model = PolicyNet()
    
    print("\n[阶段1] 监督学习预训练...")
    states, labels = generate_data(5000)
    print(f"生成 {len(states)} 条训练数据")
    train_supervised(model, states, labels, epochs=10)
    
    print(f"\n预训练后: vs规则 {test(model, 100, 'rule'):.1f}%, vs随机 {test(model, 100, 'random'):.1f}%")
    torch.save(model.state_dict(), '/workspace/projects/rl_v6_pretrained.pt')
    
    print("\n[阶段2] 自我博弈...")
    model = self_play(model, 30000)
    
    print("\n最终测试:")
    r1, r2 = test(model, 200, 'rule'), test(model, 200, 'random')
    print(f"vs规则Bot: {r1:.1f}%, vs随机Bot: {r2:.1f}%")
    torch.save(model.state_dict(), '/workspace/projects/rl_v6_final.pt')
    print("完成!")

if __name__ == '__main__':
    main()
