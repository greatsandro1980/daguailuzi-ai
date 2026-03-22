#!/usr/bin/env python3
"""微调训练 - 对抗弱化规则Bot"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(60, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 16))
    def forward(self, x): return self.net(x)
    def get_action(self, state, actions):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        scores = [(i, logits[idx].item()) for i, idx in enumerate(valid)]
        return max(scores, key=lambda x: x[1])[0]

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
    
    def get_state(self):
        s = np.zeros(60, dtype=np.float32)
        s[:15] = self.hands[self.current]
        if self.last_play is not None: s[15:30] = self.last_play
        for i in range(6): s[30+i] = self.hands[i].sum()
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
        if card >= 0 and count > 0: action[card] = count
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            self.finish_order.append(self.current)
            if all(self.finished): return True
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
        return False

def rule(game):
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]

def weak_rule(game, error_rate=0.3):
    """弱化规则Bot - 有概率犯错"""
    if np.random.random() < error_rate:
        return np.random.randint(len(game.get_actions()))
    return rule(game)

def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            acts = game.get_actions()
            if game.current % 2 == 0:
                i = model.get_action(game.get_state(), acts)
            else:
                i = rule(game) if opp == 'rule' else np.random.randint(len(acts))
            if game.step(*acts[i]):
                if game.finish_order[0] % 2 == 0: wins += 1
                break
    return wins / n * 100

print("=" * 50)
print("微调训练 - 对抗弱化规则Bot")
print("=" * 50)

model = Net()
model.load_state_dict(torch.load('rl_v9_pretrained.pt', weights_only=True))

# 测试初始状态
print(f"\n初始: vs规则 {test(model, 300, 'rule'):.1f}%, vs随机 {test(model, 300, 'random'):.1f}%")

opt = optim.Adam(model.parameters(), lr=3e-5)
game = Game()
best_rule = 0

print("\n训练中...")
for g in range(20000):
    game.reset()
    trans = []
    
    while True:
        if game.current % 2 == 0:
            s = game.get_state()
            acts = game.get_actions()
            logits = model(torch.FloatTensor(s))
            valid = [15 if c < 0 else c for c, _ in acts]
            probs = torch.softmax(logits, 0)
            vp = torch.tensor([max(probs[i].item(), 1e-8) for i in valid])
            vp = vp / vp.sum()
            ai = torch.multinomial(vp, 1).item()
            trans.append((s, valid[ai]))
            card, cnt = acts[ai]
        else:
            # 使用弱化对手
            acts = game.get_actions()
            i = weak_rule(game, error_rate=0.35)
            card, cnt = acts[i]
        
        if game.step(card, cnt):
            rw = 1.0 if game.finish_order[0] % 2 == 0 else -0.2
            for s, a in trans:
                opt.zero_grad()
                loss = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a] * rw
                loss.backward()
                opt.step()
            break
    
    if (g+1) % 2000 == 0:
        r1 = test(model, 400, 'rule')
        r2 = test(model, 400, 'random')
        if r1 > best_rule:
            best_rule = r1
            torch.save(model.state_dict(), '/workspace/projects/rl_v9_best_rule.pt')
        print(f"{g+1:>5,} | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | 最佳: {best_rule:.1f}%")
        
        json.dump({'episode': g+1, 'vs_rule': r1, 'vs_random': r2, 'best_rule': best_rule,
                  'training_type': 'v9_finetune', 'status': 'training',
                  'timestamp': datetime.now().isoformat()},
                 open('/workspace/projects/training_status.json', 'w'), indent=2)

# 最终测试
print("\n" + "=" * 50)
print("最终测试 (1000局):")
if __import__('os').path.exists('/workspace/projects/rl_v9_best_rule.pt'):
    model.load_state_dict(torch.load('/workspace/projects/rl_v9_best_rule.pt', weights_only=True))

r1 = test(model, 1000, 'rule')
r2 = test(model, 1000, 'random')
print(f"vs规则Bot: {r1:.1f}%")
print(f"vs随机Bot: {r2:.1f}%")

torch.save(model.state_dict(), '/workspace/projects/rl_v9_final.pt')
print("\n" + "=" * 50)
print("✅ vs规则Bot达标" if r1 >= 60 else f"⚠️ vs规则Bot未达标 ({r1:.1f}%)")
print("✅ vs随机Bot达标" if r2 >= 60 else f"⚠️ vs随机Bot未达标 ({r2:.1f}%)")
