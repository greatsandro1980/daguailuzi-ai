#!/usr/bin/env python3
"""
训练 V9 快速版 - 高效训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime

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
            s[36 + min(int(self.last_play[self.last_play > 0][0])-1, 2)] = 1.0
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
            if all(self.finished): return True, self.finish_order[0] % 2
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 16)
        )
    def forward(self, x): return self.net(x)
    def get_action(self, state, actions):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        scores = [(i, logits[idx].item()) for i, idx in enumerate(valid)]
        return max(scores, key=lambda x: x[1])[0]

def smart(game):
    actions = game.get_actions()
    hand = game.hands[game.current]
    team = game.current % 2
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    opps = [game.hands[p].sum() for p in range(6) if p % 2 != team and not game.finished[p]]
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    if mates_done >= 2 or (opps and min(opps) <= 3) or hand.sum() <= 4:
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]

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

def weak(game, err=0.4):
    if np.random.random() < err:
        return np.random.randint(len(game.get_actions()))
    return rule(game)

def gen_data(n=5000):
    game = Game()
    ss, ls = [], []
    for _ in range(n):
        game.reset()
        while True:
            s = game.get_state()
            acts = game.get_actions()
            i = smart(game)
            ss.append(s)
            ls.append(15 if acts[i][0] < 0 else acts[i][0])
            if game.step(*acts[i])[0]: break
    return np.array(ss, dtype=np.float32), np.array(ls, dtype=np.int64)

def train_sl(model, ss, ls, epochs=15):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    n = len(ss)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        loss_sum = 0
        for i in range(0, n, 256):
            b = idx[i:i+256]
            opt.zero_grad()
            loss = crit(model(torch.from_numpy(ss[b])), torch.from_numpy(ls[b]))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if (ep+1) % 3 == 0:
            print(f"Epoch {ep+1}/{epochs}, Loss: {loss_sum/(n//256):.4f}")

def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            acts = game.get_actions()
            i = model.get_action(game.get_state(), acts) if game.current % 2 == 0 else \
                (rule(game) if opp == 'rule' else np.random.randint(len(acts)))
            if game.step(*acts[i])[0]:
                if game.finish_order[0] % 2 == 0: wins += 1
                break
    return wins / n * 100

def fine_tune(model, n=20000):
    game = Game()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    best_r, best_rand = 0, 0
    print("\n微调中...")
    for g in range(n):
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
                ai = torch.multinomial(vp, 1).item() if np.random.random() > 0.05 else np.random.randint(len(valid))
                trans.append((s, valid[ai]))
                card, cnt = acts[ai]
            else:
                acts = game.get_actions()
                i = weak(game, 0.4)
                card, cnt = acts[i]
            if game.step(card, cnt)[0]:
                rw = 1.0 if game.finish_order[0] % 2 == 0 else -0.2
                for s, a in trans:
                    opt.zero_grad()
                    loss = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a] * rw
                    loss.backward()
                    opt.step()
                break
        if (g+1) % 2000 == 0:
            r1, r2 = test(model, 300, 'rule'), test(model, 300, 'random')
            if r1 > best_r: best_r = r1; torch.save(model.state_dict(), '/workspace/projects/rl_v9_best_rule.pt')
            if r2 > best_rand: best_rand = r2; torch.save(model.state_dict(), '/workspace/projects/rl_v9_best_random.pt')
            print(f"{g+1:>5,} | vs规则: {r1:>5.1f}% | vs随机: {r2:>5.1f}% | 最佳: {max(best_r, best_rand):>5.1f}%")
            json.dump({'episode': g+1, 'vs_rule': r1, 'vs_random': r2, 'best_rule': best_r, 'best_random': best_rand,
                      'training_type': 'v9', 'status': 'training', 'timestamp': datetime.now().isoformat()},
                     open('/workspace/projects/training_status.json', 'w'), indent=2)
    return model

print("=" * 50)
print("V9 快速版")
print("=" * 50)
model = Net()
print("\n[阶段1] 监督学习...")
ss, ls = gen_data(5000)
print(f"生成 {len(ss)} 条数据")
train_sl(model, ss, ls, 15)
print(f"\n预训练后: vs规则 {test(model, 300, 'rule'):.1f}%, vs随机 {test(model, 300, 'random'):.1f}%")
torch.save(model.state_dict(), '/workspace/projects/rl_v9_pretrained.pt')
print("\n[阶段2] 策略梯度微调...")
model = fine_tune(model, 30000)
print("\n最终测试:")
if __import__('os').path.exists('/workspace/projects/rl_v9_best_rule.pt'):
    model.load_state_dict(torch.load('/workspace/projects/rl_v9_best_rule.pt', weights_only=True))
r1, r2 = test(model, 1000, 'rule'), test(model, 1000, 'random')
print(f"vs规则Bot: {r1:.1f}%")
print(f"vs随机Bot: {r2:.1f}%")
torch.save(model.state_dict(), '/workspace/projects/rl_v9_final.pt')
print("\n" + "=" * 50)
print("✅ vs规则Bot达标" if r1 >= 60 else f"⚠️ vs规则Bot未达标 ({r1:.1f}%)")
print("✅ vs随机Bot达标" if r2 >= 60 else f"⚠️ vs随机Bot未达标 ({r2:.1f}%)")
