#!/usr/bin/env python3
"""
训练 V9 - 高效版本
策略：
1. 大规模监督学习（学习智能策略）
2. 简单策略梯度微调
3. 对手使用弱化策略（有犯错概率）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from collections import deque

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
    
    def get_state(self):
        s = np.zeros(80, dtype=np.float32)
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
        s[52] = sum(1 for p in range(6) if p % 2 == team and self.finished[p])
        s[53] = sum(1 for p in range(6) if p % 2 != team and self.finished[p])
        if self.finish_order:
            s[54] = 1.0 if self.finish_order[0] % 2 == team else -1.0
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


# ============== 网络 ==============
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(80, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 16)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, actions):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        scores = [(i, logits[idx].item()) for i, idx in enumerate(valid)]
        return max(scores, key=lambda x: x[1])[0]


# ============== 策略 ==============
def smart_action(game):
    """智能策略"""
    actions = game.get_actions()
    hand = game.hands[game.current]
    team = game.current % 2
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    opps_remaining = [game.hands[p].sum() for p in range(6) if p % 2 != team and not game.finished[p]]
    
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    
    # 关键策略
    if mates_done >= 2:
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    
    if opps_remaining and min(opps_remaining) <= 3:
        valid.sort(key=lambda x: -x[1])
        pairs = [x for x in valid if x[2] >= 2]
        if pairs: return pairs[0][0]
        return valid[0][0]
    
    if hand.sum() <= 4:
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def rule_action(game):
    """普通规则"""
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def weak_action(game, error_rate=0.35):
    """弱化策略 - 有概率犯错"""
    if np.random.random() < error_rate:
        actions = game.get_actions()
        return np.random.randint(len(actions))
    return rule_action(game)


# ============== 训练 ==============
def generate_data(n=20000, use_smart=True):
    game = Game()
    states, labels = [], []
    for _ in range(n):
        game.reset()
        while True:
            state = game.get_state()
            actions = game.get_actions()
            idx = smart_action(game) if use_smart else rule_action(game)
            states.append(state)
            labels.append(15 if actions[idx][0] < 0 else actions[idx][0])
            done, _ = game.step(*actions[idx])
            if done: break
    return np.array(states, dtype=np.float32), np.array(labels, dtype=np.int64)


def train_supervised(model, states, labels, epochs=20, batch=512):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    n = len(states)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        loss_sum = 0
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss = crit(model(torch.from_numpy(states[b])), torch.from_numpy(labels[b]))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1}/{epochs}, Loss: {loss_sum/(n//batch):.4f}")


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
                elif opp == 'smart':
                    idx = smart_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


def fine_tune(model, n_games=30000):
    """策略梯度微调 - 对抗弱化对手"""
    game = Game()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    best_rule = 0
    best_random = 0
    
    print("\n策略梯度微调...")
    for g in range(n_games):
        game.reset()
        transitions = []
        
        while True:
            if game.current % 2 == 0:
                state = game.get_state()
                actions = game.get_actions()
                logits = model(torch.FloatTensor(state))
                valid = [15 if c < 0 else c for c, _ in actions]
                probs = torch.softmax(logits, dim=0)
                valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
                valid_probs = valid_probs / valid_probs.sum()
                
                if np.random.random() < 0.05:
                    a_idx = np.random.randint(len(valid))
                else:
                    a_idx = torch.multinomial(valid_probs, 1).item()
                
                transitions.append((state, valid[a_idx]))
                card, cnt = actions[a_idx]
            else:
                actions = game.get_actions()
                idx = weak_action(game, error_rate=0.4)
                card, cnt = actions[idx]
            
            done, winner = game.step(card, cnt)
            if done:
                reward = 1.0 if winner == 0 else -0.2
                for s, a in transitions:
                    opt.zero_grad()
                    log_prob = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a]
                    loss = log_prob * reward
                    loss.backward()
                    opt.step()
                break
        
        if (g + 1) % 3000 == 0:
            r1 = test(model, 300, 'rule')
            r2 = test(model, 300, 'random')
            if r1 > best_rule: 
                best_rule = r1
                torch.save(model.state_dict(), '/workspace/projects/rl_v9_best_rule.pt')
            if r2 > best_random:
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v9_best_random.pt')
            print(f"{g+1:>6,} | vs规则: {r1:>5.1f}% | vs随机: {r2:>5.1f}% | 最佳: {max(best_rule, best_random):>5.1f}%")
            
            status = {
                'episode': g+1, 'vs_rule': r1, 'vs_random': r2,
                'best_rule': best_rule, 'best_random': best_random,
                'training_type': 'v9', 'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    return model


def main():
    print("=" * 60)
    print("V9 高效训练版")
    print("=" * 60)
    
    model = Net()
    
    # 阶段1：监督学习
    print("\n[阶段1] 监督学习预训练...")
    states, labels = generate_data(30000, use_smart=True)
    print(f"生成 {len(states)} 条训练数据")
    train_supervised(model, states, labels, epochs=25)
    
    # 测试预训练效果
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"\n预训练后: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    torch.save(model.state_dict(), '/workspace/projects/rl_v9_pretrained.pt')
    
    # 阶段2：策略梯度微调
    print("\n[阶段2] 策略梯度微调...")
    model = fine_tune(model, 50000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试 (1000局):")
    
    # 加载最佳模型
    if os.path.exists('/workspace/projects/rl_v9_best_rule.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v9_best_rule.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    r3 = test(model, 1000, 'smart')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  vs智能Bot: {r3:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v9_final.pt')
    
    print("\n" + "=" * 60)
    if r1 >= 60:
        print("✅ vs规则Bot: 达标")
    else:
        print(f"⚠️ vs规则Bot: 未达标 ({r1:.1f}% < 60%)")
    if r2 >= 60:
        print("✅ vs随机Bot: 达标")
    else:
        print(f"⚠️ vs随机Bot: 未达标 ({r2:.1f}% < 60%)")


if __name__ == '__main__':
    main()
