#!/usr/bin/env python3
"""训练 V7 - 学习强规则Bot策略"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
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
        s = np.zeros(80, dtype=np.float32)
        s[:15] = self.hands[self.current]
        if self.last_play is not None:
            s[15:30] = self.last_play
        for i in range(6):
            s[30+i] = self.hands[i].sum() / 10.0
        
        # 出牌类型
        if self.last_play is not None and self.last_play.sum() > 0:
            cnt = int(self.last_play[self.last_play > 0][0])
            s[36 + min(cnt-1, 2)] = 1.0
        
        s[39] = 1.0 if self.last_play is None else 0.0
        
        team = self.current % 2
        for i, p in enumerate([p for p in range(6) if p % 2 == team and p != self.current]):
            s[40+i] = 1.0 if self.finished[p] else 0.0
            s[46+i] = self.hands[p].sum() / 10.0
        for i, p in enumerate([p for p in range(6) if p % 2 != team]):
            s[43+i] = 1.0 if self.finished[p] else 0.0
            s[49+i] = self.hands[p].sum() / 10.0
        
        hand = self.hands[self.current]
        s[52] = sum(1 for i in range(13) if hand[i] == 1) / 13.0
        s[53] = sum(1 for i in range(13) if hand[i] == 2) / 6.0
        s[54] = sum(1 for i in range(13) if hand[i] >= 3) / 4.0
        s[55] = hand[13] / 2.0
        s[56] = hand[14] / 2.0
        s[57] = hand.sum() / 10.0
        
        # 队友已出完数量
        s[58] = sum(1 for p in range(6) if p % 2 == team and self.finished[p]) / 3.0
        # 对手已出完数量
        s[59] = sum(1 for p in range(6) if p % 2 != team and self.finished[p]) / 3.0
        
        # 当前领先队伍
        if self.finish_order:
            s[60] = 1.0 if self.finish_order[0] % 2 == team else -1.0
        
        # 关键牌信息
        s[61] = 1.0 if hand[13] > 0 else 0.0  # 有小王
        s[62] = 1.0 if hand[14] > 0 else 0.0  # 有大王
        s[63] = hand[:13].max() / 3.0 if hand[:13].max() > 0 else 0  # 最多同值牌
        
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
            nn.Linear(80, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 16)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, actions):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid_idx = [15 if c < 0 else c for c, _ in actions]
        scores = [(i, logits[idx].item()) for i, idx in enumerate(valid_idx)]
        return max(scores, key=lambda x: x[1])[0]

def smart_rule_action(game):
    """智能规则Bot - 考虑更多策略因素"""
    actions = game.get_actions()
    hand = game.hands[game.current]
    team = game.current % 2
    
    # 统计
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    opps_done = sum(1 for p in range(6) if p % 2 != team and game.finished[p])
    mates_remaining = [game.hands[p].sum() for p in range(6) if p % 2 == team and not game.finished[p] and p != game.current]
    opps_remaining = [game.hands[p].sum() for p in range(6) if p % 2 != team and not game.finished[p]]
    
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    
    # 策略1: 如果队友都快出完了，帮队友控场（出大牌抢控制权）
    if mates_done >= 2:
        # 出最大的牌抢控制权
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    
    # 策略2: 如果对手有人牌少，封锁对手
    if opps_remaining and min(opps_remaining) <= 3:
        # 出大牌压制
        valid.sort(key=lambda x: -x[1])
        pairs = [x for x in valid if x[2] >= 2]
        if pairs:
            return pairs[0][0]
        return valid[0][0]
    
    # 策略3: 如果自己牌少，快速出完
    if hand.sum() <= 4:
        valid.sort(key=lambda x: -x[1])  # 大牌优先
        return valid[0][0]
    
    # 策略4: 正常情况，优先对子三张，小牌优先
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])  # 小牌优先
        return pairs[0][0]
    
    # 单张小牌优先
    valid.sort(key=lambda x: x[1])
    return valid[0][0]

def rule_action(game):
    """普通规则Bot"""
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]

def generate_data(n=10000, use_smart=True):
    game = SimpleGame()
    states, labels = [], []
    for _ in range(n):
        game.reset()
        while True:
            state = game.get_state()
            actions = game.get_actions()
            idx = smart_rule_action(game) if use_smart else rule_action(game)
            states.append(state)
            labels.append(15 if actions[idx][0] < 0 else actions[idx][0])
            done, _ = game.step(*actions[idx])
            if done: break
    return np.array(states), np.array(labels)

def train_supervised(model, states, labels, epochs=15):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    n = len(states)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        loss_sum = 0
        for i in range(0, n, 256):
            b = idx[i:i+256]
            opt.zero_grad()
            loss = crit(model(torch.FloatTensor(states[b])), torch.LongTensor(labels[b]))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {ep+1}/{epochs}, Loss: {loss_sum/(n//256):.4f}")

def test(model, n=200, opp='rule'):
    game = SimpleGame()
    wins = 0
    scores = 0
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
                    idx = smart_rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0:
                    wins += 1
                    if game.finish_order and game.finish_order[-1] % 2 == 1:
                        scores += 1
                break
    return wins / n * 100, scores / n * 100

def main():
    print("=" * 60)
    print("V7 训练 - 学习智能规则Bot策略")
    print("=" * 60)
    
    model = PolicyNet()
    
    print("\n[阶段1] 生成数据 (学习智能策略)...")
    states, labels = generate_data(10000, use_smart=True)
    print(f"生成 {len(states)} 条训练数据")
    
    print("\n[阶段2] 监督学习...")
    train_supervised(model, states, labels, epochs=15)
    
    print("\n[阶段3] 测试...")
    r1, s1 = test(model, 500, 'rule')
    r2, s2 = test(model, 500, 'random')
    r3, s3 = test(model, 500, 'smart')
    
    print(f"\n最终结果:")
    print(f"  vs规则Bot: {r1:.1f}% (得分率: {s1:.1f}%)")
    print(f"  vs随机Bot: {r2:.1f}% (得分率: {s2:.1f}%)")
    print(f"  vs智能Bot: {r3:.1f}% (得分率: {s3:.1f}%)")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v7_final.pt')
    print("\n模型已保存!")
    
    # 保存状态
    status = {
        'vs_rule': round(r1, 1),
        'vs_random': round(r2, 1),
        'vs_smart': round(r3, 1),
        'best_rate': round(max(r1, r2), 1),
        'training_type': 'v7_supervised',
        'status': 'completed',
        'timestamp': datetime.now().isoformat()
    }
    with open('/workspace/projects/training_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    # 判断是否达标
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
