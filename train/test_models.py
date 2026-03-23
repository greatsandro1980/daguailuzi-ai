#!/usr/bin/env python3
"""最终模型测试脚本"""
import torch
import torch.nn as nn
import numpy as np
import os

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


def main():
    print("=" * 60)
    print("最终模型测试")
    print("=" * 60)
    
    models = {
        'V9 最佳': '/workspace/projects/rl_v9_best_rule.pt',
        'V12 快速': '/workspace/projects/rl_v12_quick_rule.pt',
        'V12b 最佳': '/workspace/projects/rl_v12b_best_random.pt',
    }
    
    results = []
    for name, path in models.items():
        if os.path.exists(path):
            model = Net()
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            
            r_rule = test(model, 500, 'rule')
            r_random = test(model, 500, 'random')
            
            results.append({
                'name': name,
                'vs_rule': r_rule,
                'vs_random': r_random
            })
            
            print(f"\n{name}:")
            print(f"  vs规则Bot: {r_rule:.1f}%")
            print(f"  vs随机Bot: {r_random:.1f}%")
        else:
            print(f"\n{name}: 模型文件不存在")
    
    print("\n" + "=" * 60)
    print("结果汇总:")
    print("-" * 60)
    print(f"{'模型':<15} {'vs规则':>10} {'vs随机':>10} {'状态':>15}")
    print("-" * 60)
    
    for r in results:
        rule_status = "✅达标" if r['vs_rule'] >= 60 else "❌未达标"
        random_status = "✅达标" if r['vs_random'] >= 70 else "❌未达标"
        status = f"{rule_status}/{random_status}"
        print(f"{r['name']:<15} {r['vs_rule']:>9.1f}% {r['vs_random']:>9.1f}% {status:>15}")
    
    print("=" * 60)
    
    # 找出最佳模型
    best = max(results, key=lambda x: x['vs_rule'] + x['vs_random'])
    print(f"\n最佳模型: {best['name']}")
    print(f"综合得分: vs规则 {best['vs_rule']:.1f}% + vs随机 {best['vs_random']:.1f}%")


if __name__ == '__main__':
    main()
