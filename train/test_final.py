#!/usr/bin/env python3
"""最终测试 - 对比各模型性能"""

import torch
import torch.nn as nn
import numpy as np
import os
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# 游戏环境和网络
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.trick_count = 0
        
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
        hand = self.hands[self.current]
        team = self.current % 2
        
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
            self.trick_count += 1
        
        team0_done = all(self.finished[p] for p in [0, 2, 4])
        team1_done = all(self.finished[p] for p in [1, 3, 5])
        
        if team0_done or team1_done:
            winner = 0 if team0_done else 1
            return True, winner
        return False, -1


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(60, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 16))
        self.critic = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
    
    def forward(self, x):
        return self.actor(self.shared(x)), self.critic(self.shared(x))
    
    def get_action(self, state, actions):
        with torch.no_grad():
            logits, _ = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits, 0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        return valid_probs.argmax().item()


def rule_action(game):
    actions = game.get_actions()
    valid = [(i, c) for i, (c, _) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c) for i, c in valid if actions[i][1] >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def test(model, n=1000, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.get_action(game.get_state(), actions)
            else:
                idx = rule_action(game) if opp == 'rule' else np.random.randint(len(actions))
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


def main():
    print("=" * 60)
    print("最终测试 - 对比各模型性能")
    print("=" * 60)
    
    models = {
        'V14b': '/workspace/projects/rl_v14b_best.pt',
        'V18': '/workspace/projects/rl_v18_best.pt',
        'V19': '/workspace/projects/rl_v19_best.pt',
        'V20': '/workspace/projects/rl_v20_best.pt',
    }
    
    results = []
    
    for name, path in models.items():
        if os.path.exists(path):
            model = ActorCritic()
            model.load_state_dict(torch.load(path, weights_only=True))
            
            r1 = test(model, 1000, 'rule')
            r2 = test(model, 1000, 'random')
            combined = r1 + r2
            
            results.append((name, r1, r2, combined))
            print(f"{name}: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%, 综合 {combined:.1f}%")
    
    # 找最佳
    results.sort(key=lambda x: x[3], reverse=True)
    best = results[0]
    
    print("\n" + "=" * 60)
    print("最佳模型:")
    print(f"  {best[0]}: vs规则 {best[1]:.1f}%, vs随机 {best[2]:.1f}%")
    
    # 目标检查
    print("\n目标达成情况:")
    print(f"  {'✓' if best[1] >= 90 else '○'} vs规则 ≥ 90%: {best[1]:.1f}%")
    print(f"  {'✓' if best[2] >= 90 else '○'} vs随机 ≥ 90%: {best[2]:.1f}%")


if __name__ == '__main__':
    main()
