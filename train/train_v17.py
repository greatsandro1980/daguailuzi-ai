#!/usr/bin/env python3
"""
V17 - 专项优化vs随机Bot
使用遗传算法+强化学习混合策略
目标：vs规则 ≥90%, vs随机 ≥90%
"""

import torch
import torch.nn as nn
import numpy as np
import os
import functools
import random
from collections import deque
print = functools.partial(print, flush=True)

# ============== 游戏环境（精简版） ==============
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
            s[30+i] = self.hands[i].sum() / 27.0
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
        s[51] = hand.sum() / 27.0
        s[52] = self.trick_count / 15.0
        s[53:58] = np.bincount([i for i in range(6) if self.finished[i]], minlength=6).astype(np.float32)[:5]
        s[58] = 1.0 if self.last_play is not None and (self.last_play[13] > 0 or self.last_play[14] > 0) else 0.0
        s[59] = 1.0 if self.last_play is None or self.last_player == self.current else 0.0
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
        team = self.current % 2
        
        if card >= 0:
            hand[card] -= cnt
            reward += 0.1 + cnt * 0.1
            
            # 策略奖励
            if self.last_play is None or self.last_player == self.current:
                if card <= 5:
                    reward += 0.05
                elif card >= 13:
                    reward -= 0.1
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None else 0
                if last_val >= 10:
                    reward += 0.05
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
            return True, winner, reward + (2.0 if winner == 0 else 0)
        
        return False, -1, reward


# ============== 网络架构 ==============
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)
    
    def act(self, state, actions, temp=1.0):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        
        valid = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits[valid] / temp, dim=0)
        return torch.multinomial(probs, 1).item()


# ============== 对手策略 ==============
def random_action(game):
    actions = game.get_actions()
    return np.random.randint(len(actions))


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


def mixed_action(game, model, temp=1.0):
    """混合对手：随机+模型+规则"""
    if np.random.random() < 0.4:
        return random_action(game)
    elif np.random.random() < 0.7:
        return rule_action(game)
    else:
        return model.act(game.get_state(), game.get_actions(), temp)


# ============== 测试函数 ==============
def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.act(game.get_state(), actions, temp=0.1)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                else:
                    idx = random_action(game)
            done, winner, _ = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== 进化策略优化 ==============
def evolve_population(population, game, n_episodes=50, mutation_rate=0.1):
    """评估种群并进化"""
    scores = []
    
    for model in population:
        r1 = test(model, n_episodes, 'rule')
        r2 = test(model, n_episodes, 'random')
        combined = r1 + r2
        scores.append((combined, r1, r2, model))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # 保留前25%
    n_keep = max(2, len(population) // 4)
    survivors = [s[3] for s in scores[:n_keep]]
    
    print(f"  最佳: 综合{scores[0][0]:.1f}% (规则{scores[0][1]:.1f}%/随机{scores[0][2]:.1f}%)")
    
    # 创建新一代
    new_pop = survivors.copy()
    while len(new_pop) < len(population):
        # 选择父代
        parent = random.choice(survivors)
        child = PolicyNet()
        child.load_state_dict(parent.state_dict())
        
        # 变异
        with torch.no_grad():
            for p in child.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(p) * 0.05
                    p.add_(noise)
        
        new_pop.append(child)
    
    return new_pop, scores[0]


# ============== 主训练函数 ==============
def main():
    print("=" * 60)
    print("V17 - 专项优化vs随机Bot")
    print("策略：进化算法 + 强化学习混合")
    print("目标：vs规则 ≥90%, vs随机 ≥90%")
    print("=" * 60)
    
    game = Game()
    population_size = 20
    n_generations = 30
    
    # 初始化种群
    population = [PolicyNet() for _ in range(population_size)]
    
    # 加载预训练模型
    if os.path.exists('/workspace/projects/rl_v14b_best.pt'):
        print("从V14b加载预训练权重...")
        state_dict = torch.load('/workspace/projects/rl_v14b_best.pt', weights_only=True)
        for model in population:
            model.load_state_dict(state_dict)
        print("加载成功!")
    
    # 初始测试
    best = population[0]
    r1 = test(best, 500, 'rule')
    r2 = test(best, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%\n")
    
    best_combined = r1 + r2
    best_rule = r1
    best_random = r2
    best_model = None
    
    # 进化训练
    print(f"开始进化训练 ({n_generations}代, 种群{population_size})...")
    print("=" * 60)
    
    for gen in range(n_generations):
        # 动态调整变异率
        mutation_rate = max(0.02, 0.2 - gen * 0.005)
        
        population, (combined, rule, rand) = evolve_population(
            population, game, n_episodes=100, mutation_rate=mutation_rate
        )
        
        if combined > best_combined:
            best_combined = combined
            best_rule = rule
            best_random = rand
            best_model = population[0]
            torch.save(best_model.state_dict(), '/workspace/projects/rl_v17_best.pt')
            print(f"  🌟 第{gen+1}代: 新最佳! 综合得分{combined:.1f}%")
        
        # 目标检查
        if rule >= 90 and rand >= 90:
            print(f"\n🎯 目标达成! vs规则{rule:.1f}%, vs随机{rand:.1f}%")
            break
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试 (1000局):")
    
    if os.path.exists('/workspace/projects/rl_v17_best.pt'):
        final_model = PolicyNet()
        final_model.load_state_dict(torch.load('/workspace/projects/rl_v17_best.pt', weights_only=True))
    else:
        final_model = population[0]
    
    r1 = test(final_model, 1000, 'rule')
    r2 = test(final_model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  综合得分: {r1+r2:.1f}%")
    
    # 结果评估
    if r1 >= 90 and r2 >= 90:
        print("\n🎯 双目标达成!")
        print("   ✓ vs规则 ≥ 90%")
        print("   ✓ vs随机 ≥ 90%")
    else:
        print("\n📊 当前状态:")
        print(f"   {'✓' if r1 >= 90 else '○'} vs规则: {r1:.1f}% (目标90%)")
        print(f"   {'✓' if r2 >= 90 else '○'} vs随机: {r2:.1f}% (目标90%)")


if __name__ == '__main__':
    main()
