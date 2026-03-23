#!/usr/bin/env python3
"""
V19 - 纯进化策略优化版
只使用进化策略，避免PPO微调导致的性能下降
目标：vs规则 ≥90%, vs随机 ≥90%
"""

import torch
import torch.nn as nn
import numpy as np
import os
import functools
import random
from copy import deepcopy
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
        reward = 0.0
        strategic_reward = 0.0
        hand = self.hands[self.current]
        team = self.current % 2
        
        if card >= 0:
            hand[card] -= cnt
            reward += 0.1 + cnt * 0.1
            if cnt >= 2:
                reward += 0.1
            
            # 策略奖励
            if self.last_play is None or self.last_player == self.current:
                if card <= 5: strategic_reward += 0.05
                elif card >= 11 and cnt == 1: strategic_reward -= 0.03
            
            if card == 14:
                if self.trick_count < 3: strategic_reward -= 0.2
                elif self.trick_count >= 6: strategic_reward += 0.1
            elif card == 13:
                if self.trick_count < 2: strategic_reward -= 0.15
                elif self.trick_count >= 5: strategic_reward += 0.05
            
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None else 0
                if last_val >= 11: strategic_reward -= 0.15
                elif last_val >= 9: strategic_reward -= 0.05
            
            if hand.sum() <= 3: strategic_reward += 0.1
            elif hand.sum() <= 5: strategic_reward += 0.05
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            reward -= 0.02
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None else 0
                if last_val >= 10: strategic_reward += 0.1
            self.passes += 1
        
        total_reward = reward + strategic_reward
        
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
            if winner == 0:
                total_reward += 2.0
            return True, winner, total_reward
        
        return False, -1, total_reward


# ============== ActorCritic网络 ==============
class ActorCritic(nn.Module):
    def __init__(self, state_dim=60, action_dim=16, hidden_dim=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, state, actions, deterministic=True):
        with torch.no_grad():
            logits, value = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits, 0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        valid_probs = valid_probs / valid_probs.sum()
        if deterministic:
            a_idx = valid_probs.argmax().item()
        else:
            a_idx = torch.multinomial(valid_probs, 1).item()
        return a_idx, valid[a_idx], value.item()


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


# ============== 测试函数 ==============
def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx, _, _ = model.get_action(game.get_state(), actions, deterministic=True)
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


# ============== 进化策略训练 ==============
def evolve_train(base_model, n_generations=100, pop_size=30, n_test=300):
    """使用进化策略优化模型"""
    
    print(f"\n进化策略训练: {n_generations}代, 种群{pop_size}, 测试{n_test}局")
    print("=" * 60)
    
    best_model = deepcopy(base_model)
    best_rule = 0
    best_random = 0
    best_combined = 0
    
    for gen in range(n_generations):
        # 动态变异率
        if gen < 20:
            noise_scale = 0.03
        elif gen < 50:
            noise_scale = 0.02
        else:
            noise_scale = 0.01
        
        # 变异生成种群
        population = [deepcopy(best_model)]
        for _ in range(pop_size - 1):
            child = deepcopy(best_model)
            with torch.no_grad():
                for p in child.parameters():
                    noise = torch.randn_like(p) * noise_scale
                    p.add_(noise)
            population.append(child)
        
        # 评估
        scores = []
        for i, model in enumerate(population):
            r1 = test(model, n_test, 'rule')
            r2 = test(model, n_test, 'random')
            combined = r1 + r2
            scores.append((combined, r1, r2, model))
        
        # 选择最佳
        scores.sort(reverse=True, key=lambda x: x[0])
        best = scores[0]
        
        if best[0] > best_combined:
            best_combined = best[0]
            best_rule = best[1]
            best_random = best[2]
            best_model = best[3]
            torch.save(best_model.state_dict(), '/workspace/projects/rl_v19_best.pt')
            marker = "★ 新最佳"
        else:
            marker = ""
        
        print(f"  Gen {gen+1:>3}: 规则{best[1]:>5.1f}% | 随机{best[2]:>5.1f}% | 综合{best[0]:>5.1f}% {marker}")
        
        # 目标检查
        if best[1] >= 90 and best[2] >= 90:
            print(f"\n🎯 目标达成! 第{gen+1}代")
            break
    
    return best_model, best_rule, best_random


# ============== 主函数 ==============
def main():
    print("=" * 60)
    print("V19 - 纯进化策略优化版")
    print("策略：只使用进化策略，避免PPO微调导致性能下降")
    print("目标：vs规则 ≥90%, vs随机 ≥90%")
    print("=" * 60)
    
    model = ActorCritic()
    
    # 加载V14b
    if os.path.exists('/workspace/projects/rl_v14b_best.pt'):
        print("\n从V14b加载预训练权重...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v14b_best.pt', weights_only=True))
        print("加载成功!")
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"\n初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 进化训练
    model, r1, r2 = evolve_train(model, n_generations=100, pop_size=30, n_test=300)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试 (1000局):")
    
    if os.path.exists('/workspace/projects/rl_v19_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v19_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  综合得分: {r1+r2:.1f}%")
    
    # 结果评估
    print("\n" + "=" * 60)
    if r1 >= 90 and r2 >= 90:
        print("🎯 双目标达成!")
        print("   ✓ vs规则 ≥ 90%")
        print("   ✓ vs随机 ≥ 90%")
    else:
        print("📊 当前状态:")
        print(f"   {'✓' if r1 >= 90 else '○'} vs规则: {r1:.1f}% (目标90%)")
        print(f"   {'✓' if r2 >= 90 else '○'} vs随机: {r2:.1f}% (目标90%)")


if __name__ == '__main__':
    main()
