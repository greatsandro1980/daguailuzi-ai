#!/usr/bin/env python3
"""
V14d - 直接基于V14b训练脚本优化
目标：vs规则 ≥90%, vs随机 ≥90%
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import functools
print = functools.partial(print, flush=True)

# ============== 完全复制V14b的游戏环境 ==============
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
        """增强版step，包含更多策略奖励"""
        reward = 0.0
        strategic_reward = 0.0
        
        hand = self.hands[self.current]
        team = self.current % 2
        prev_jokers = hand[13] + hand[14]
        
        if card >= 0:
            hand[card] -= cnt
            
            # 基础奖励
            reward += 0.1 + cnt * 0.1
            if cnt >= 2:
                reward += 0.1
            
            # ====== 增强策略奖励 ======
            
            # 1. 首发策略：强烈鼓励小牌
            if self.last_play is None or self.last_player == self.current:
                if card <= 5:  # 小牌 (2-7)
                    strategic_reward += 0.08
                elif card <= 8:
                    strategic_reward += 0.03
                elif card >= 11 and card < 13:  # JQK首发单张
                    strategic_reward -= 0.04
                elif card >= 13:  # 王牌首发
                    strategic_reward -= 0.5  # 更强惩罚
                    
                # 首发多张优于单张
                if cnt >= 2:
                    strategic_reward += 0.06
            
            # 2. 大小王策略：更严格控制
            if card == 14:  # 大王
                if self.trick_count < 3:
                    strategic_reward -= 0.35
                elif self.trick_count < 6:
                    strategic_reward -= 0.15
                elif hand.sum() <= 3:  # 收尾阶段
                    strategic_reward += 0.2
                else:
                    strategic_reward += 0.05
            elif card == 13:  # 小王
                if self.trick_count < 2:
                    strategic_reward -= 0.25
                elif self.trick_count < 5:
                    strategic_reward -= 0.08
                elif hand.sum() <= 2:
                    strategic_reward += 0.15
                else:
                    strategic_reward += 0.03
            
            # 3. 配合策略：队友大牌不压
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None and self.last_play.max() > 0 else 0
                if last_val >= 11:
                    strategic_reward -= 0.25
                elif last_val >= 9:
                    strategic_reward -= 0.08
            
            # 4. 清牌奖励
            if hand.sum() <= 3:
                strategic_reward += 0.12
            elif hand.sum() <= 5:
                strategic_reward += 0.06
            
            # 5. 减少单张碎片奖励
            new_singles = sum(1 for i in range(13) if hand[i] == 1)
            if new_singles <= 2:
                strategic_reward += 0.03
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.6
        else:
            # pass
            reward -= 0.025
            
            # 配合奖励
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None and self.last_play.max() > 0 else 0
                if last_val >= 10:
                    strategic_reward += 0.12
            
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
                total_reward += 2.5  # 更高的胜利奖励
                # 快速获胜奖励
                if self.trick_count <= 5:
                    total_reward += 1.5
            return True, winner, total_reward
        
        return False, -1, total_reward


# ============== 网络结构（与V14b完全一致） ==============
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
                idx, _, _ = model.get_action(game.get_state(), actions, deterministic=True)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner, _ = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== 增强训练 ==============
def train(model, n_games=8000, lr=8e-5):
    game = Game()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_games)
    
    best_rule = 0
    best_random = 0
    best_combined = 0
    
    memory = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': []}
    
    print(f"\n强化训练 ({n_games}局)...")
    print("=" * 60)
    
    for g in range(n_games):
        game.reset()
        episode_data = []
        
        while True:
            state = game.get_state()
            actions = game.get_actions()
            
            if game.current % 2 == 0:
                idx, action, value = model.get_action(state, actions, deterministic=False)
                with torch.no_grad():
                    logits, _ = model(torch.FloatTensor(state))
                log_prob = torch.log_softmax(logits, 0)[action].item()
                episode_data.append({'state': state, 'action': action, 'value': value, 'log_prob': log_prob})
                card, cnt = actions[idx]
            else:
                # 混合对手：60%规则 + 40%随机
                if np.random.random() < 0.6:
                    idx = rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
                card, cnt = actions[idx]
            
            done, winner, reward = game.step(card, cnt)
            if episode_data:
                episode_data[-1]['reward'] = reward
            
            if done:
                returns = []
                R = 0
                for data in reversed(episode_data):
                    R = data.get('reward', 0) + 0.99 * R
                    returns.insert(0, R)
                
                for i, data in enumerate(episode_data):
                    memory['states'].append(data['state'])
                    memory['actions'].append(data['action'])
                    memory['rewards'].append(returns[i])
                    memory['values'].append(data['value'])
                    memory['log_probs'].append(data['log_prob'])
                break
        
        if len(memory['states']) >= 1024:
            states = torch.FloatTensor(np.array(memory['states']))
            actions = torch.LongTensor(memory['actions'])
            old_log_probs = torch.FloatTensor(memory['log_probs'])
            returns = torch.FloatTensor(memory['rewards'])
            old_values = torch.FloatTensor(memory['values'])
            
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 多轮PPO更新
            for _ in range(3):
                logits, new_values = model(states)
                new_log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.75, 1.25) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values.squeeze(), returns)
                entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.015 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            memory = {k: [] for k in memory}
            scheduler.step()
        
        if (g + 1) % 500 == 0:
            r1 = test(model, 500, 'rule')
            r2 = test(model, 500, 'random')
            combined = r1 + r2
            
            print(f"  {g+1:>5,}局 | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | 综合: {combined:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v14d_best.pt')
                print(f"    ★ 新最佳!")
            
            if r1 >= 90 and r2 >= 90:
                print(f"    🎯 目标达成!")
                break
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V14d - 冲击90%+胜率优化版")
    print("目标：vs规则 ≥90%, vs随机 ≥90%")
    print("=" * 60)
    
    model = ActorCritic(hidden_dim=512)
    
    # 加载V14b模型
    if os.path.exists('/workspace/projects/rl_v14b_best.pt'):
        print("加载V14b模型...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v14b_best.pt', weights_only=True))
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 训练
    model, best_rule, best_random = train(model, n_games=8000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试 (1000局):")
    
    if os.path.exists('/workspace/projects/rl_v14d_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v14d_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  综合得分: {r1+r2:.1f}%")
    
    if r1 >= 90 and r2 >= 90:
        print("\n🎯 目标达成!")
    else:
        print(f"\n当前最佳: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%")


if __name__ == '__main__':
    main()
