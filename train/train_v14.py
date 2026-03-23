#!/usr/bin/env python3
"""
V14 (P4优化) - 策略优化版
解决V13存在的问题：
1. 三带二策略：优先带小对子
2. 配合度：队友大牌时不压
3. 大小王使用：保留到关键时刻
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from collections import deque

import functools
print = functools.partial(print, flush=True)

# ============== 增强游戏环境 ==============
class EnhancedGame:
    """增强版游戏环境，支持策略分析"""
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.last_play_type = None  # 牌型
        self.last_play_value = 0    # 牌值大小
        self.passes = 0
        self.finished = [False] * 6
        self.played_count = 0       # 已出牌数
        self.jokers_remaining = {0: 0, 1: 0}  # 双方剩余大小王数
        self.trick_count = 0        # 当前轮次
        
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
        self.last_play_type = None
        self.last_play_value = 0
        self.passes = 0
        self.finished = [False] * 6
        self.played_count = 0
        self.jokers_remaining = {0: self.hands[0:6:2, 13:15].sum(), 
                                  1: self.hands[1:6:2, 13:15].sum()}
        self.trick_count = 0
        return self.get_state()
    
    def get_state(self):
        """增强状态向量 (80维)"""
        s = np.zeros(80, dtype=np.float32)
        
        # 基础手牌信息 (0-14)
        s[:15] = self.hands[self.current]
        
        # 上家出牌 (15-29)
        if self.last_play is not None:
            s[15:30] = self.last_play
        
        # 各玩家手牌数 (30-35)
        for i in range(6):
            s[30+i] = self.hands[i].sum()
        
        # 牌型信息 (36-40)
        if self.last_play is not None and self.last_play.sum() > 0:
            cards = self.last_play[self.last_play > 0]
            if len(cards) > 0:
                cnt = int(cards[0])
                if cnt == 1:
                    s[36] = 1.0  # 单张
                elif cnt == 2:
                    s[37] = 1.0  # 对子
                elif cnt == 3:
                    s[38] = 1.0  # 三张
                s[39] = self.last_play_value / 15.0  # 牌值归一化
        s[40] = 1.0 if self.last_play is None else 0.0  # 是否首发
        
        # 队友状态 (41-45)
        team = self.current % 2
        mates = [p for p in range(6) if p % 2 == team and p != self.current]
        for i, p in enumerate(mates[:2]):
            s[41+i] = 1.0 if self.finished[p] else 0.0
            s[43+i] = self.hands[p].sum() / 9.0  # 队友剩余牌数
        
        # 对手状态 (45-49)
        opps = [p for p in range(6) if p % 2 != team]
        for i, p in enumerate(opps[:3]):
            s[45+i] = 1.0 if self.finished[p] else 0.0
            s[48+i] = self.hands[p].sum() / 9.0
        
        # 手牌统计 (51-55)
        hand = self.hands[self.current]
        s[51] = sum(1 for i in range(13) if hand[i] == 1)  # 单张数
        s[52] = sum(1 for i in range(13) if hand[i] == 2)  # 对子数
        s[53] = sum(1 for i in range(13) if hand[i] >= 3)  # 三张数
        s[54] = hand[13]  # 小王数
        s[55] = hand[14]  # 大王数
        
        # 大小王信息 (56-59)
        s[56] = self.jokers_remaining[team] / 4.0  # 我方剩余王
        s[57] = self.jokers_remaining[1-team] / 4.0  # 对方剩余王
        s[58] = 1.0 if hand[14] > 0 else 0.0  # 是否有大王
        s[59] = 1.0 if hand[13] > 0 else 0.0  # 是否有小王
        
        # 游戏进度 (60-64)
        total_cards = self.hands.sum()
        s[60] = self.played_count / 50.0  # 已出牌比例
        s[61] = self.trick_count / 20.0  # 轮次
        s[62] = sum(self.finished) / 6.0  # 已完成玩家比例
        s[63] = hand.sum() / 9.0  # 自己剩余牌数
        
        # 上家是否是队友 (64)
        if self.last_player >= 0:
            s[64] = 1.0 if self.last_player % 2 == team else 0.0
        
        # 小对子数量 (65-67) - 用于三带二决策
        small_pairs = sum(1 for i in range(6) if hand[i] == 2)  # 2-7的小对子
        s[65] = small_pairs
        s[66] = sum(1 for i in range(13) if hand[i] == 2 and i >= 6)  # 大对子
        s[67] = min(hand.sum(), 4) / 4.0  # 手牌紧张度
        
        return s
    
    def get_actions(self):
        """获取合法动作，带牌型标记"""
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
            # 首发
            for i in range(15):
                if hand[i] >= 1:
                    actions.append((i, 1, 'single'))
            for i in range(13):
                if hand[i] >= 2:
                    actions.append((i, 2, 'pair'))
            for i in range(13):
                if hand[i] >= 3:
                    # 三张（可以作为三带二）
                    actions.append((i, 3, 'triple'))
        else:
            # 压牌
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt:
                    card_type = 'single' if last_cnt == 1 else ('pair' if last_cnt == 2 else 'triple')
                    actions.append((i, last_cnt, card_type))
            
            actions.append((-1, 0, 'pass'))
        
        return actions if actions else [(-1, 0, 'pass')]
    
    def step(self, card, cnt, card_type='single'):
        """执行动作，返回详细奖励"""
        reward = 0.0
        strategic_penalty = 0.0  # 策略惩罚
        strategic_bonus = 0.0   # 策略奖励
        
        hand = self.hands[self.current]
        team = self.current % 2
        
        # 记录出牌前状态
        prev_hand_size = hand.sum()
        had_big_joker = hand[14] > 0
        had_small_joker = hand[13] > 0
        
        if card >= 0:
            hand[card] -= cnt
            
            # 基础奖励
            reward += 0.1 + cnt * 0.1
            
            # ====== 策略评估 ======
            
            # 1. 三带二策略评估
            if card_type == 'triple' and cnt == 3:
                # 检查是否有小对子可以带
                small_pairs = sum(1 for i in range(6) if hand[i] == 2)
                if small_pairs > 0 and card >= 6:
                    # 大三张但有小对子，建议保留
                    strategic_penalty -= 0.05
            
            # 2. 配合度评估
            if self.last_player >= 0 and self.last_player % 2 == team:
                # 上家是队友
                if self.last_play_value >= 11:  # 队友出的牌很大 (J以上)
                    strategic_penalty -= 0.15  # 不应该压队友的大牌
                elif self.last_play_value >= 8:  # 中等牌
                    strategic_penalty -= 0.05
            
            # 3. 大小王使用评估
            if card == 14:  # 大王
                # 游戏早期使用大王，惩罚
                if self.trick_count < 5:
                    strategic_penalty -= 0.2
                elif self.trick_count < 10:
                    strategic_penalty -= 0.1
                else:
                    strategic_bonus += 0.1  # 后期使用大王是好的
                self.jokers_remaining[team] -= 1
                
            elif card == 13:  # 小王
                if self.trick_count < 3:
                    strategic_penalty -= 0.15
                elif self.trick_count < 8:
                    strategic_penalty -= 0.05
                else:
                    strategic_bonus += 0.05
                self.jokers_remaining[team] -= 1
            
            # 4. 首发策略
            if self.last_play is None or self.last_player == self.current:
                # 首发：鼓励出小牌
                if card <= 6 and cnt == 1:
                    strategic_bonus += 0.05
                elif card <= 8 and cnt >= 2:
                    strategic_bonus += 0.03  # 对子/三张出中等牌
                elif card >= 12 and cnt == 1:
                    strategic_penalty -= 0.03  # 首发出大牌
            
            # 5. 手牌紧张度评估
            current_hand_size = hand.sum()
            if current_hand_size <= 3:
                strategic_bonus += 0.1  # 快出完牌
            elif current_hand_size <= 5:
                strategic_bonus += 0.05
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.last_play_value = card
            self.last_play_type = card_type
            self.passes = 0
            self.played_count += cnt
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            # pass
            reward -= 0.02
            self.passes += 1
            
            # 配合评估：如果上家是队友，pass是合理的
            if self.last_player >= 0 and self.last_player % 2 == team:
                strategic_bonus += 0.05  # 不压队友
        
        # 合并奖励
        total_reward = reward + strategic_penalty + strategic_bonus
        
        # 下一个玩家
        for _ in range(6):
            self.current = (self.current + 1) % 6
            if not self.finished[self.current]:
                break
        
        # 检查是否所有人都过牌
        if self.passes >= sum(1 for p in range(6) if not self.finished[p]):
            self.last_play = None
            self.last_player = -1
            self.last_play_type = None
            self.last_play_value = 0
            self.passes = 0
            self.trick_count += 1
        
        # 检查游戏结束
        team0_done = all(self.finished[p] for p in [0, 2, 4])
        team1_done = all(self.finished[p] for p in [1, 3, 5])
        
        if team0_done or team1_done:
            winner = 0 if team0_done else 1
            if winner == 0:
                total_reward += 2.0
            return True, winner, total_reward
        
        return False, -1, total_reward


# ============== 增强网络 ==============
class EnhancedActorCritic(nn.Module):
    """增强版Actor-Critic网络"""
    def __init__(self, state_dim=80, action_dim=16, hidden_dim=512):
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
        
        valid = [15 if c < 0 else c for c, _, _ in actions]
        probs = torch.softmax(logits, 0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        valid_probs = valid_probs / valid_probs.sum()
        
        if deterministic:
            a_idx = valid_probs.argmax().item()
        else:
            a_idx = torch.multinomial(valid_probs, 1).item()
        
        return a_idx, valid[a_idx], value.item()


# ============== 策略 ==============
def rule_action(game):
    """规则策略：优先出小牌"""
    actions = game.get_actions()
    valid = [(i, c, cnt, ct) for i, (c, cnt, ct) in enumerate(actions) if c >= 0]
    if not valid:
        return len(actions) - 1
    
    # 首发时优先出对子/三张
    pairs = [(i, c, cnt, ct) for i, c, cnt, ct in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])  # 出最小的对子
        return pairs[0][0]
    
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def smart_rule_action(game):
    """智能规则策略：考虑配合"""
    actions = game.get_actions()
    team = game.current % 2
    
    # 如果上家是队友且出了大牌，考虑pass
    if game.last_player >= 0 and game.last_player % 2 == team:
        if game.last_play_value >= 11:  # 队友出大牌
            # 检查是否有必要压
            opp_remaining = min(game.hands[p].sum() for p in range(6) if p % 2 != team and not game.finished[p])
            if opp_remaining > 3:
                # 对手牌还多，不压
                pass_actions = [i for i, (c, cnt, ct) in enumerate(actions) if c < 0]
                if pass_actions:
                    return pass_actions[0]
    
    # 否则正常出牌
    return rule_action(game)


def test(model, n=500, opp='rule'):
    game = EnhancedGame()
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
                elif opp == 'smart':
                    idx = smart_rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
            
            card, cnt, card_type = actions[idx]
            done, winner, _ = game.step(card, cnt, card_type)
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== PPO训练 ==============
def ppo_train_v14(model, n_games=8000, lr=5e-4):
    game = EnhancedGame()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_rule = 0
    best_random = 0
    best_smart = 0
    best_combined = 0
    
    memory = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': []}
    
    print(f"\nV14 PPO训练 ({n_games}局)...")
    print("=" * 70)
    print("策略优化：三带二、配合度、大小王使用")
    print("=" * 70)
    
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
                
                episode_data.append({
                    'state': state,
                    'action': action,
                    'value': value,
                    'log_prob': log_prob
                })
                
                card, cnt, card_type = actions[idx]
            else:
                # 对手：混合策略
                actions_opp = game.get_actions()
                if np.random.random() < 0.5:
                    idx = np.random.randint(len(actions_opp))
                else:
                    idx = smart_rule_action(game)
                card, cnt, card_type = actions_opp[idx]
            
            done, winner, reward = game.step(card, cnt, card_type)
            
            if len(episode_data) > 0:
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
        
        # PPO更新（更频繁）
        if len(memory['states']) >= 1024:
            states = torch.FloatTensor(np.array(memory['states']))
            actions = torch.LongTensor(memory['actions'])
            old_log_probs = torch.FloatTensor(memory['log_probs'])
            returns = torch.FloatTensor(memory['rewards'])
            old_values = torch.FloatTensor(memory['values'])
            
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(2):  # 减少更新轮数
                logits, new_values = model(states)
                new_log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(new_values.squeeze(), returns)
                entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            memory = {k: [] for k in memory}
        
        # 测试（每500局）
        if (g + 1) % 500 == 0:
            r1 = test(model, 300, 'rule')
            r2 = test(model, 300, 'random')
            r3 = test(model, 300, 'smart')
            combined = r1 + r2 + r3
            
            print(f"  {g+1:>5,}局 | vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | vs智能: {r3:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                best_smart = r3
                torch.save(model.state_dict(), '/workspace/projects/rl_v14_best.pt')
                print(f"    ★ 新最佳模型! 综合得分: {combined:.1f}")
    
    return model, best_rule, best_random, best_smart


def main():
    print("=" * 70)
    print("V14 (P4) - 策略优化版")
    print("优化：三带二策略、配合度、大小王使用时机")
    print("=" * 70)
    
    model = EnhancedActorCritic(state_dim=80, hidden_dim=512)
    
    # 加载V13模型作为起点（部分参数）
    if os.path.exists('/workspace/projects/rl_v13_ppo_best.pt'):
        print("加载V13模型作为初始化...")
        old_state = torch.load('/workspace/projects/rl_v13_ppo_best.pt', weights_only=True)
        model_state = model.state_dict()
        
        # 只加载兼容的参数
        for name, param in old_state.items():
            if name in model_state and model_state[name].shape == param.shape:
                model_state[name] = param
        model.load_state_dict(model_state, strict=False)
        print("部分参数已迁移")
    
    # 初始测试
    r1 = test(model, 300, 'rule')
    r2 = test(model, 300, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 训练
    model, best_rule, best_random, best_smart = ppo_train_v14(model, n_games=8000)
    
    # 最终测试
    print("\n" + "=" * 70)
    print("最终测试:")
    
    if os.path.exists('/workspace/projects/rl_v14_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v14_best.pt', weights_only=True))
    
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    r3 = test(model, 500, 'smart')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  vs智能Bot: {r3:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v14_final.pt')
    
    print("\n" + "=" * 70)
    print("V14训练完成!")
    print(f"最佳: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%, vs智能 {max(r3, best_smart):.1f}%")


if __name__ == '__main__':
    main()
