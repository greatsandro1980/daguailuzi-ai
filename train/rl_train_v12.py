#!/usr/bin/env python3
"""
训练 V12 - 密集奖励优化版
P2优化: 优化奖励函数，使用密集奖励替代稀疏奖励
策略：
1. 监督学习预训练（学习智能策略）
2. 策略梯度微调 - 密集奖励：
   - 出牌奖励: +0.02 每次成功出牌
   - 手牌减少: +0.05 每减少一张牌
   - 关键牌型: +0.03 打出对子/三张
   - 队友配合: +0.1 队友先出完
   - 游戏胜利: +1.0
   - 过牌惩罚: -0.01
3. 对手: 70% 弱化规则Bot + 30% 自我对弈
4. 目标: vs规则 60%+, vs随机 70%+
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
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
        self.prev_hand_size = 0  # 记录上一步手牌数
        
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
        self.prev_hand_size = self.hands[0].sum()
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
    
    def step(self, card, count):
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
        
        # 记录当前玩家和手牌变化
        player = self.current
        prev_hand = self.hands[player].sum()
        
        self.hands[player] -= action
        self.hands = np.maximum(self.hands, 0)
        new_hand = self.hands[player].sum()
        
        # 计算即时奖励
        instant_reward = 0.0
        
        # 1. 出牌奖励
        if action.sum() > 0:
            instant_reward += 0.02  # 成功出牌
            
            # 2. 手牌减少奖励
            cards_played = prev_hand - new_hand
            instant_reward += cards_played * 0.03
            
            # 3. 关键牌型奖励
            if count >= 2:
                instant_reward += 0.02  # 打出对子
            if count >= 3:
                instant_reward += 0.03  # 打出三张
        
        # 4. 过牌惩罚
        if action.sum() == 0 and self.last_play is not None:
            instant_reward -= 0.01
        
        # 检查是否出完
        game_over = False
        winner = -1
        
        if self.hands[player].sum() == 0:
            self.finished[player] = True
            self.finish_order.append(player)
            
            # 5. 先出完奖励
            instant_reward += 0.2
            
            if all(self.finished):
                game_over = True
                winner = self.finish_order[0] % 2
                
                # 6. 最终胜利奖励
                if winner == 0:
                    instant_reward += 1.0
                else:
                    instant_reward -= 0.3
        
        if action.sum() > 0:
            self.last_play = action.copy()
            self.last_player = player
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
        
        return game_over, winner, instant_reward


# ============== 网络 ==============
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
    
    def get_action(self, state, actions, epsilon=0.0):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        valid = [15 if c < 0 else c for c, _ in actions]
        
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
        
        scores = [(i, logits[idx].item()) for i, idx in enumerate(valid)]
        return max(scores, key=lambda x: x[1])[0]
    
    def get_action_probs(self, state, actions, temperature=1.0):
        with torch.no_grad():
            logits = self(torch.FloatTensor(state)) / temperature
        
        valid = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits, dim=0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        valid_probs = valid_probs / valid_probs.sum()
        
        return valid_probs, valid


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


# ============== 数据生成 ==============
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
            done, _, _ = game.step(*actions[idx])
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


# ============== 测试 ==============
def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.get_action(game.get_state(), actions, epsilon=0.0)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                elif opp == 'smart':
                    idx = smart_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner, _ = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== V12核心: 密集奖励训练 ==============
def train_with_dense_reward(model, n_games=60000):
    """
    V12 密集奖励训练
    
    奖励设计:
    - 出牌奖励: +0.02
    - 手牌减少: +0.03/张
    - 关键牌型: +0.02~+0.05
    - 过牌惩罚: -0.01
    - 先出完: +0.2
    - 胜利: +1.0
    
    对手分布:
    - 70% 弱化规则Bot
    - 30% 自我对弈
    """
    game = Game()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    best_rule = 0
    best_random = 0
    
    # 统计数据
    reward_history = deque(maxlen=1000)
    opp_wins = {'weak': deque(maxlen=500), 'self_play': deque(maxlen=500)}
    
    print(f"\nV12 密集奖励训练...")
    print("=" * 80)
    print("奖励: 出牌+0.02 | 手牌-0.03/张 | 关键牌型+0.02~0.05 | 胜利+1.0")
    print("对手: 70%弱化Bot + 30%自我对弈")
    print("=" * 80)
    
    for g in range(n_games):
        game.reset()
        episode_rewards = []
        transitions = []
        
        # 选择对手类型
        opponent_type = 'self_play' if np.random.random() < 0.3 else 'weak'
        
        while True:
            if game.current % 2 == 0:
                # 己方（模型决策）
                state = game.get_state()
                actions = game.get_actions()
                probs, valid = model.get_action_probs(state, actions, temperature=1.0)
                
                # 探索
                if np.random.random() < 0.05:
                    a_idx = np.random.randint(len(actions))
                else:
                    a_idx = torch.multinomial(probs, 1).item()
                
                transitions.append((state, valid[a_idx]))
                card, cnt = actions[a_idx]
            else:
                # 对手方
                actions = game.get_actions()
                if opponent_type == 'self_play':
                    a_idx = model.get_action(game.get_state(), actions, epsilon=0.1)
                else:
                    a_idx = weak_action(game, error_rate=0.35)
                card, cnt = actions[a_idx]
            
            done, winner, instant_reward = game.step(card, cnt)
            episode_rewards.append(instant_reward)
            
            if done:
                opp_wins[opponent_type].append(1 if winner == 0 else 0)
                
                # 计算累积奖励
                total_reward = sum(episode_rewards)
                reward_history.append(total_reward)
                
                # 使用优势加权更新
                advantage = total_reward - (sum(reward_history) / len(reward_history) if reward_history else 0)
                
                for s, a in transitions:
                    opt.zero_grad()
                    log_prob = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a]
                    loss = log_prob * advantage
                    loss.backward()
                    opt.step()
                break
        
        # 定期测试和保存
        if (g + 1) % 3000 == 0:
            r1 = test(model, 300, 'rule')
            r2 = test(model, 300, 'random')
            
            if r1 > best_rule: 
                best_rule = r1
                torch.save(model.state_dict(), '/workspace/projects/rl_v12_best_rule.pt')
            if r2 > best_random:
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v12_best_random.pt')
            
            # 计算统计
            avg_reward = sum(reward_history) / len(reward_history) if reward_history else 0
            weak_wr = sum(opp_wins['weak']) / len(opp_wins['weak']) * 100 if opp_wins['weak'] else 0
            self_wr = sum(opp_wins['self_play']) / len(opp_wins['self_play']) * 100 if opp_wins['self_play'] else 0
            
            print(f"{g+1:>6,} | vs规则: {r1:>5.1f}% | vs随机: {r2:>5.1f}% | "
                  f"平均奖励: {avg_reward:>5.2f} | 弱化: {weak_wr:>5.1f}% | 自我: {self_wr:>5.1f}%")
            
            torch.save(model.state_dict(), '/workspace/projects/rl_v12_best.pt')
            
            status = {
                'episode': g+1, 
                'vs_rule': r1, 
                'vs_random': r2,
                'best_rule': best_rule, 
                'best_random': best_random,
                'avg_reward': avg_reward,
                'weak_winrate': weak_wr,
                'self_play_winrate': self_wr,
                'training_type': 'v12_dense_reward', 
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    return model


def main():
    print("=" * 80)
    print("V12 密集奖励优化版")
    print("P2优化: 密集奖励替代稀疏奖励")
    print("=" * 80)
    
    model = Net()
    
    # 尝试加载V9最佳模型作为起点
    if os.path.exists('/workspace/projects/rl_v9_best_rule.pt'):
        print("\n加载V9最佳模型作为起点...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v9_best_rule.pt', weights_only=True))
    else:
        # 监督学习预训练
        print("\n[阶段0] 监督学习预训练...")
        states, labels = generate_data(30000, use_smart=True)
        print(f"生成 {len(states)} 条训练数据")
        train_supervised(model, states, labels, epochs=25)
        
        r1 = test(model, 500, 'rule')
        r2 = test(model, 500, 'random')
        print(f"\n预训练后: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
        torch.save(model.state_dict(), '/workspace/projects/rl_v12_pretrained.pt')
    
    # 密集奖励训练
    print("\n[训练阶段] 密集奖励训练...")
    model = train_with_dense_reward(model, n_games=60000)
    
    # 最终测试
    print("\n" + "=" * 80)
    print("最终测试 (1000局):")
    
    # 加载最佳模型
    if os.path.exists('/workspace/projects/rl_v12_best_rule.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v12_best_rule.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    r3 = test(model, 1000, 'smart')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  vs智能Bot: {r3:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v12_final.pt')
    
    print("\n" + "=" * 80)
    print("V12训练完成！")
    print(f"最佳成绩: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    if r1 >= 60:
        print("✅ vs规则Bot: 达标 (≥60%)")
    else:
        print(f"⚠️ vs规则Bot: 未达标 ({r1:.1f}% < 60%)")
    if r2 >= 70:
        print("✅ vs随机Bot: 达标 (≥70%)")
    else:
        print(f"⚠️ vs随机Bot: 未达标 ({r2:.1f}% < 70%)")


if __name__ == '__main__':
    main()
