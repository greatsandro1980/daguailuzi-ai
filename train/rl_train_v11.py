#!/usr/bin/env python3
"""
训练 V11 - 多对手混合训练版
优化: 降低自我对弈比例，增加规则Bot对抗
策略：
1. 监督学习预训练（学习智能策略）
2. 策略梯度微调 - 多对手混合：
   - 30% 自我对弈 (Self-Play)
   - 40% 弱化规则Bot (Weak Rule Bot, error_rate=0.35)
   - 20% 中等规则Bot (Medium Rule Bot, error_rate=0.15)
   - 10% 随机Bot (Random Bot)
3. 课程学习: 前20k局只用弱化对手，之后逐步增加难度
4. 目标: vs规则 62-65%, vs随机 70%+
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
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== V11核心: 多对手混合训练 ==============
def train_mixed_opponents(model, n_games=60000):
    """
    V11 多对手混合训练
    
    对手分布:
    - 阶段1 (0-20k): 100% 弱化规则Bot (error_rate=0.35)
    - 阶段2 (20k-40k): 30%自我对弈 + 50%弱化Bot + 20%随机Bot
    - 阶段3 (40k-60k): 30%自我对弈 + 30%弱化Bot + 30%中等Bot + 10%随机Bot
    """
    game = Game()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    best_rule = 0
    best_random = 0
    
    # 统计数据
    opponent_wins = {'self_play': deque(maxlen=500), 
                     'weak': deque(maxlen=500), 
                     'medium': deque(maxlen=500), 
                     'random': deque(maxlen=500)}
    
    print(f"\nV11 多对手混合训练...")
    print("=" * 80)
    print("阶段1: 纯弱化对手 | 阶段2: 混合训练 | 阶段3: 增加难度")
    print("=" * 80)
    
    for g in range(n_games):
        game.reset()
        transitions = []
        
        # 根据训练阶段选择对手分布
        if g < 20000:
            # 阶段1: 只用弱化对手
            opponent_type = 'weak'
        elif g < 40000:
            # 阶段2: 混合训练
            r = np.random.random()
            if r < 0.30:
                opponent_type = 'self_play'
            elif r < 0.80:
                opponent_type = 'weak'
            else:
                opponent_type = 'random'
        else:
            # 阶段3: 增加难度
            r = np.random.random()
            if r < 0.30:
                opponent_type = 'self_play'
            elif r < 0.60:
                opponent_type = 'weak'
            elif r < 0.90:
                opponent_type = 'medium'
            else:
                opponent_type = 'random'
        
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
                
                transitions.append((state, valid[a_idx], opponent_type))
                card, cnt = actions[a_idx]
            else:
                # 对手方
                actions = game.get_actions()
                if opponent_type == 'self_play':
                    a_idx = model.get_action(game.get_state(), actions, epsilon=0.1)
                elif opponent_type == 'weak':
                    a_idx = weak_action(game, error_rate=0.35)
                elif opponent_type == 'medium':
                    a_idx = weak_action(game, error_rate=0.15)
                else:  # random
                    a_idx = np.random.randint(len(actions))
                card, cnt = actions[a_idx]
            
            done, winner = game.step(card, cnt)
            if done:
                # 计算奖励
                reward = 1.2 if winner == 0 else -0.25
                opponent_wins[opponent_type].append(1 if winner == 0 else 0)
                
                # 更新模型
                for s, a, opp_type in transitions:
                    opt.zero_grad()
                    log_prob = -torch.log_softmax(model(torch.FloatTensor(s)), 0)[a]
                    loss = log_prob * reward
                    loss.backward()
                    opt.step()
                break
        
        # 定期测试和保存
        if (g + 1) % 3000 == 0:
            r1 = test(model, 300, 'rule')
            r2 = test(model, 300, 'random')
            
            if r1 > best_rule: 
                best_rule = r1
                torch.save(model.state_dict(), '/workspace/projects/rl_v11_best_rule.pt')
            if r2 > best_random:
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v11_best_random.pt')
            
            # 计算各对手胜率
            win_rates = {}
            for k, v in opponent_wins.items():
                if v:
                    win_rates[k] = sum(v) / len(v) * 100
            
            # 判断阶段
            phase = 1 if g < 20000 else (2 if g < 40000 else 3)
            
            print(f"{g+1:>6,} [阶段{phase}] | vs规则: {r1:>5.1f}% | vs随机: {r2:>5.1f}% | "
                  f"自我: {win_rates.get('self_play', 0):>5.1f}% | "
                  f"弱化: {win_rates.get('weak', 0):>5.1f}% | "
                  f"中等: {win_rates.get('medium', 0):>5.1f}%")
            
            # 保存最佳模型
            torch.save(model.state_dict(), '/workspace/projects/rl_v11_best.pt')
            
            status = {
                'episode': g+1, 
                'vs_rule': r1, 
                'vs_random': r2,
                'best_rule': best_rule, 
                'best_random': best_random,
                'phase': phase,
                'win_rates': win_rates,
                'training_type': 'v11_mixed', 
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    return model


def main():
    print("=" * 80)
    print("V11 多对手混合训练版")
    print("优化: 30%自我对弈 + 60%规则Bot + 10%随机Bot")
    print("课程学习: 阶段递进，逐步增加难度")
    print("=" * 80)
    
    model = Net()
    
    # 尝试加载V9最佳模型作为起点
    if os.path.exists('/workspace/projects/rl_v9_best_rule.pt'):
        print("\n加载V9最佳模型作为起点...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v9_best_rule.pt', weights_only=True))
    elif os.path.exists('/workspace/projects/rl_v10_best_rule.pt'):
        print("\n加载V10最佳模型作为起点...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v10_best_rule.pt', weights_only=True))
    else:
        # 监督学习预训练
        print("\n[阶段0] 监督学习预训练...")
        states, labels = generate_data(30000, use_smart=True)
        print(f"生成 {len(states)} 条训练数据")
        train_supervised(model, states, labels, epochs=25)
        
        r1 = test(model, 500, 'rule')
        r2 = test(model, 500, 'random')
        print(f"\n预训练后: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
        torch.save(model.state_dict(), '/workspace/projects/rl_v11_pretrained.pt')
    
    # 多对手混合训练
    print("\n[训练阶段] 多对手混合训练...")
    model = train_mixed_opponents(model, n_games=60000)
    
    # 最终测试
    print("\n" + "=" * 80)
    print("最终测试 (1000局):")
    
    # 加载最佳模型
    if os.path.exists('/workspace/projects/rl_v11_best_rule.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v11_best_rule.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    r3 = test(model, 1000, 'smart')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  vs智能Bot: {r3:.1f}%")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v11_final.pt')
    
    print("\n" + "=" * 80)
    print("V11训练完成！")
    print(f"最佳成绩: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    if r1 >= 62:
        print("✅ vs规则Bot: 达标 (≥62%)")
    else:
        print(f"⚠️ vs规则Bot: 未达标 ({r1:.1f}% < 62%)")
    if r2 >= 68:
        print("✅ vs随机Bot: 达标 (≥68%)")
    else:
        print(f"⚠️ vs随机Bot: 未达标 ({r2:.1f}% < 68%)")


if __name__ == '__main__':
    main()
