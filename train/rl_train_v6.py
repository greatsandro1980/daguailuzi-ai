#!/usr/bin/env python3
"""
训练 V6 - 监督学习预训练 + 自我博弈提升
策略：
1. 先用监督学习让模型学会规则Bot的策略
2. 再用自我博弈提升实力
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import deque

# ============== 游戏环境（简化版）==============
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
        """简化的状态表示 (60维)"""
        s = np.zeros(60, dtype=np.float32)
        
        # 自己手牌 (15)
        s[:15] = self.hands[self.current]
        
        # 上一轮出牌 (15)
        if self.last_play is not None:
            s[15:30] = self.last_play
        
        # 所有玩家剩余牌数 (6)
        for i in range(6):
            s[30+i] = self.hands[i].sum()
        
        # 当前出牌类型 (3): 是否需要跟单张/对子/三张
        if self.last_play is not None and self.last_play.sum() > 0:
            cnt = int(self.last_play[self.last_play > 0][0])
            if cnt == 1:
                s[36] = 1.0
            elif cnt == 2:
                s[37] = 1.0
            else:
                s[38] = 1.0
        
        # 是否首发 (1)
        s[39] = 1.0 if self.last_play is None else 0.0
        
        # 队友是否已出完 (3)
        team = self.current % 2
        for i, mate in enumerate([p for p in range(6) if p % 2 == team and p != self.current]):
            s[40+i] = 1.0 if self.finished[mate] else 0.0
        
        # 对手是否已出完 (3)
        for i, opp in enumerate([p for p in range(6) if p % 2 != team]):
            s[43+i] = 1.0 if self.finished[opp] else 0.0
        
        # 自己手牌统计 (10)
        hand = self.hands[self.current]
        s[46] = sum(1 for i in range(13) if hand[i] == 1)  # 单张数
        s[47] = sum(1 for i in range(13) if hand[i] == 2)  # 对子数
        s[48] = sum(1 for i in range(13) if hand[i] >= 3)  # 三张数
        s[49] = hand[13]  # 小王
        s[50] = hand[14]  # 大王
        s[51] = hand.sum()  # 总牌数
        s[52] = 1.0 if hand[13] > 0 or hand[14] > 0 else 0.0  # 是否有王
        s[53] = min(hand[:13].max(), 3) if hand[:13].max() > 0 else 0  # 最多同值牌数
        s[54] = np.argmax(hand[:13]) if hand[:13].sum() > 0 else 0  # 最大牌值
        s[55] = np.argmin(hand[:13] + (1 - (hand[:13] > 0)) * 100)  # 最小牌值
        
        # 已出完玩家数 (4)
        s[56] = sum(self.finished)
        s[57] = sum(1 for p in range(6) if p % 2 == 0 and self.finished[p])  # 红队出完数
        s[58] = sum(1 for p in range(6) if p % 2 == 1 and self.finished[p])  # 蓝队出完数
        
        return s
    
    def get_actions(self):
        """获取合法动作"""
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
            # 首发
            for i in range(15):
                if hand[i] >= 1:
                    actions.append((i, 1))
            for i in range(13):
                if hand[i] >= 2:
                    actions.append((i, 2))
            for i in range(13):
                if hand[i] >= 3:
                    actions.append((i, 3))
        else:
            # 跟牌
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt:
                    actions.append((i, last_cnt))
            actions.append((-1, 0))  # 过牌
        
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


# ============== 策略网络 ==============
class PolicyNet(nn.Module):
    def __init__(self, state_dim=60, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16)  # 输出16类：15种牌 + 过牌
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, actions):
        """选择动作"""
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
        
        # 只考虑合法动作
        valid_indices = []
        for card, count in actions:
            if card < 0:
                valid_indices.append(15)  # 过牌
            else:
                valid_indices.append(card)  # 牌值就是索引
        
        # 从合法动作中选择logit最大的
        valid_logits = [(i, logits[idx].item()) for i, idx in enumerate(valid_indices)]
        best_idx = max(valid_logits, key=lambda x: x[1])[0]
        return best_idx


# ============== 规则Bot策略 ==============
def rule_bot_action(game):
    """规则Bot决策"""
    actions = game.get_actions()
    hand = game.hands[game.current]
    team = game.current % 2
    
    # 检查队友状态
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    
    if not valid:
        return len(actions) - 1  # 过牌
    
    # 策略：优先出对子和三张（消耗更多牌）
    if mates_done >= 2:
        # 队友都快出完了，激进出大牌
        valid.sort(key=lambda x: -x[1])
    else:
        # 优先对子三张
        pairs = [(i, card, count) for i, card, count in valid if count >= 2]
        if pairs:
            pairs.sort(key=lambda x: x[1])
            for i, card, count in pairs:
                if i < len(actions):
                    return i
        valid.sort(key=lambda x: x[1])
    
    return valid[0][0] if valid else len(actions) - 1


def strong_rule_bot_action(game):
    """更强的规则Bot"""
    actions = game.get_actions()
    hand = game.hands[game.current]
    team = game.current % 2
    
    # 分析局面
    opps_remaining = [game.hands[p].sum() for p in range(6) if p % 2 != team and not game.finished[p]]
    mates_remaining = [game.hands[p].sum() for p in range(6) if p % 2 == team and not game.finished[p] and p != game.current]
    
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    
    if not valid:
        return len(actions) - 1
    
    # 如果队友牌少，帮队友控场（出小牌）
    if mates_remaining and min(mates_remaining) <= 3:
        # 先出小牌
        valid.sort(key=lambda x: x[1])
        return valid[0][0]
    
    # 如果对手牌少，封锁对手（出大牌）
    if opps_remaining and min(opps_remaining) <= 3:
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    
    # 默认：优先对子三张，小牌优先
    pairs = [(i, card, count) for i, card, count in valid if count >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


# ============== 监督学习 ==============
def generate_training_data(num_games=10000):
    """生成训练数据"""
    game = SimpleGame()
    states = []
    labels = []
    
    for _ in range(num_games):
        game.reset()
        while True:
            player = game.current
            state = game.get_state()
            actions = game.get_actions()
            
            # 规则Bot决策
            idx = rule_bot_action(game)
            
            # 存储
            states.append(state)
            if actions[idx][0] < 0:
                labels.append(15)  # 过牌
            else:
                labels.append(actions[idx][0])  # 牌值
            
            # 执行
            card, count = actions[idx]
            done, _ = game.step(card, count)
            if done:
                break
    
    return np.array(states), np.array(labels)


def train_supervised(model, states, labels, epochs=20, batch_size=256, lr=1e-3):
    """监督学习训练"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    n = len(states)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0
        
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_states = torch.FloatTensor(states[batch_idx])
            batch_labels = torch.LongTensor(labels[batch_idx])
            
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(n//batch_size):.4f}")
    
    return model


# ============== 自我博弈训练 ==============
def play_game(model, opponent='rule'):
    """进行一局游戏"""
    game = SimpleGame()
    game.reset()
    
    while True:
        player = game.current
        state = game.get_state()
        actions = game.get_actions()
        
        if player % 2 == 0:  # 红队：模型
            idx = model.get_action(state, actions)
        else:  # 蓝队：对手
            if opponent == 'rule':
                idx = rule_bot_action(game)
            elif opponent == 'strong':
                idx = strong_rule_bot_action(game)
            else:
                idx = np.random.randint(len(actions))
        
        card, count = actions[idx]
        done, winner = game.step(card, count)
        
        if done:
            return winner, game.finish_order


def test_model(model, num=100, opponent='rule'):
    """测试模型胜率"""
    wins = 0
    scores = 0
    
    for _ in range(num):
        winner, finish_order = play_game(model, opponent)
        if winner == 0:
            wins += 1
            if finish_order and finish_order[-1] % 2 == 1:
                scores += 1
    
    return wins / num * 100, scores / num * 100


def self_play_train(model, num_games=50000, lr=1e-4):
    """自我博弈训练（简单策略梯度）"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    game = SimpleGame()
    win_history = deque(maxlen=100)
    best_rate = 0
    
    print("\n自我博弈训练...")
    print("-" * 60)
    
    for game_idx in range(num_games):
        game.reset()
        transitions = []  # (state, action, reward)
        
        while True:
            player = game.current
            state = game.get_state()
            actions = game.get_actions()
            
            if player % 2 == 0:  # 红队：模型
                # 记录状态和动作
                logits = model(torch.FloatTensor(state))
                
                # 计算合法动作的概率
                valid_indices = []
                for card, count in actions:
                    if card < 0:
                        valid_indices.append(15)
                    else:
                        valid_indices.append(card)
                
                probs = torch.softmax(logits[valid_indices], dim=0)
                action_idx = torch.multinomial(probs, 1).item()
                actual_action = valid_indices[action_idx]
                
                transitions.append((state.copy(), actual_action, 0))
                
                # 找到对应的动作
                for i, (c, cnt) in enumerate(actions):
                    if (c < 0 and actual_action == 15) or (c == actual_action):
                        card, count = c, cnt
                        break
            else:  # 蓝队：规则Bot
                idx = rule_bot_action(game)
                card, count = actions[idx]
            
            done, winner = game.step(card, count)
            if done:
                # 计算奖励
                reward = 1.0 if winner == 0 else -0.3
                if winner == 0 and game.finish_order and game.finish_order[-1] % 2 == 1:
                    reward += 0.5  # 得分奖励
                
                # 更新转移记录的奖励
                for i in range(len(transitions)):
                    s, a, _ = transitions[i]
                    transitions[i] = (s, a, reward)
                break
        
        win_history.append(1 if winner == 0 else 0)
        
        # 策略梯度更新
        if winner == 0:  # 只在获胜时更新
            for state, action, reward in transitions:
                logits = model(torch.FloatTensor(state))
                log_prob = -torch.log_softmax(logits, dim=0)[action]
                loss = log_prob * reward
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 测试
        if (game_idx + 1) % 1000 == 0:
            rate_rule, score_rule = test_model(model, 100, 'rule')
            rate_random, score_random = test_model(model, 100, 'random')
            
            if rate_rule > best_rate:
                best_rate = rate_rule
                torch.save(model.state_dict(), '/workspace/projects/rl_v6_best.pt')
            
            train_rate = sum(win_history) / len(win_history) * 100
            print(f"{game_idx+1:>6,} | vs规则: {rate_rule:>5.1f}% | vs随机: {rate_random:>5.1f}% | "
                  f"训练胜率: {train_rate:>5.1f}% | 最佳: {best_rate:>5.1f}%")
            
            # 保存状态
            status = {
                'episode': game_idx + 1,
                'target': num_games,
                'progress': (game_idx + 1) / num_games * 100,
                'vs_rule': round(rate_rule, 1),
                'vs_random': round(rate_random, 1),
                'best_rate': round(best_rate, 1),
                'training_type': 'supervised_self_play_v6',
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    return model


def main():
    print("=" * 60)
    print("训练 V6 - 监督学习预训练 + 自我博弈提升")
    print("=" * 60)
    
    model = PolicyNet()
    
    # 第一阶段：监督学习预训练
    print("\n[阶段1] 生成训练数据...")
    states, labels = generate_training_data(20000)
    print(f"生成 {len(states)} 条训练数据")
    
    print("\n[阶段1] 监督学习预训练...")
    model = train_supervised(model, states, labels, epochs=30, lr=1e-3)
    
    # 测试预训练效果
    print("\n预训练后测试:")
    rate_rule, _ = test_model(model, 100, 'rule')
    rate_random, _ = test_model(model, 100, 'random')
    print(f"vs规则Bot: {rate_rule:.1f}%, vs随机Bot: {rate_random:.1f}%")
    
    # 保存预训练模型
    torch.save(model.state_dict(), '/workspace/projects/rl_v6_pretrained.pt')
    
    # 第二阶段：自我博弈提升
    print("\n[阶段2] 自我博弈训练...")
    model = self_play_train(model, num_games=50000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试:")
    rate_rule, score_rule = test_model(model, 200, 'rule')
    rate_random, score_random = test_model(model, 200, 'random')
    rate_strong, _ = test_model(model, 200, 'strong')
    print(f"vs规则Bot: {rate_rule:.1f}% (得分率: {score_rule:.1f}%)")
    print(f"vs随机Bot: {rate_random:.1f}% (得分率: {score_random:.1f}%)")
    print(f"vs强规则Bot: {rate_strong:.1f}%")
    
    # 保存最终模型
    torch.save(model.state_dict(), '/workspace/projects/rl_v6_final.pt')
    print("\n模型已保存!")


if __name__ == '__main__':
    main()
