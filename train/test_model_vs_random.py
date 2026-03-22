#!/usr/bin/env python3
"""
模型 vs 随机Bot PK测试
红队：训练好的神经网络模型
蓝队：随机策略
"""

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

# ============== 游戏环境 ==============
class FastGame:
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
        s = np.zeros(90, dtype=np.float32)
        s[:15] = self.hands[self.current] / 4.0
        if self.last_play is not None:
            s[75:90] = self.last_play / 4.0
        return s
    
    def get_actions(self):
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
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
            last_val = int(self.last_play.max())
            last_cnt = int(self.last_play[self.last_play > 0][0])
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt:
                    actions.append((i, last_cnt))
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
            team = self.current % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return True, team
        
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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(90, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 15)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


# ============== 动作选择 ==============
def model_action(model, state, actions):
    """模型决策"""
    with torch.no_grad():
        probs = model(torch.FloatTensor(state))
    
    scores = []
    for i, (card, count) in enumerate(actions):
        if card >= 0:
            scores.append((i, probs[card].item() * count))
        else:
            scores.append((i, 0.01))
    
    return max(scores, key=lambda x: x[1])[0]


def random_action(actions):
    """随机策略"""
    return np.random.randint(len(actions))


def rule_action(actions):
    """规则Bot：小牌优先"""
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    if valid:
        valid.sort(key=lambda x: (x[1], -x[2]))
        return valid[0][0]
    return len(actions) - 1


# ============== PK测试 ==============
def play_game(model, blue_strategy='random'):
    """进行一局游戏
    blue_strategy: 'random' 随机, 'rule' 规则Bot
    """
    game = FastGame()
    game.reset()
    
    while True:
        player = game.current
        actions = game.get_actions()
        state = game.get_state()
        
        if player % 2 == 0:  # 红队：模型
            idx = model_action(model, state, actions)
        else:  # 蓝队：指定策略
            if blue_strategy == 'random':
                idx = random_action(actions)
            else:
                idx = rule_action(actions)
        
        card, count = actions[idx]
        done, winner = game.step(card, count)
        
        if done:
            return winner  # 0=红队胜, 1=蓝队胜


def pk_test(model_path='/workspace/projects/rl_best.pt', num_games=1000):
    """PK测试"""
    print("=" * 70)
    print("🎮 模型 vs 随机Bot PK测试")
    print("=" * 70)
    
    # 加载模型
    model = PolicyNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"✅ 加载模型: {model_path}")
    
    # 测试1：模型 vs 随机Bot
    print(f"\n📊 测试1: 模型 vs 随机Bot ({num_games}局)")
    print("-" * 50)
    
    red_wins = 0
    for i in range(num_games):
        winner = play_game(model, 'random')
        if winner == 0:
            red_wins += 1
        
        if (i + 1) % 200 == 0:
            rate = red_wins / (i + 1) * 100
            print(f"  {i+1:>5}局 | 红队(模型): {red_wins}胜 | 胜率: {rate:.1f}%")
    
    random_rate = red_wins / num_games * 100
    print(f"\n✅ 模型 vs 随机Bot 最终胜率: {random_rate:.1f}%")
    
    # 测试2：模型 vs 规则Bot
    print(f"\n📊 测试2: 模型 vs 规则Bot ({num_games}局)")
    print("-" * 50)
    
    red_wins2 = 0
    for i in range(num_games):
        winner = play_game(model, 'rule')
        if winner == 0:
            red_wins2 += 1
        
        if (i + 1) % 200 == 0:
            rate = red_wins2 / (i + 1) * 100
            print(f"  {i+1:>5}局 | 红队(模型): {red_wins2}胜 | 胜率: {rate:.1f}%")
    
    rule_rate = red_wins2 / num_games * 100
    print(f"\n✅ 模型 vs 规则Bot 最终胜率: {rule_rate:.1f}%")
    
    # 测试3：随机Bot vs 规则Bot（对照组）
    print(f"\n📊 测试3: 随机Bot vs 规则Bot ({num_games}局) [对照组]")
    print("-" * 50)
    
    # 临时游戏函数
    def play_random_vs_rule():
        game = FastGame()
        game.reset()
        while True:
            player = game.current
            actions = game.get_actions()
            
            if player % 2 == 0:  # 红队：随机
                idx = random_action(actions)
            else:  # 蓝队：规则
                idx = rule_action(actions)
            
            card, count = actions[idx]
            done, winner = game.step(card, count)
            if done:
                return winner
    
    red_wins3 = 0
    for i in range(num_games):
        winner = play_random_vs_rule()
        if winner == 0:
            red_wins3 += 1
    
    control_rate = red_wins3 / num_games * 100
    print(f"✅ 随机Bot vs 规则Bot 胜率: {control_rate:.1f}%")
    
    # 汇总
    print("\n" + "=" * 70)
    print("📊 PK测试汇总")
    print("=" * 70)
    print(f"{'对战':<25} {'红队胜率':>10} {'评价':>15}")
    print("-" * 70)
    print(f"{'模型 vs 随机Bot':<25} {random_rate:>9.1f}% {'🎉 强大' if random_rate > 70 else '✅ 正常' if random_rate > 50 else '⚠️ 需改进'}")
    print(f"{'模型 vs 规则Bot':<25} {rule_rate:>9.1f}% {'🎉 达标' if rule_rate >= 60 else '⚠️ 未达标'}")
    print(f"{'随机Bot vs 规则Bot':<25} {control_rate:>9.1f}% (对照组)")
    print("=" * 70)
    
    # 保存结果
    result = {
        'timestamp': datetime.now().isoformat(),
        'num_games': num_games,
        'model_path': model_path,
        'model_vs_random': round(random_rate, 1),
        'model_vs_rule': round(rule_rate, 1),
        'random_vs_rule': round(control_rate, 1)
    }
    
    with open('/workspace/projects/pk_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: /workspace/projects/pk_result.json")
    
    return result


if __name__ == '__main__':
    pk_test(num_games=1000)
