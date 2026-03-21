#!/usr/bin/env python3
"""
强化学习训练 V2 - 快速版
优化：批量更新，提高训练速度
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime

# ============== 极简游戏环境 ==============
class FastGame:
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
                    actions.append((i, 1))  # (牌值, 张数)
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
            actions.append((-1, 0))  # 过牌
        
        return actions if actions else [(-1, 0)]
    
    def step(self, card, count):
        """执行动作"""
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
        
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            team = self.current % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return team
        
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
        
        return -1

# ============== 策略网络 ==============
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(90, 64)
        self.fc2 = nn.Linear(64, 15)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# ============== 动作选择 ==============
def model_action(model, state, actions, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    
    with torch.no_grad():
        probs = model(torch.FloatTensor(state))
    
    scores = [probs[card].item() for card, _ in actions]
    return int(np.argmax(scores))

def rule_action(actions):
    valid = [(i, card) for i, (card, _) in enumerate(actions) if card >= 0]
    if valid:
        return min(valid, key=lambda x: x[1])[0]
    return len(actions) - 1

# ============== 训练 ==============
def play_game(model, epsilon=0.15):
    game = FastGame()
    game.reset()
    
    model_states = []
    model_actions = []
    
    while True:
        player = game.current
        actions = game.get_actions()
        state = game.get_state()
        
        if player % 2 == 0:
            idx = model_action(model, state, actions, epsilon)
            model_states.append(state)
            model_actions.append(actions[idx][0])  # 保存牌值
        else:
            idx = rule_action(actions)
        
        card, count = actions[idx]
        result = game.step(card, count)
        
        if result >= 0:
            return model_states, model_actions, result

def train_batch(model, optimizer, games_data):
    """批量训练"""
    total_loss = 0
    
    for states, actions, winner in games_data:
        reward = 1.0 if winner == 0 else -0.5
        
        for state, action in zip(states, actions):
            probs = model(torch.FloatTensor(state))
            prob = probs[action] if action >= 0 else probs.mean()
            
            loss = -torch.log(prob + 1e-8) * reward
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss

def test(model, num=50):
    wins = 0
    for _ in range(num):
        _, _, winner = play_game(model, epsilon=0.0)
        if winner == 0:
            wins += 1
    return wins / num * 100

def main():
    print("=" * 60)
    print("强化学习训练 V2 - 快速版")
    print("目标：对规则Bot胜率 ≥ 60%")
    print("=" * 60)
    
    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 加载检查点
    if os.path.exists('/workspace/projects/rl_model.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_model.pt', map_location='cpu', weights_only=True))
        print("加载模型")
    
    target = 100000
    batch_size = 100
    test_interval = 1000
    save_interval = 5000
    
    best_rate = 0
    start_time = time.time()
    wins = 0
    
    print(f"开始训练: 0 -> {target}")
    print("-" * 60)
    
    batch_data = []
    
    for ep in range(target):
        states, actions, winner = play_game(model)
        batch_data.append((states, actions, winner))
        
        if winner == 0:
            wins += 1
        
        # 批量训练
        if len(batch_data) >= batch_size:
            train_batch(model, optimizer, batch_data)
            batch_data = []
        
        # 测试
        if (ep + 1) % test_interval == 0:
            rate = test(model, 50)
            elapsed = time.time() - start_time
            speed = (ep + 1) / elapsed
            
            train_rate = wins / (ep + 1) * 100
            print(f"[{ep+1}/{target}] 测试胜率: {rate:.1f}% | 训练胜率: {train_rate:.1f}% | 速度: {speed:.0f}局/秒")
            
            if rate > best_rate:
                best_rate = rate
                torch.save(model.state_dict(), '/workspace/projects/rl_best.pt')
                print(f"  ✅ 新最佳: {best_rate:.1f}%")
            
            if rate >= 60:
                print("=" * 60)
                print(f"🎉 达到目标！胜率: {rate:.1f}%")
                break
        
        # 保存
        if (ep + 1) % save_interval == 0:
            torch.save(model.state_dict(), '/workspace/projects/rl_model.pt')
            status = {
                'episode': ep + 1,
                'target': target,
                'progress': (ep + 1) / target * 100,
                'win_rate': wins / (ep + 1) * 100,
                'best_rate': best_rate,
                'training_type': 'reinforcement_learning',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    print("=" * 60)
    print(f"训练完成！最佳胜率: {best_rate:.1f}%")
    torch.save(model.state_dict(), '/workspace/projects/rl_final.pt')

if __name__ == '__main__':
    main()
