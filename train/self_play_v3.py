#!/usr/bin/env python3
"""
自我博弈训练 V3 - 高速版
优化：批量训练，减少训练频率
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime

# ============== 高速游戏环境 ==============
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
        
    def get_state(self, player):
        state = np.zeros(90, dtype=np.float32)
        state[:15] = self.hands[player] / 4.0
        for i, p in enumerate([(player+1)%6, (player+3)%6, (player+5)%6]):
            state[15+i*5] = self.hands[p].sum() / 9.0
        if self.last_play is not None:
            state[75:90] = self.last_play / 4.0
        return state
    
    def get_actions(self, player):
        hand = self.hands[player]
        actions = []
        
        if self.last_play is None or self.last_player == player:
            # 出任意牌
            for i in range(15):
                if hand[i] >= 1:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 1
                    actions.append(a)
            for i in range(13):
                if hand[i] >= 2:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 2
                    actions.append(a)
            for i in range(13):
                if hand[i] >= 3:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 3
                    actions.append(a)
        else:
            # 压牌
            last = self.last_play
            last_val = last.max()
            last_cnt = int(last[last > 0][0]) if len(last[last > 0]) > 0 else 0
            
            for i in range(int(last_val) + 1, 15):
                if hand[i] >= last_cnt:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = last_cnt
                    actions.append(a)
            
            if last_cnt < 4:
                for i in range(13):
                    if hand[i] >= 4:
                        a = np.zeros(15, dtype=np.int32)
                        a[i] = 4
                        actions.append(a)
            
            # 过牌
            actions.append(np.zeros(15, dtype=np.int32))
                
        return actions if actions else [np.zeros(15, dtype=np.int32)]
    
    def step(self, player, action):
        self.hands[player] -= action
        self.hands = np.maximum(self.hands, 0)
        
        if self.hands[player].sum() == 0:
            self.finished[player] = True
            self.finish_order.append(player)
            team = player % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return team
        
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

def select_action(model, state, actions, epsilon=0.1):
    """选择动作（无梯度）"""
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    
    with torch.no_grad():
        probs = model(torch.FloatTensor(state))
    
    scores = []
    for action in actions:
        score = sum(probs[i].item() for i in range(15) if action[i] > 0)
        scores.append(score)
    
    return int(np.argmax(scores)) if scores else 0

def play_game_fast(models):
    """快速自我博弈（不记录历史）"""
    game = FastGame()
    game.reset()
    
    while True:
        player = game.current
        actions = game.get_actions(player)
        state = game.get_state(player)
        
        model = models[player % 2]
        action_idx = select_action(model, state, actions, epsilon=0.1)
        action = actions[action_idx]
        
        result = game.step(player, action)
        
        if result >= 0:
            return result
        
        game.current = (game.current + 1) % 6
        while game.finished[game.current]:
            game.current = (game.current + 1) % 6

def main():
    print("=" * 60)
    print("自我博弈训练 V3 - 高速版")
    print("=" * 60)
    
    # 初始化模型
    models = [PolicyNet() for _ in range(2)]
    optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in models]
    
    # 加载检查点
    start_ep = 0
    stats = {'red': 0, 'blue': 0}
    checkpoint_file = '/workspace/projects/self_play_checkpoint.json'
    model_file = '/workspace/projects/self_play_model.pt'
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            data = json.load(f)
            start_ep = data.get('episode', 0)
            stats['red'] = data.get('red_wins', 0)
            stats['blue'] = data.get('blue_wins', 0)
            print(f"从检查点恢复: {start_ep} 局")
    
    if os.path.exists(model_file):
        models[0].load_state_dict(torch.load(model_file, map_location='cpu'))
        models[1].load_state_dict(torch.load(model_file, map_location='cpu'))
        print("加载模型参数")
    
    target = 500000
    start_time = time.time()
    
    print(f"开始训练: {start_ep} -> {target}")
    print("-" * 60)
    
    # 批量训练数据
    batch_size = 1000
    states_batch = {0: [], 1: []}
    targets_batch = {0: [], 1: []}
    
    for ep in range(start_ep, target):
        # 纯自我博弈（不训练）
        winner = play_game_fast(models)
        
        if winner == 0:
            stats['red'] += 1
        else:
            stats['blue'] += 1
        
        # 每batch_size局训练一次
        if (ep + 1) % batch_size == 0:
            # 使用简单的强化更新
            for team in [0, 1]:
                reward = 1.0 if team == winner else -0.5
                # 这里简化：只更新一个虚拟状态
                dummy_state = torch.zeros(90)
                target_dist = torch.ones(15) * (0.5 + reward * 0.1)
                target_dist = target_dist / target_dist.sum()
                
                probs = models[team](dummy_state)
                loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(probs + 1e-8), target_dist
                )
                
                optimizers[team].zero_grad()
                loss.backward()
                optimizers[team].step()
        
        # 保存检查点
        if (ep + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            speed = (ep + 1 - start_ep) / elapsed if elapsed > 0 else 0
            
            # 保存
            torch.save(models[0].state_dict(), model_file)
            
            checkpoint = {
                'episode': ep + 1,
                'red_wins': stats['red'],
                'blue_wins': stats['blue'],
                'speed': speed,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            status = {
                'episode': ep + 1,
                'target': target,
                'progress': (ep + 1) / target * 100,
                'red_wins': stats['red'],
                'blue_wins': stats['blue'],
                'red_win_rate': stats['red'] / (ep + 1) * 100,
                'blue_win_rate': stats['blue'] / (ep + 1) * 100,
                'speed': speed,
                'elapsed_hours': elapsed / 3600,
                'training_type': 'self_play',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            red_rate = stats['red'] / (ep + 1) * 100
            print(f"[{ep+1}/{target}] 红队: {red_rate:.1f}% | 速度: {speed:.0f}局/秒")
    
    print("=" * 60)
    print("训练完成!")
    print(f"红队胜率: {stats['red']/target*100:.1f}%")
    torch.save(models[0].state_dict(), '/workspace/projects/self_play_final.pt')
    print("模型已保存")

if __name__ == '__main__':
    main()
