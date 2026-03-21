#!/usr/bin/env python3
"""
自我博弈训练 V2 - 简化版
目标：50万局，通过自我博弈突破50%胜率
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
    """极简游戏环境"""
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        
    def reset(self):
        # 创建牌堆
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
        """获取玩家视角的状态"""
        state = np.zeros(90, dtype=np.float32)
        state[:15] = self.hands[player] / 4.0
        # 上家、对家、下家的牌数
        for i, p in enumerate([(player+1)%6, (player+3)%6, (player+5)%6]):
            state[15+i*5:20+i*5] = [self.hands[p].sum()/9.0] + [0,0,0,0]
        if self.last_play is not None:
            state[75:90] = self.last_play / 4.0
        return state
    
    def get_actions(self, player):
        """获取合法动作"""
        hand = self.hands[player]
        actions = []
        
        # 必须出牌的情况
        if self.last_play is None or self.last_player == player:
            # 单牌
            for i in range(15):
                if hand[i] >= 1:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 1
                    actions.append(a)
            # 对子
            for i in range(13):
                if hand[i] >= 2:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 2
                    actions.append(a)
            # 三张
            for i in range(13):
                if hand[i] >= 3:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 3
                    actions.append(a)
            # 炸弹
            for i in range(13):
                if hand[i] >= 4:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 4
                    actions.append(a)
        else:
            # 压牌
            last = self.last_play
            last_val = last.max()
            last_cnt = int(last[last > 0][0]) if len(last[last > 0]) > 0 else 0
            
            # 同类型压
            for i in range(int(last_val) + 1, 15):
                if hand[i] >= last_cnt:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = last_cnt
                    actions.append(a)
            
            # 炸弹压
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
        """执行动作"""
        self.hands[player] -= action
        self.hands = np.maximum(self.hands, 0)
        
        # 检查是否出完
        if self.hands[player].sum() == 0:
            self.finished[player] = True
            self.finish_order.append(player)
            team = player % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return team  # 该队获胜
        
        # 更新
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
            
        return -1  # 继续

# ============== 简化神经网络 ==============
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(90, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 15),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, state, actions, epsilon=0.1):
        """选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
            
        with torch.no_grad():
            probs = self.forward(torch.FloatTensor(state))
        
        # 计算每个动作的得分
        scores = []
        for action in actions:
            score = sum(probs[i].item() for i in range(15) if action[i] > 0)
            scores.append(score)
        
        return int(np.argmax(scores)) if scores else 0

# ============== 训练 ==============
def play_game(models):
    """自我博弈一局"""
    game = FastGame()
    game.reset()
    
    history = {i: [] for i in range(6)}
    
    while True:
        player = game.current
        actions = game.get_actions(player)
        state = game.get_state(player)
        
        model = models[player % 2]
        action_idx = model.get_action(state, actions, epsilon=0.15)
        action = actions[action_idx]
        
        history[player].append((state.copy(), action_idx, actions.copy()))
        
        result = game.step(player, action)
        
        if result >= 0:
            winner_team = result
            break
            
        # 下一个玩家
        game.current = (game.current + 1) % 6
        while game.finished[game.current]:
            game.current = (game.current + 1) % 6
    
    return history, winner_team

def train_batch(models, optimizers, history, winner_team):
    """训练一批数据"""
    losses = []
    
    for player in range(6):
        if not history[player]:
            continue
            
        model = models[player % 2]
        optimizer = optimizers[player % 2]
        
        # 胜利队伍奖励
        team = player % 2
        reward = 1.0 if team == winner_team else -0.5
        
        for state, action_idx, actions in history[player]:
            # 计算目标：增强胜利动作
            target = torch.zeros(15)
            if reward > 0:
                # 胜利时，增强选中的动作
                action = actions[action_idx]
                for i in range(15):
                    if action[i] > 0:
                        target[i] = 0.3
            else:
                # 失败时，减弱
                target = torch.zeros(15)
            
            # 前向
            probs = model(torch.FloatTensor(state))
            
            # 损失
            loss = nn.MSELoss()(probs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
    
    return np.mean(losses) if losses else 0

def main():
    print("=" * 60)
    print("自我博弈训练 V2 - 目标50万局")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # 初始化模型
    models = [SimpleNet().to(device) for _ in range(2)]
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
        models[0].load_state_dict(torch.load(model_file, map_location=device))
        models[1].load_state_dict(torch.load(model_file, map_location=device))
        print("加载模型参数")
    
    target = 500000
    start_time = time.time()
    
    print(f"开始训练: {start_ep} -> {target}")
    print("-" * 60)
    
    for ep in range(start_ep, target):
        # 自我博弈
        history, winner = play_game(models)
        
        # 统计
        if winner == 0:
            stats['red'] += 1
        else:
            stats['blue'] += 1
        
        # 训练
        loss = train_batch(models, optimizers, history, winner)
        
        # 保存检查点
        if (ep + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            speed = (ep + 1 - start_ep) / elapsed if elapsed > 0 else 0
            
            # 保存模型
            torch.save(models[0].state_dict(), model_file)
            
            # 保存检查点
            checkpoint = {
                'episode': ep + 1,
                'red_wins': stats['red'],
                'blue_wins': stats['blue'],
                'speed': speed,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # 更新状态
            status = {
                'episode': ep + 1,
                'target': target,
                'progress': (ep + 1) / target * 100,
                'red_wins': stats['red'],
                'blue_wins': stats['blue'],
                'red_win_rate': stats['red'] / (ep + 1) * 100,
                'blue_win_rate': stats['blue'] / (ep + 1) * 100,
                'avg_loss': loss,
                'speed': speed,
                'elapsed_hours': elapsed / 3600,
                'training_type': 'self_play',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            red_rate = stats['red'] / (ep + 1) * 100
            print(f"[{ep+1}/{target}] 红队: {red_rate:.1f}% | 速度: {speed:.0f}局/秒 | 损失: {loss:.4f}")
    
    # 完成
    print("=" * 60)
    print("训练完成!")
    print(f"红队胜率: {stats['red']/target*100:.1f}%")
    print(f"蓝队胜率: {stats['blue']/target*100:.1f}%")
    torch.save(models[0].state_dict(), '/workspace/projects/self_play_final.pt')
    print("模型已保存: self_play_final.pt")

if __name__ == '__main__':
    main()
