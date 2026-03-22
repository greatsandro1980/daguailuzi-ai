#!/usr/bin/env python3
"""
强化学习训练 V3 - 完整统计版
统计：即时胜率、累计胜率、得分率
目标：训练 100,000 局
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

# ============== 游戏环境（带完整统计）==============
class FastGame:
    """简化版大怪路子游戏"""
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []  # 记录出完牌的顺序
        
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
            # 首发或获得控制权
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
            # 必须跟牌
            last_val = int(self.last_play.max())
            last_cnt = int(self.last_play[self.last_play > 0][0])
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt:
                    actions.append((i, last_cnt))
            actions.append((-1, 0))  # 过牌
        
        return actions if actions else [(-1, 0)]
    
    def step(self, card, count):
        """执行动作，返回 (游戏是否结束, 胜者队伍)"""
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
        
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        
        # 检查是否出完
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            self.finish_order.append(self.current)
            
            # 检查该队是否全部出完
            team = self.current % 2  # 0=红队, 1=蓝队
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return True, team, self.finish_order
        
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
        
        return False, -1, self.finish_order


def calculate_score(finish_order, winner_team):
    """
    计算得分
    大怪路子规则：
    - 获胜队获得头游（第一个出完）
    - 如果获胜队的队友是最后一名，对手得1分
    - 否则获胜队得1分
    """
    if not finish_order:
        return 0  # 红队得分（正=红队得分，负=蓝队得分）
    
    first_player = finish_order[0]  # 头游
    last_player = finish_order[-1]  # 最后一名
    
    first_team = first_player % 2  # 0=红队, 1=蓝队
    last_team = last_player % 2
    
    # 头游所在队伍获胜
    if first_team == 0:  # 红队获胜
        # 如果最后一名是红队（获胜队的队友），蓝队得分
        if last_team == 0:
            return -1  # 蓝队得分
        else:
            return 1  # 红队得分
    else:  # 蓝队获胜
        # 如果最后一名是蓝队，红队得分
        if last_team == 1:
            return 1  # 红队得分
        else:
            return -1  # 蓝队得分


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
def model_action(model, state, actions, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    
    with torch.no_grad():
        probs = model(torch.FloatTensor(state))
    
    # 根据牌值选择概率最高的有效动作
    scores = []
    for i, (card, count) in enumerate(actions):
        if card >= 0:
            scores.append((i, probs[card].item() * count))  # 大牌加权
        else:
            scores.append((i, 0.01))  # 过牌低概率
    
    return max(scores, key=lambda x: x[1])[0]


def rule_action(actions):
    """规则Bot：小牌优先策略"""
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    if valid:
        # 优先出最小的牌，但尽量出多张
        valid.sort(key=lambda x: (x[1], -x[2]))  # 牌值升序，张数降序
        return valid[0][0]
    return len(actions) - 1  # 过牌


# ============== 训练函数 ==============
def play_game(model, epsilon=0.15):
    """进行一局游戏，返回 (states, actions, 胜者, 得分)"""
    game = FastGame()
    game.reset()
    
    model_states = []
    model_actions = []
    
    while True:
        player = game.current
        actions = game.get_actions()
        state = game.get_state()
        
        if player % 2 == 0:  # 红队用模型
            idx = model_action(model, state, actions, epsilon)
            model_states.append(state)
            model_actions.append(actions[idx][0])
        else:  # 蓝队用规则
            idx = rule_action(actions)
        
        card, count = actions[idx]
        done, winner, finish_order = game.step(card, count)
        
        if done:
            score = calculate_score(finish_order, winner)
            return model_states, model_actions, winner, score


def train_batch(model, optimizer, games_data):
    """批量训练"""
    total_loss = 0
    
    for states, actions, winner, score in games_data:
        # 奖励：胜者+1，败者-0.5，得分额外奖励
        if winner == 0:  # 红队胜
            reward = 1.0 + (0.5 if score > 0 else 0)
        else:  # 蓝队胜
            reward = -0.5 - (0.3 if score < 0 else 0)
        
        for state, action in zip(states, actions):
            probs = model(torch.FloatTensor(state))
            prob = probs[action] if 0 <= action < 15 else probs.mean()
            
            loss = -torch.log(prob + 1e-8) * reward
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss


def main():
    print("=" * 70)
    print("强化学习训练 V3 - 完整统计版")
    print("统计: 即时胜率 | 累计胜率 | 得分率")
    print("目标: 训练 100,000 局")
    print("=" * 70)
    
    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # 加载已有模型
    if os.path.exists('/workspace/projects/rl_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_best.pt', map_location='cpu', weights_only=True))
        print("✅ 加载已有最佳模型")
    
    target = 100000
    batch_size = 50
    test_interval = 500
    save_interval = 2000
    
    # 统计数据
    total_wins = 0  # 累计胜利
    total_games = 0
    total_scores = 0  # 累计得分（正=红队得分）
    recent_wins = deque(maxlen=100)  # 最近100局
    recent_scores = deque(maxlen=100)
    best_rate = 0
    
    start_time = time.time()
    batch_data = []
    
    print(f"\n开始训练: 0 -> {target:,} 局")
    print("-" * 70)
    print(f"{'局数':>10} | {'即时胜率':>8} | {'累计胜率':>8} | {'得分率':>8} | {'最佳':>8} | {'速度':>10}")
    print("-" * 70)
    
    for ep in range(target):
        states, actions, winner, score = play_game(model)
        batch_data.append((states, actions, winner, score))
        
        total_games += 1
        if winner == 0:
            total_wins += 1
            recent_wins.append(1)
        else:
            recent_wins.append(0)
        
        total_scores += score
        recent_scores.append(score)
        
        # 批量训练
        if len(batch_data) >= batch_size:
            train_batch(model, optimizer, batch_data)
            batch_data = []
        
        # 定期输出统计
        if (ep + 1) % test_interval == 0:
            elapsed = time.time() - start_time
            speed = (ep + 1) / elapsed
            
            # 计算各项指标
            instant_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
            total_rate = total_wins / total_games * 100
            score_rate = sum(1 for s in recent_scores if s > 0) / len(recent_scores) * 100 if recent_scores else 0
            
            if instant_rate > best_rate:
                best_rate = instant_rate
                torch.save(model.state_dict(), '/workspace/projects/rl_best.pt')
            
            print(f"{ep+1:>10,} | {instant_rate:>7.1f}% | {total_rate:>7.1f}% | {score_rate:>7.1f}% | {best_rate:>7.1f}% | {speed:>8.0f}局/秒")
            
            # 保存状态
            status = {
                'episode': ep + 1,
                'target': target,
                'progress': (ep + 1) / target * 100,
                'win_rate': round(instant_rate, 1),  # 即时胜率
                'total_win_rate': round(total_rate, 1),  # 累计胜率
                'score_rate': round(score_rate, 1),  # 得分率
                'best_rate': round(best_rate, 1),
                'total_wins': total_wins,
                'total_games': total_games,
                'total_scores': total_scores,
                'speed': round(speed, 1),
                'elapsed_hours': round(elapsed / 3600, 2),
                'training_type': 'reinforcement_learning',
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
        
        # 定期保存模型
        if (ep + 1) % save_interval == 0:
            torch.save(model.state_dict(), '/workspace/projects/rl_model.pt')
    
    # 训练完成
    elapsed = time.time() - start_time
    final_rate = total_wins / total_games * 100
    final_score_rate = sum(1 for s in recent_scores if s > 0) / len(recent_scores) * 100 if recent_scores else 0
    
    print("-" * 70)
    print(f"训练完成！")
    print(f"  累计胜率: {final_rate:.1f}%")
    print(f"  最佳即时胜率: {best_rate:.1f}%")
    print(f"  得分率: {final_score_rate:.1f}%")
    print(f"  总耗时: {elapsed/3600:.2f} 小时")
    print("=" * 70)
    
    torch.save(model.state_dict(), '/workspace/projects/rl_final.pt')
    
    # 最终状态
    status = {
        'episode': target,
        'target': target,
        'progress': 100.0,
        'win_rate': round(sum(recent_wins) / len(recent_wins) * 100, 1) if recent_wins else 0,
        'total_win_rate': round(final_rate, 1),
        'score_rate': round(final_score_rate, 1),
        'best_rate': round(best_rate, 1),
        'total_wins': total_wins,
        'total_games': total_games,
        'status': 'completed',
        'training_type': 'reinforcement_learning',
        'elapsed_hours': round(elapsed / 3600, 2),
        'timestamp': datetime.now().isoformat()
    }
    with open('/workspace/projects/training_status.json', 'w') as f:
        json.dump(status, f, indent=2)


if __name__ == '__main__':
    main()
