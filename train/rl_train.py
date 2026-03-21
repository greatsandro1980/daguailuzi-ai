#!/usr/bin/env python3
"""
强化学习训练 - 模型 vs 规则Bot
目标：对规则Bot胜率 ≥ 60%

训练方式：
- 红队：模型（待训练）
- 蓝队：规则Bot（固定策略）
- 奖励：胜利 +1，失败 -1
- 算法：策略梯度（Policy Gradient）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime

# ============== 游戏环境 ==============
class Game:
    """简化游戏环境"""
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
        return self.get_state()
    
    def get_state(self):
        """获取状态（90维）"""
        state = np.zeros(90, dtype=np.float32)
        state[:15] = self.hands[self.current] / 4.0
        # 其他玩家牌数
        for i, p in enumerate([(self.current+1)%6, (self.current+3)%6, (self.current+5)%6]):
            state[15+i*5] = self.hands[p].sum() / 9.0
        # 上家出牌
        if self.last_play is not None:
            state[75:90] = self.last_play / 4.0
        return state
    
    def get_actions(self):
        """获取合法动作"""
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
            # 出任意牌
            for i in range(15):
                if hand[i] >= 1:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 1
                    actions.append(('single', i, a))
            for i in range(13):
                if hand[i] >= 2:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 2
                    actions.append(('pair', i, a))
            for i in range(13):
                if hand[i] >= 3:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = 3
                    actions.append(('triple', i, a))
        else:
            # 压牌
            last_val = self.last_play.max()
            last_cnt = int(self.last_play[self.last_play > 0][0]) if len(self.last_play[self.last_play > 0]) > 0 else 0
            
            for i in range(int(last_val) + 1, 15):
                if hand[i] >= last_cnt:
                    a = np.zeros(15, dtype=np.int32)
                    a[i] = last_cnt
                    actions.append(('beat', i, a))
            
            # 过牌
            actions.append(('pass', 0, np.zeros(15, dtype=np.int32)))
        
        return actions if actions else [('pass', 0, np.zeros(15, dtype=np.int32))]
    
    def step(self, action):
        """执行动作"""
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            team = self.current % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return team  # 返回获胜队伍
        
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
        
        return -1  # 继续游戏

# ============== 策略网络 ==============
class PolicyNet(nn.Module):
    """策略网络：输出每个动作的概率"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(90, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, 15)  # 输出15种牌的概率
        
    def forward(self, x):
        x = self.net(x)
        return torch.softmax(self.action_head(x), dim=-1)
    
    def get_action(self, state, actions, epsilon=0.1):
        """选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
        
        with torch.no_grad():
            probs = self(torch.FloatTensor(state))
        
        # 计算每个动作的得分
        scores = []
        for name, card, action in actions:
            score = probs[card].item()
            scores.append(score)
        
        return int(np.argmax(scores))

# ============== 规则Bot ==============
def rule_bot_action(game, actions):
    """规则Bot：小牌优先策略"""
    # 过滤出牌动作（排除过牌）
    valid = [(i, name, card, action) for i, (name, card, action) in enumerate(actions) if action.sum() > 0]
    
    if not valid:
        return len(actions) - 1  # 过牌
    
    # 优先出最小的牌
    best = min(valid, key=lambda x: x[2])  # 按牌面值排序
    return best[0]

# ============== 训练函数 ==============
def play_game(model, epsilon=0.1):
    """玩一局游戏，返回历史和结果"""
    game = Game()
    game.reset()
    
    history = []  # (state, action_idx, actions)
    
    while True:
        player = game.current
        actions = game.get_actions()
        state = game.get_state()
        
        if player % 2 == 0:  # 红队（模型）
            action_idx = model.get_action(state, actions, epsilon)
        else:  # 蓝队（规则Bot）
            action_idx = rule_bot_action(game, actions)
        
        if player % 2 == 0:
            history.append((state.copy(), action_idx, [a[2] for a in actions]))
        
        _, _, action = actions[action_idx]
        result = game.step(action)
        
        if result >= 0:
            # result: 0=红队赢, 1=蓝队赢
            return history, result

def train_episode(model, optimizer, history, winner):
    """训练一局"""
    # 计算奖励：红队赢=+1，蓝队赢=-1
    reward = 1.0 if winner == 0 else -1.0
    
    total_loss = 0
    
    for state, action_idx, actions in history:
        # 前向传播
        probs = model(torch.FloatTensor(state))
        
        # 计算选中动作的概率
        action = actions[action_idx]
        card = np.argmax(action) if action.sum() > 0 else 0
        prob = probs[card]
        
        # 策略梯度损失
        loss = -torch.log(prob + 1e-8) * reward
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(history) if history else 0

def test_model(model, num_games=100):
    """测试模型对规则Bot的胜率"""
    wins = 0
    for _ in range(num_games):
        _, winner = play_game(model, epsilon=0.0)  # 测试时不探索
        if winner == 0:
            wins += 1
    return wins / num_games * 100

def main():
    print("=" * 60)
    print("强化学习训练 - 模型 vs 规则Bot")
    print("目标：胜率 ≥ 60%")
    print("=" * 60)
    
    # 初始化模型
    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 加载检查点
    checkpoint_file = '/workspace/projects/rl_checkpoint.json'
    model_file = '/workspace/projects/rl_model.pt'
    
    start_ep = 0
    best_rate = 0
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            data = json.load(f)
            start_ep = data.get('episode', 0)
            best_rate = data.get('best_rate', 0)
            print(f"从检查点恢复: {start_ep} 局, 最佳胜率: {best_rate:.1f}%")
    
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
        print("加载模型参数")
    
    target_episodes = 100000  # 目标局数
    test_interval = 1000     # 每1000局测试一次
    save_interval = 5000     # 每5000局保存一次
    
    stats = {'wins': 0, 'total': 0}
    start_time = time.time()
    
    print(f"开始训练: {start_ep} -> {target_episodes}")
    print("-" * 60)
    
    for ep in range(start_ep, target_episodes):
        # 玩一局
        history, winner = play_game(model, epsilon=0.2)
        
        # 统计
        stats['total'] += 1
        if winner == 0:
            stats['wins'] += 1
        
        # 训练
        loss = train_episode(model, optimizer, history, winner)
        
        # 定期测试
        if (ep + 1) % test_interval == 0:
            win_rate = test_model(model, 50)
            elapsed = time.time() - start_time
            speed = (ep + 1 - start_ep) / elapsed if elapsed > 0 else 0
            
            print(f"[{ep+1}/{target_episodes}] 胜率: {win_rate:.1f}% | "
                  f"训练胜率: {stats['wins']/stats['total']*100:.1f}% | "
                  f"速度: {speed:.0f}局/秒")
            
            # 更新最佳
            if win_rate > best_rate:
                best_rate = win_rate
                torch.save(model.state_dict(), '/workspace/projects/rl_best.pt')
                print(f"  ✅ 新最佳胜率: {best_rate:.1f}%")
            
            # 达到目标
            if win_rate >= 60:
                print("=" * 60)
                print(f"🎉 达到目标！胜率: {win_rate:.1f}%")
                print("=" * 60)
        
        # 保存检查点
        if (ep + 1) % save_interval == 0:
            torch.save(model.state_dict(), model_file)
            
            checkpoint = {
                'episode': ep + 1,
                'best_rate': best_rate,
                'wins': stats['wins'],
                'total': stats['total'],
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # 更新训练状态
            status = {
                'episode': ep + 1,
                'target': target_episodes,
                'progress': (ep + 1) / target_episodes * 100,
                'win_rate': stats['wins'] / stats['total'] * 100,
                'best_rate': best_rate,
                'speed': speed if 'speed' in dir() else 0,
                'training_type': 'reinforcement_learning',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
    
    # 训练完成
    print("=" * 60)
    print("训练完成!")
    print(f"最佳胜率: {best_rate:.1f}%")
    torch.save(model.state_dict(), '/workspace/projects/rl_final.pt')
    print("模型已保存: rl_final.pt")

if __name__ == '__main__':
    main()
