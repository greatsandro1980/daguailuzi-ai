#!/usr/bin/env python3
"""
自我博弈训练 V1
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
from collections import deque

# ============== 游戏环境 ==============
CARD_TYPES = {
    'single': 1, 'pair': 2, 'triple': 3, 'bomb': 4,
    'triple_single': 4, 'triple_pair': 5, 'straight': 5,
    'double_straight': 6, 'airplane': 6, 'four_two': 6
}

class FastGame:
    """高速游戏环境"""
    def __init__(self):
        self.cards = np.zeros(15, dtype=np.int32)
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current_player = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        
    def reset(self):
        """重置游戏"""
        self.cards = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,1,1], dtype=np.int32)
        np.random.shuffle(self.cards)
        
        self.hands = np.zeros((6, 15), dtype=np.int32)
        idx = 0
        for p in range(6):
            for _ in range(9 if p < 4 else 8):
                self.hands[p, self.cards[idx]] += 1
                idx += 1
                
        self.current_player = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        return self.get_state()
    
    def get_state(self):
        """获取状态"""
        hand = self.hands[self.current_player]
        state = np.zeros(105, dtype=np.float32)
        state[:15] = hand / 4.0
        state[15:30] = self.hands[(self.current_player + 1) % 6] / 4.0
        state[30:45] = self.hands[(self.current_player + 2) % 6] / 4.0
        state[45:60] = self.hands[(self.current_player + 3) % 6] / 4.0
        state[60:75] = self.hands[(self.current_player + 4) % 6] / 4.0
        state[75:90] = self.hands[(self.current_player + 5) % 6] / 4.0
        if self.last_play is not None:
            state[90:105] = self.last_play / 4.0
        return state
    
    def get_actions(self):
        """获取合法动作"""
        hand = self.hands[self.current_player]
        actions = []
        
        # 必须出牌
        if self.last_play is None or self.last_player == self.current_player:
            # 出任意牌
            for i in range(15):
                if hand[i] >= 1:
                    action = np.zeros(15, dtype=np.int32)
                    action[i] = 1
                    actions.append(action)
            for i in range(13):
                if hand[i] >= 2:
                    action = np.zeros(15, dtype=np.int32)
                    action[i] = 2
                    actions.append(action)
            for i in range(13):
                if hand[i] >= 3:
                    action = np.zeros(15, dtype=np.int32)
                    action[i] = 3
                    actions.append(action)
            # 炸弹
            for i in range(13):
                if hand[i] >= 4:
                    action = np.zeros(15, dtype=np.int32)
                    action[i] = 4
                    actions.append(action)
            # 顺子
            for start in range(8):
                if all(hand[start+i] >= 1 for i in range(5)):
                    action = np.zeros(15, dtype=np.int32)
                    for i in range(5):
                        action[start+i] = 1
                    actions.append(action)
        else:
            # 必须压过上家
            last = self.last_play
            last_count = last[last > 0][0] if len(last[last > 0]) > 0 else 0
            last_max = np.where(last > 0)[0].max() if len(last[last > 0]) > 0 else -1
            
            # 同类型压
            for i in range(last_max + 1, 15):
                if hand[i] >= last_count:
                    action = np.zeros(15, dtype=np.int32)
                    action[i] = last_count
                    actions.append(action)
            
            # 炸弹压非炸弹
            if last_count < 4:
                for i in range(15):
                    if hand[i] >= 4:
                        action = np.zeros(15, dtype=np.int32)
                        action[i] = 4
                        actions.append(action)
        
        # 过牌
        if self.last_play is not None and self.last_player != self.current_player:
            actions.append(np.zeros(15, dtype=np.int32))
            
        return actions if len(actions) > 0 else [np.zeros(15, dtype=np.int32)]
    
    def step(self, action):
        """执行动作"""
        self.hands[self.current_player] -= action
        self.hands[self.hands < 0] = 0
        
        # 检查是否出完
        if self.hands[self.current_player].sum() == 0:
            self.finished[self.current_player] = True
            self.finish_order.append(self.current_player)
            
            # 检查是否一队都出完
            team = self.current_player % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                # 该队获胜
                winner_team = team
                done = True
                return self.get_state(), winner_team, done
        
        # 更新状态
        if action.sum() > 0:
            self.last_play = action.copy()
            self.last_player = self.current_player
            self.passes = 0
        else:
            self.passes += 1
            
        # 检查是否所有人都过
        if self.passes >= 5:
            self.last_play = None
            self.last_player = -1
            self.passes = 0
            
        # 下一个玩家
        self.current_player = (self.current_player + 1) % 6
        while self.finished[self.current_player]:
            self.current_player = (self.current_player + 1) % 6
            
        return self.get_state(), -1, False

# ============== 神经网络 ==============
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(105, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 15)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value
    
    def get_action(self, state, actions, epsilon=0.1):
        """选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(len(actions))
            
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(state_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        
        # 计算每个动作的概率
        action_probs = []
        for action in actions:
            prob = sum(probs[i] for i in range(15) if action[i] > 0)
            action_probs.append(prob)
            
        if sum(action_probs) > 0:
            return np.argmax(action_probs)
        return np.random.randint(len(actions))

# ============== 训练函数 ==============
def self_play_episode(models, device):
    """自我博弈一局"""
    game = FastGame()
    states = {i: [] for i in range(6)}
    actions_taken = {i: [] for i in range(6)}
    
    state = game.reset()
    done = False
    
    while not done:
        player = game.current_player
        actions = game.get_actions()
        
        # 使用对应玩家的模型
        model = models[player % 2]  # 只用2个模型，红队用models[0]，蓝队用models[1]
        action_idx = model.get_action(state, actions, epsilon=0.15)
        action = actions[action_idx]
        
        states[player].append(state.copy())
        actions_taken[player].append(action_idx)
        
        state, winner_team, done = game.step(action)
    
    # 计算奖励
    rewards = {}
    for player in range(6):
        team = player % 2
        if team == winner_team:
            # 胜利队伍：先出完的奖励更高
            if player in game.finish_order:
                order_bonus = (6 - game.finish_order.index(player)) * 0.1
                rewards[player] = 1.0 + order_bonus
            else:
                rewards[player] = 1.0
        else:
            rewards[player] = -0.5
            
    return states, actions_taken, rewards, winner_team

def train_step(models, optimizers, states, actions, rewards, device):
    """训练一步"""
    total_loss = 0
    
    for player in range(6):
        if len(states[player]) == 0:
            continue
            
        model = models[player % 2]
        optimizer = optimizers[player % 2]
        
        states_t = torch.FloatTensor(np.array(states[player])).to(device)
        rewards_t = torch.FloatTensor([rewards[player]] * len(states[player])).to(device)
        
        logits, values = model(states_t)
        values = values.squeeze(-1)
        
        # Critic loss
        critic_loss = nn.MSELoss()(values, rewards_t)
        
        # Actor loss (policy gradient)
        probs = torch.softmax(logits, dim=-1)
        action_probs = torch.gather(probs, 2, 
            torch.tensor(actions[player]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 15).to(device)
        ).squeeze()
        log_probs = torch.log(action_probs + 1e-8)
        advantage = rewards_t - values.detach()
        actor_loss = -(log_probs * advantage).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / 6

def main():
    print("=" * 60)
    print("自我博弈训练 V1 - 目标50万局")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # 初始化模型（红队、蓝队各一个）
    models = [ActorCritic().to(device) for _ in range(2)]
    optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in models]
    
    # 加载检查点
    start_episode = 0
    checkpoint_path = '/workspace/projects/self_play_checkpoint.json'
    model_path = '/workspace/projects/self_play_model.pt'
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
            start_episode = data.get('episode', 0)
            print(f"从检查点恢复: {start_episode} 局")
            
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        models[0].load_state_dict(state_dict)
        models[1].load_state_dict(state_dict)
        print("加载模型参数")
    
    # 训练统计
    target_episodes = 500000
    stats = {
        'episodes': start_episode,
        'red_wins': 0,
        'blue_wins': 0,
        'losses': [],
        'start_time': time.time(),
        'last_save': time.time()
    }
    
    # 加载已有统计
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
            stats['red_wins'] = data.get('red_wins', 0)
            stats['blue_wins'] = data.get('blue_wins', 0)
    
    print(f"开始训练: {start_episode} -> {target_episodes}")
    print("-" * 60)
    
    save_interval = 5000
    log_interval = 1000
    
    for episode in range(start_episode, target_episodes):
        # 自我博弈
        states, actions, rewards, winner_team = self_play_episode(models, device)
        
        # 统计
        if winner_team == 0:
            stats['red_wins'] += 1
        else:
            stats['blue_wins'] += 1
            
        stats['episodes'] = episode + 1
        
        # 训练
        loss = train_step(models, optimizers, states, actions, rewards, device)
        stats['losses'].append(loss)
        
        # 保存状态
        if (episode + 1) % save_interval == 0:
            elapsed = time.time() - stats['start_time']
            speed = (episode + 1 - start_episode) / elapsed if elapsed > 0 else 0
            
            # 保存模型
            torch.save(models[0].state_dict(), model_path)
            
            # 保存检查点
            checkpoint = {
                'episode': episode + 1,
                'red_wins': stats['red_wins'],
                'blue_wins': stats['blue_wins'],
                'win_rate': stats['red_wins'] / (episode + 1) * 100,
                'speed': speed,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # 更新训练状态（给监控用）
            status = {
                'episode': episode + 1,
                'target': target_episodes,
                'progress': (episode + 1) / target_episodes * 100,
                'red_wins': stats['red_wins'],
                'blue_wins': stats['blue_wins'],
                'red_win_rate': stats['red_wins'] / (episode + 1) * 100,
                'blue_win_rate': stats['blue_wins'] / (episode + 1) * 100,
                'avg_loss': np.mean(stats['losses'][-100:]) if stats['losses'] else 0,
                'speed': speed,
                'elapsed_hours': elapsed / 3600,
                'training_type': 'self_play',
                'timestamp': datetime.now().isoformat()
            }
            
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"[{episode+1}/{target_episodes}] 红队胜率: {status['red_win_rate']:.1f}% "
                  f"速度: {speed:.0f}局/秒 损失: {status['avg_loss']:.3f}")
        
        # 定期打印
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - stats['start_time']
            speed = (episode + 1 - start_episode) / elapsed if elapsed > 0 else 0
            red_rate = stats['red_wins'] / (episode + 1) * 100
            print(f"[{episode+1}] 红队: {red_rate:.1f}% 速度: {speed:.0f}局/秒")
    
    # 训练完成
    print("=" * 60)
    print("训练完成!")
    print(f"总局数: {target_episodes}")
    print(f"红队胜率: {stats['red_wins'] / target_episodes * 100:.1f}%")
    print(f"蓝队胜率: {stats['blue_wins'] / target_episodes * 100:.1f}%")
    
    # 保存最终模型
    torch.save(models[0].state_dict(), '/workspace/projects/self_play_final.pt')
    print("模型已保存: self_play_final.pt")

if __name__ == '__main__':
    main()
